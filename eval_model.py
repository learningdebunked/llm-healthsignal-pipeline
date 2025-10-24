import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Reuse preprocessing utilities from model_train
from model_train import (
    normalize,
    bandpass_filter,
    load_physionet_dataset,
    segment_signal_data,
)


def prepare_data(overlap=0.5, seed=42):
    np.random.seed(seed)
    datasets = [
        ("sleep-edf", "slp01"),
        ("mitdb", "100"),
        ("ptbdb", "patient001/s0010_re"),
        ("ptb-xl", "records100/00000"),
        ("challenge-2020", "A00001"),
        ("mimic3wdb", "3000003_0003"),
    ]
    all_segments, all_labels = [], []
    for db, rec in datasets:
        try:
            data = load_physionet_dataset(db, rec)
            signal = normalize(bandpass_filter(data["signal"], fs=data["fs"]))
            X, y = segment_signal_data(signal, data["annotations"], overlap=overlap)
            if len(X) == 0 or len(y) == 0:
                continue
            all_segments.append(X)
            all_labels.append(y)
        except Exception as e:
            print(f"Failed to load {db}/{rec}: {e}")
    X = np.concatenate(all_segments, axis=0)
    y = np.concatenate(all_labels, axis=0)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, le


def build_model(input_timesteps, input_features, n_classes):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(input_timesteps, input_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(n_classes, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def compute_specificity(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    # Compute per-class specificity: TN / (TN + FP)
    specificities = []
    for c in range(n_classes):
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        denom = TN + FP
        spec = TN / denom if denom > 0 else 0.0
        specificities.append(spec)
    return float(np.mean(specificities))


def main():
    seed = 42
    overlap = 0.5
    np.random.seed(seed)

    # Prepare data
    X, y_enc, le = prepare_data(overlap=overlap, seed=seed)
    n_classes = len(np.unique(y_enc))

    # Train/test split
    X_train, X_test, y_train_labels, y_test_labels = train_test_split(
        X, y_enc, test_size=0.2, random_state=seed, stratify=y_enc if n_classes > 1 else None
    )

    # One-hot for training
    from tensorflow.keras.utils import to_categorical

    y_train = to_categorical(y_train_labels, num_classes=n_classes)
    y_test = to_categorical(y_test_labels, num_classes=n_classes)

    # Class weights
    classes = np.arange(n_classes)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_labels)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

    # Build model
    model = build_model(X.shape[1], X.shape[2], n_classes)

    # LR scheduler
    lr_cb = ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)

    # Train
    model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        callbacks=[lr_cb],
        verbose=1,
    )

    # Predict
    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)

    # Metrics
    acc = accuracy_score(y_test_labels, y_pred)
    sens_macro = recall_score(y_test_labels, y_pred, average="macro", zero_division=0)  # Sensitivity
    f1_macro = f1_score(y_test_labels, y_pred, average="macro", zero_division=0)
    spec_macro = compute_specificity(y_test_labels, y_pred, n_classes)

    # AUC (macro, OVR) — requires probability and binarized labels
    try:
        y_test_bin = label_binarize(y_test_labels, classes=classes)
        auc_macro = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
    except Exception as e:
        print(f"AUC computation failed: {e}")
        auc_macro = float("nan")

    print("\n==== Evaluation Metrics (Test Set) ====")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Sensitivity:  {sens_macro:.4f} (macro recall)")
    print(f"Specificity:  {spec_macro:.4f} (macro)")
    print(f"F1-score:     {f1_macro:.4f} (macro)")
    print(f"ROC AUC:      {auc_macro:.4f} (macro OVR)")


if __name__ == "__main__":
    main()
