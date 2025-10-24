import os
import numpy as np
import wfdb
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau


# Predefined templates to generate prompts for the LLM
prebuilt_prompts = {
    "explain_ecg": "Explain this ECG result: {}",
    "next_steps": "Given the diagnosis '{}', what are the recommended next steps?",
    "generate_summary": "Summarize the patient data: {}",
    "abnormal_eeg": "What might an abnormal EEG pattern like '{}' indicate?",
    "health_advice": "Suggest lifestyle advice for a patient with {}."
}

def get_prompt(template_name, context):
    """
    Retrieves a prompt template and formats it using context for LLM use.
    """
    template = prebuilt_prompts.get(template_name, "{}")
    return template.format(context)

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=250.0, order=5):
    """
    Applies a bandpass Butterworth filter to remove noise from biomedical signals.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal, axis=0)

def normalize(signal):
    """
    Z-score normalization per channel: (x - mean) / std
    """
    eps = 1e-8
    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True)
    return (signal - mean) / (std + eps)

def load_physionet_dataset(database_name, record_id):
    """
    Downloads and loads a dataset record from PhysioNet. Returns signal and annotations.
    """
    wfdb.dl_database(database_name, dl_dir=database_name)
    record = wfdb.rdrecord(os.path.join(database_name, record_id))
    annotation = None
    try:
        annotation = wfdb.rdann(os.path.join(database_name, record_id), 'atr')
    except:
        try:
            annotation = wfdb.rdann(os.path.join(database_name, record_id), 'hypnogram')
        except:
            pass
    annotations = annotation.symbol if annotation else []
    return {
        'signal': record.p_signal,
        'annotations': annotations,
        'fs': record.fs,
        'fields': record.sig_name
    }

def segment_signal_data(signal, annotations, window_size=3000, overlap=0.5):
    """
    Splits a long signal into fixed-size windows with overlap and assigns labels per segment index.
    """
    segments, labels = [], []
    step_size = max(1, int(window_size * (1 - overlap)))
    seg_idx = 0
    for i in range(0, len(signal) - window_size + 1, step_size):
        segments.append(signal[i:i+window_size])
        if seg_idx < len(annotations):
            labels.append(annotations[seg_idx])
        seg_idx += 1
    return np.array(segments), np.array(labels)

def train_combined_model(
    augment_prob=0.5,
    noise_factor=0.05,
    scale_range=(0.8, 1.2),
    seed=42,
    overlap=0.5,
):
    """
    Trains an LSTM classifier using multiple PhysioNet datasets.
    """
    # Reproducibility
    np.random.seed(seed)
    datasets = [
        ('sleep-edf', 'slp01'),
        ('mitdb', '100'),
        ('ptbdb', 'patient001/s0010_re'),
        ('ptb-xl', 'records100/00000'),
        ('challenge-2020', 'A00001'),
        ('mimic3wdb', '3000003_0003')
    ]
    all_segments, all_labels = [], []
    for db, rec in datasets:
        try:
            data = load_physionet_dataset(db, rec)
            signal = normalize(bandpass_filter(data['signal'], fs=data['fs']))
            X, y = segment_signal_data(signal, data['annotations'], overlap=overlap)
            all_segments.append(X)
            all_labels.append(y)
        except Exception as e:
            print(f"Failed to load {db}/{rec}: {e}")
    X = np.concatenate(all_segments, axis=0)
    y = np.concatenate(all_labels, axis=0)
    le = LabelEncoder()
    y_enc = to_categorical(le.fit_transform(y))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42
    )

    # Data augmentation: training set only
    if augment_prob and noise_factor and scale_range:
        low_s, high_s = scale_range
        for i in range(X_train.shape[0]):
            if np.random.rand() < float(augment_prob):
                seg = X_train[i]
                noise = np.random.normal(0, float(noise_factor), seg.shape)
                scale = np.random.uniform(float(low_s), float(high_s))
                X_train[i] = (seg + noise) * scale

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_enc.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Class weights for imbalanced classes
    y_train_labels = np.argmax(y_train, axis=1)
    classes = np.arange(y_enc.shape[1])
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_labels)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

    # Learning rate scheduler
    lr_cb = ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)

    model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        callbacks=[lr_cb],
    )
    return model

# Trains the model at runtime (can be moved to a startup script)
trained_health_model = train_combined_model()

