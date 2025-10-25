import os
import numpy as np
import wfdb
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam


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

def augment_signal(signal, noise_factor=0.05, scale_range=(0.8, 1.2), max_shift_ratio=0.1):
    """
    Applies data augmentation to biomedical signals as specified in paper Section III.C.1.
    
    Implements three augmentation techniques:
    1. Temporal shifting: Shifts signal along time axis
    2. Additive Gaussian noise: Adds random noise to simulate artifacts
    3. Amplitude scaling: Randomly scales signal amplitude
    
    Args:
        signal: Input signal array of shape (time_steps, channels)
        noise_factor: Standard deviation of Gaussian noise
        scale_range: Tuple (min_scale, max_scale) for amplitude scaling
        max_shift_ratio: Maximum temporal shift as ratio of signal length
    
    Returns:
        Augmented signal with same shape as input
    """
    augmented = signal.copy()
    
    # 1. Temporal shifting
    max_shift = int(signal.shape[0] * max_shift_ratio)
    shift_amount = np.random.randint(-max_shift, max_shift + 1)
    if shift_amount != 0:
        augmented = np.roll(augmented, shift_amount, axis=0)
    
    # 2. Additive Gaussian noise
    noise = np.random.normal(0, noise_factor, augmented.shape)
    augmented = augmented + noise
    
    # 3. Amplitude scaling
    low_scale, high_scale = scale_range
    scale = np.random.uniform(low_scale, high_scale)
    augmented = augmented * scale
    
    return augmented

class MetricsCallback(Callback):
    """
    Custom callback to compute comprehensive evaluation metrics during training.
    Implements metrics from paper Section VI.B:
    - Accuracy
    - Sensitivity (Recall)
    - Specificity
    - F1-Score
    - AUC (Area Under ROC Curve)
    """
    def __init__(self, validation_data, n_classes):
        super().__init__()
        self.validation_data = validation_data
        self.n_classes = n_classes
        self.history = {
            'val_sensitivity': [],
            'val_specificity': [],
            'val_f1': [],
            'val_auc': []
        }
    
    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        
        # Get predictions
        y_pred_proba = self.model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_val, axis=1)
        
        # 1. Accuracy (already in logs)
        accuracy = accuracy_score(y_true, y_pred)
        
        # 2. Sensitivity (Recall) - macro average
        sensitivity = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # 3. Specificity - computed per class then averaged
        specificity = self._compute_specificity(y_true, y_pred)
        
        # 4. F1-Score - macro average
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # 5. AUC - macro OVR (One-vs-Rest)
        try:
            auc = roc_auc_score(y_val, y_pred_proba, average='macro', multi_class='ovr')
        except ValueError:
            auc = 0.0  # Handle cases with missing classes
        
        # Store in history
        self.history['val_sensitivity'].append(float(sensitivity))
        self.history['val_specificity'].append(float(specificity))
        self.history['val_f1'].append(float(f1))
        self.history['val_auc'].append(float(auc))
        
        # Update logs for display
        logs['val_sensitivity'] = sensitivity
        logs['val_specificity'] = specificity
        logs['val_f1'] = f1
        logs['val_auc'] = auc
        
        # Print metrics
        print(f"\n  Validation Metrics:")
        print(f"    Accuracy:     {accuracy:.4f}")
        print(f"    Sensitivity:  {sensitivity:.4f} (macro recall)")
        print(f"    Specificity:  {specificity:.4f} (macro)")
        print(f"    F1-score:     {f1:.4f} (macro)")
        print(f"    ROC AUC:      {auc:.4f} (macro OVR)")
    
    def _compute_specificity(self, y_true, y_pred):
        """
        Compute specificity per class and return macro average.
        Specificity = TN / (TN + FP)
        """
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.n_classes)))
        specificities = []
        
        for c in range(self.n_classes):
            TP = cm[c, c]
            FN = cm[c, :].sum() - TP
            FP = cm[:, c].sum() - TP
            TN = cm.sum() - (TP + FP + FN)
            denom = TN + FP
            spec = TN / denom if denom > 0 else 0.0
            specificities.append(spec)
        
        return float(np.mean(specificities))

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
    # Implements temporal shifting, amplitude scaling, and additive Gaussian noise
    # as specified in paper Section III.C.1
    if augment_prob and noise_factor and scale_range:
        print(f"Applying data augmentation to {X_train.shape[0]} training samples...")
        augmented_count = 0
        for i in range(X_train.shape[0]):
            if np.random.rand() < float(augment_prob):
                X_train[i] = augment_signal(
                    X_train[i],
                    noise_factor=noise_factor,
                    scale_range=scale_range,
                    max_shift_ratio=0.1
                )
                augmented_count += 1
        print(f"✓ Augmented {augmented_count}/{X_train.shape[0]} samples ({augmented_count/X_train.shape[0]*100:.1f}%)")

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_enc.shape[1], activation='softmax'))
    
    # Configure Adam optimizer with paper-specified parameters (Section III.C.3)
    optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # Class weights for imbalanced classes
    y_train_labels = np.argmax(y_train, axis=1)
    classes = np.arange(y_enc.shape[1])
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_labels)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

    # Learning rate scheduler
    lr_cb = ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)
    
    # Comprehensive metrics callback (Section VI.B)
    metrics_cb = MetricsCallback(
        validation_data=(X_test, y_test),
        n_classes=y_enc.shape[1]
    )

    print("\n" + "="*60)
    print("Starting training with comprehensive evaluation metrics")
    print("Metrics computed per epoch: Accuracy, Sensitivity, Specificity, F1, AUC")
    print("="*60 + "\n")

    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        callbacks=[lr_cb, metrics_cb],
        verbose=1
    )
    
    # Print final metrics summary
    print("\n" + "="*60)
    print("FINAL EVALUATION METRICS (Test Set)")
    print("="*60)
    if metrics_cb.history['val_sensitivity']:
        final_idx = -1
        print(f"Accuracy:     {history.history['val_accuracy'][final_idx]:.4f}")
        print(f"Sensitivity:  {metrics_cb.history['val_sensitivity'][final_idx]:.4f} (macro recall)")
        print(f"Specificity:  {metrics_cb.history['val_specificity'][final_idx]:.4f} (macro)")
        print(f"F1-score:     {metrics_cb.history['val_f1'][final_idx]:.4f} (macro)")
        print(f"ROC AUC:      {metrics_cb.history['val_auc'][final_idx]:.4f} (macro OVR)")
    print("="*60 + "\n")
    
    # Attach metrics history to model for later access
    model.metrics_history = metrics_cb.history
    
    return model

# Trains the model at runtime (can be moved to a startup script)
trained_health_model = train_combined_model()

