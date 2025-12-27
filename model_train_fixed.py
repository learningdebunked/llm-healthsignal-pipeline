"""
Fixed Model Training with Patient-Level Splitting
Addresses data leakage concerns by implementing strict patient-level separation
"""
import os
import json
import numpy as np
import wfdb
from scipy.signal import butter, lfilter
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


# ============================================================================
# PATIENT-LEVEL SPLITTING (LEAKAGE PREVENTION)
# ============================================================================

def extract_patient_id(database_name, record_id):
    """
    Extract patient identifier from record ID for each dataset.
    Ensures consistent patient-level grouping.
    """
    patient_id_map = {
        'mitdb': lambda r: r,  # Record number is patient ID
        'ptbdb': lambda r: r.split('/')[0],  # Patient folder
        'ptb-xl': lambda r: r.split('/')[0],  # Patient folder
        'challenge-2020': lambda r: r,  # Record is patient
        'mimic3wdb': lambda r: r.split('_')[0],  # Subject ID
        'sleep-edf': lambda r: r.split('_')[0] if '_' in r else r[:5],  # Subject ID
    }
    
    extractor = patient_id_map.get(database_name, lambda r: r)
    return f"{database_name}_{extractor(record_id)}"


def patient_level_split(records_data, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split data at patient/record level to prevent leakage.
    
    Critical for preventing data leakage:
    - All segments from a patient go to ONE split only
    - No patient appears in multiple splits
    - Maintains temporal integrity
    
    Args:
        records_data: List of dicts with 'patient_id', 'segments', 'labels', 'database'
        train_ratio: Training set proportion (default: 0.70)
        val_ratio: Validation set proportion (default: 0.15)
        test_ratio: Test set proportion (default: 0.15)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        train_data, val_data, test_data, split_info
    """
    np.random.seed(seed)
    
    # Group records by patient
    patient_records = {}
    for record in records_data:
        pid = record['patient_id']
        if pid not in patient_records:
            patient_records[pid] = []
        patient_records[pid].append(record)
    
    # Get unique patient IDs and shuffle
    unique_patients = list(patient_records.keys())
    np.random.shuffle(unique_patients)
    
    # Calculate split indices
    n_patients = len(unique_patients)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    # Split patient IDs
    train_patients = unique_patients[:n_train]
    val_patients = unique_patients[n_train:n_train + n_val]
    test_patients = unique_patients[n_train + n_val:]
    
    # Assign all records from each patient to appropriate split
    train_data = []
    val_data = []
    test_data = []
    
    for pid in train_patients:
        train_data.extend(patient_records[pid])
    for pid in val_patients:
        val_data.extend(patient_records[pid])
    for pid in test_patients:
        test_data.extend(patient_records[pid])
    
    # Create split info for documentation
    split_info = {
        'total_patients': n_patients,
        'train_patients': len(train_patients),
        'val_patients': len(val_patients),
        'test_patients': len(test_patients),
        'train_patient_ids': train_patients,
        'val_patient_ids': val_patients,
        'test_patient_ids': test_patients,
        'train_records': len(train_data),
        'val_records': len(val_data),
        'test_records': len(test_data),
        'random_seed': seed,
        'split_ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        }
    }
    
    return train_data, val_data, test_data, split_info


def verify_no_leakage(train_data, val_data, test_data):
    """
    Verify that no patient appears in multiple splits.
    Critical quality check for data integrity.
    """
    train_patients = set(r['patient_id'] for r in train_data)
    val_patients = set(r['patient_id'] for r in val_data)
    test_patients = set(r['patient_id'] for r in test_data)
    
    # Check for overlaps
    train_val_overlap = train_patients & val_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap = val_patients & test_patients
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise ValueError(
            f"DATA LEAKAGE DETECTED!\n"
            f"Train-Val overlap: {len(train_val_overlap)} patients\n"
            f"Train-Test overlap: {len(train_test_overlap)} patients\n"
            f"Val-Test overlap: {len(val_test_overlap)} patients"
        )
    
    print("✓ No data leakage detected - all patients are in exactly one split")
    return True


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def bandpass_filter(signal, database_name, fs, order=5):
    """
    Applies modality-specific bandpass Butterworth filter to remove noise from biomedical signals.

    Implements modality-adaptive preprocessing:
    - ECG signals (MIT-BIH, MIMIC-III): 0.5-50 Hz to preserve QRS complexes and remove baseline wander
    - EEG signals (Sleep-EDF): 0.5-30 Hz to preserve sleep-related frequency bands (delta, theta, alpha, beta)

    Args:
        signal: Input signal array of shape (n_samples, n_channels)
        database_name: Dataset identifier for modality detection
        fs: Sampling frequency in Hz
        order: Filter order (default: 5)

    Returns:
        Filtered signal with same shape as input
    """
    # Modality-specific filter parameters
    filter_params = {
        'mitdb':      {'low': 0.5, 'high': 50},   # ECG: preserve QRS, remove baseline wander
        'mimic3wdb':  {'low': 0.5, 'high': 50},   # ECG: standard cardiac filtering
        'sleep-edf':  {'low': 0.5, 'high': 30},   # EEG: preserve sleep frequency bands
    }

    # Get parameters for this dataset (fallback to generic 0.5-40 Hz)
    params = filter_params.get(database_name, {'low': 0.5, 'high': 40})
    lowcut, highcut = params['low'], params['high']

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal, axis=0)


def normalize(signal):
    """Z-score normalization: (x - mean) / std"""
    eps = 1e-8
    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True)
    return (signal - mean) / (std + eps)


def load_physionet_dataset(database_name, record_id):
    """Downloads and loads a dataset record from PhysioNet."""
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
    Splits signal into fixed-size windows with overlap.
    
    Note: Overlap is acceptable WITHIN a patient's data, but patients
    must be separated at the split level to prevent leakage.
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
    Data augmentation (applied to training set only).
    Implements three techniques from paper Section III.C.1:
    1. Temporal shifting
    2. Additive Gaussian noise
    3. Amplitude scaling
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


# ============================================================================
# METRICS CALLBACK
# ============================================================================

class MetricsCallback(Callback):
    """
    Computes comprehensive evaluation metrics during training.
    Implements metrics from paper Section VI.B.
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
        
        y_pred_proba = self.model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_val, axis=1)
        
        accuracy = accuracy_score(y_true, y_pred)
        sensitivity = recall_score(y_true, y_pred, average='macro', zero_division=0)
        specificity = self._compute_specificity(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        try:
            auc = roc_auc_score(y_val, y_pred_proba, average='macro', multi_class='ovr')
        except ValueError:
            auc = 0.0
        
        self.history['val_sensitivity'].append(float(sensitivity))
        self.history['val_specificity'].append(float(specificity))
        self.history['val_f1'].append(float(f1))
        self.history['val_auc'].append(float(auc))
        
        logs['val_sensitivity'] = sensitivity
        logs['val_specificity'] = specificity
        logs['val_f1'] = f1
        logs['val_auc'] = auc
        
        print(f"\n  Validation Metrics:")
        print(f"    Accuracy:     {accuracy:.4f}")
        print(f"    Sensitivity:  {sensitivity:.4f}")
        print(f"    Specificity:  {specificity:.4f}")
        print(f"    F1-score:     {f1:.4f}")
        print(f"    ROC AUC:      {auc:.4f}")
    
    def _compute_specificity(self, y_true, y_pred):
        """Compute specificity per class and return macro average."""
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


# ============================================================================
# MAIN TRAINING FUNCTION (FIXED)
# ============================================================================

def train_combined_model_fixed(
    augment_prob=0.5,
    noise_factor=0.05,
    scale_range=(0.8, 1.2),
    seed=42,
    overlap=0.5,
    save_splits=True
):
    """
    Trains LSTM classifier with PROPER patient-level splitting.
    
    Key improvements:
    1. Patient-level splitting (no leakage)
    2. Separate validation set (not using test for validation)
    3. Documented splits for reproducibility
    4. Verification of no patient overlap
    """
    np.random.seed(seed)
    
    # Dataset configuration
    datasets = [
        ('mitdb', '100'),
        ('mitdb', '101'),
        ('mitdb', '102'),
        ('ptbdb', 'patient001/s0010_re'),
        ('ptbdb', 'patient002/s0015_re'),
        # Add more records as needed
    ]
    
    print("\n" + "="*70)
    print("LOADING DATA WITH PATIENT-LEVEL TRACKING")
    print("="*70)
    
    # Load all records with patient tracking
    all_records = []
    for db, rec in datasets:
        try:
            print(f"Loading {db}/{rec}...")
            data = load_physionet_dataset(db, rec)
            signal = normalize(bandpass_filter(data['signal'], db, data['fs']))
            segments, labels = segment_signal_data(signal, data['annotations'], overlap=overlap)
            
            # Extract patient ID
            patient_id = extract_patient_id(db, rec)
            
            # Store record with patient tracking
            all_records.append({
                'patient_id': patient_id,
                'database': db,
                'record_id': rec,
                'segments': segments,
                'labels': labels
            })
            print(f"  ✓ Loaded {len(segments)} segments from patient {patient_id}")
            
        except Exception as e:
            print(f"  ✗ Failed to load {db}/{rec}: {e}")
    
    print(f"\n✓ Total records loaded: {len(all_records)}")
    print(f"✓ Unique patients: {len(set(r['patient_id'] for r in all_records))}")
    
    # CRITICAL: Patient-level split
    print("\n" + "="*70)
    print("PERFORMING PATIENT-LEVEL SPLIT (LEAKAGE PREVENTION)")
    print("="*70)
    
    train_records, val_records, test_records, split_info = patient_level_split(
        all_records,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed
    )
    
    print(f"\nSplit Summary:")
    print(f"  Train: {split_info['train_patients']} patients, {split_info['train_records']} records")
    print(f"  Val:   {split_info['val_patients']} patients, {split_info['val_records']} records")
    print(f"  Test:  {split_info['test_patients']} patients, {split_info['test_records']} records")
    
    # Verify no leakage
    verify_no_leakage(train_records, val_records, test_records)
    
    # Save split information for reproducibility
    if save_splits:
        os.makedirs('splits', exist_ok=True)
        with open('splits/patient_splits.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        print("\n✓ Split information saved to splits/patient_splits.json")
    
    # Combine segments from all records in each split
    print("\n" + "="*70)
    print("PREPARING TRAINING DATA")
    print("="*70)
    
    X_train = np.concatenate([r['segments'] for r in train_records], axis=0)
    y_train = np.concatenate([r['labels'] for r in train_records], axis=0)
    
    X_val = np.concatenate([r['segments'] for r in val_records], axis=0)
    y_val = np.concatenate([r['labels'] for r in val_records], axis=0)
    
    X_test = np.concatenate([r['segments'] for r in test_records], axis=0)
    y_test = np.concatenate([r['labels'] for r in test_records], axis=0)
    
    print(f"Train: {X_train.shape[0]} segments")
    print(f"Val:   {X_val.shape[0]} segments")
    print(f"Test:  {X_test.shape[0]} segments")
    
    # Encode labels
    le = LabelEncoder()
    all_labels = np.concatenate([y_train, y_val, y_test])
    le.fit(all_labels)
    
    y_train_enc = to_categorical(le.transform(y_train))
    y_val_enc = to_categorical(le.transform(y_val))
    y_test_enc = to_categorical(le.transform(y_test))
    
    # Data augmentation (training set only)
    if augment_prob > 0:
        print(f"\nApplying data augmentation to training set...")
        augmented_count = 0
        for i in range(X_train.shape[0]):
            if np.random.rand() < augment_prob:
                X_train[i] = augment_signal(
                    X_train[i],
                    noise_factor=noise_factor,
                    scale_range=scale_range,
                    max_shift_ratio=0.1
                )
                augmented_count += 1
        print(f"✓ Augmented {augmented_count}/{X_train.shape[0]} samples")
    
    # Build model
    print("\n" + "="*70)
    print("BUILDING MODEL")
    print("="*70)
    
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(y_train_enc.shape[1], activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Class weights
    y_train_labels = np.argmax(y_train_enc, axis=1)
    classes = np.arange(y_train_enc.shape[1])
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_labels)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}
    
    # Callbacks
    lr_cb = ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)
    metrics_cb = MetricsCallback(
        validation_data=(X_val, y_val_enc),  # Using proper validation set
        n_classes=y_train_enc.shape[1]
    )
    
    # Train
    print("\n" + "="*70)
    print("TRAINING MODEL (with proper validation set)")
    print("="*70)
    
    history = model.fit(
        X_train,
        y_train_enc,
        epochs=5,
        batch_size=32,
        validation_data=(X_val, y_val_enc),  # Proper validation
        class_weight=class_weight_dict,
        callbacks=[lr_cb, metrics_cb],
        verbose=1
    )
    
    # Final evaluation on TEST set (used only once)
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET (No Leakage)")
    print("="*70)
    
    y_test_pred_proba = model.predict(X_test, verbose=0)
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)
    y_test_true = np.argmax(y_test_enc, axis=1)
    
    test_accuracy = accuracy_score(y_test_true, y_test_pred)
    test_sensitivity = recall_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    test_f1 = f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    
    try:
        test_auc = roc_auc_score(y_test_enc, y_test_pred_proba, average='macro', multi_class='ovr')
    except ValueError:
        test_auc = 0.0
    
    # Compute specificity
    cm = confusion_matrix(y_test_true, y_test_pred)
    specificities = []
    for c in range(y_train_enc.shape[1]):
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        denom = TN + FP
        spec = TN / denom if denom > 0 else 0.0
        specificities.append(spec)
    test_specificity = float(np.mean(specificities))
    
    print(f"\nTest Set Performance (True Generalization):")
    print(f"  Accuracy:     {test_accuracy:.4f}")
    print(f"  Sensitivity:  {test_sensitivity:.4f}")
    print(f"  Specificity:  {test_specificity:.4f}")
    print(f"  F1-score:     {test_f1:.4f}")
    print(f"  ROC AUC:      {test_auc:.4f}")
    print("="*70)
    
    # Save results
    results = {
        'split_info': split_info,
        'test_metrics': {
            'accuracy': float(test_accuracy),
            'sensitivity': float(test_sensitivity),
            'specificity': float(test_specificity),
            'f1_score': float(test_f1),
            'roc_auc': float(test_auc)
        },
        'training_history': {
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'val_sensitivity': metrics_cb.history['val_sensitivity'],
            'val_specificity': metrics_cb.history['val_specificity'],
            'val_f1': metrics_cb.history['val_f1'],
            'val_auc': metrics_cb.history['val_auc']
        }
    }
    
    with open('splits/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved to splits/training_results.json")
    print("✓ Training complete with proper patient-level splitting!")
    
    return model, results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FIXED MODEL TRAINING - PATIENT-LEVEL SPLITTING")
    print("Addresses data leakage concerns from manuscript review")
    print("="*70)
    
    model, results = train_combined_model_fixed()
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    print("✓ Patient-level splitting implemented")
    print("✓ No data leakage verified")
    print("✓ Separate validation and test sets")
    print("✓ Reproducible splits documented")
    print("✓ Results saved for manuscript reporting")
