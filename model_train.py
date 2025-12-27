import os
import numpy as np
import wfdb
from scipy.signal import butter, lfilter
# Note: train_test_split removed - now using patient_level_split to prevent data leakage
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

def bandpass_filter(signal, database_name, fs, order=5):
    """
    Applies modality-specific bandpass Butterworth filter to remove noise from biomedical signals.

    Implements modality-adaptive preprocessing as described in paper Section III.A.1:
    - ECG signals (MIT-BIH, MIMIC-III): 0.5-50 Hz to preserve QRS complexes and remove baseline wander
    - EEG signals (Sleep-EDF): 0.5-30 Hz to preserve sleep-related frequency bands (delta, theta, alpha, beta)

    Args:
        signal: Input signal array of shape (n_samples, n_channels)
        database_name: Dataset identifier for modality detection
        fs: Sampling frequency in Hz
        order: Filter order (default: 5)

    Returns:
        Filtered signal with same shape as input

    Raises:
        ValueError: If sampling rate is too low for filter parameters
        AssertionError: If signal contains NaN or infinite values
    """
    # Input validation
    assert not np.any(np.isnan(signal)), "Signal contains NaN values"
    assert not np.any(np.isinf(signal)), "Signal contains infinite values"
    assert fs > 0, f"Sampling frequency must be positive, got {fs}"

    # Modality-specific filter parameters (Table I in paper)
    filter_params = {
        'mitdb':      {'low': 0.5, 'high': 50},   # ECG: preserve QRS, remove baseline wander
        'mimic3wdb':  {'low': 0.5, 'high': 50},   # ECG: standard cardiac filtering
        'sleep-edf':  {'low': 0.5, 'high': 30},   # EEG: preserve sleep frequency bands (0.5-30 Hz)
    }

    # Get parameters for this dataset (fallback to generic 0.5-40 Hz)
    params = filter_params.get(database_name, {'low': 0.5, 'high': 40})
    lowcut, highcut = params['low'], params['high']

    # Validate filter parameters against Nyquist frequency
    nyq = 0.5 * fs
    if highcut >= nyq:
        raise ValueError(
            f"High cutoff frequency ({highcut} Hz) must be below Nyquist frequency ({nyq} Hz). "
            f"Sampling rate {fs} Hz is insufficient for {database_name} filtering."
        )

    print(f"  Applying {lowcut}-{highcut} Hz bandpass filter for {database_name} (fs={fs} Hz)")

    # Design and apply Butterworth bandpass filter
    low = lowcut / nyq
    high = highcut / nyq

    # Sanity check: normalized frequencies must be in (0, 1)
    assert 0 < low < 1, f"Normalized low frequency {low} out of range (0, 1)"
    assert 0 < high < 1, f"Normalized high frequency {high} out of range (0, 1)"
    assert low < high, f"Low cutoff {lowcut} must be less than high cutoff {highcut}"

    b, a = butter(order, [low, high], btype='band')
    filtered = lfilter(b, a, signal, axis=0)

    # Verify output quality
    assert not np.any(np.isnan(filtered)), "Filtering produced NaN values"
    assert not np.any(np.isinf(filtered)), "Filtering produced infinite values"

    return filtered

def normalize(signal):
    """
    Per-segment z-score normalization: (x - mean) / std

    IMPORTANT: Normalization is applied independently to each signal segment
    (NOT across the entire recording). This ensures consistent amplitude scaling
    while preserving segment-specific characteristics.

    Args:
        signal: Input signal segment of shape (n_samples, n_channels)

    Returns:
        Normalized signal segment with zero mean and unit variance

    Raises:
        AssertionError: If signal contains NaN/inf or has zero std deviation
    """
    # Input validation
    assert not np.any(np.isnan(signal)), "Input signal contains NaN values"
    assert not np.any(np.isinf(signal)), "Input signal contains infinite values"
    assert signal.size > 0, "Input signal is empty"

    eps = 1e-8  # Numerical stability epsilon
    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True)

    # Verify std is not zero (would indicate constant signal)
    if np.any(std < eps):
        print(f"  Warning: Signal has near-zero std deviation ({std.min():.2e}). "
              f"This may indicate a flat/constant signal segment.")

    normalized = (signal - mean) / (std + eps)

    # Verify output quality: normalized signal should have ~0 mean and ~1 std
    output_mean = np.mean(normalized)
    output_std = np.std(normalized)
    assert abs(output_mean) < 0.01, f"Normalization failed: mean={output_mean:.4f} (expected ~0)"
    assert 0.95 < output_std < 1.05, f"Normalization failed: std={output_std:.4f} (expected ~1)"
    assert not np.any(np.isnan(normalized)), "Normalization produced NaN values"
    assert not np.any(np.isinf(normalized)), "Normalization produced infinite values"

    return normalized

def select_lead(signal, database_name, lead_config=None):
    """
    Select appropriate lead from multi-lead signal for consistent processing.
    
    Strategy: Use clinically relevant primary lead for each dataset type.
    This ensures consistent input dimensionality and uses standard leads.
    
    Args:
        signal: Array of shape (n_samples, n_leads) or (n_samples,)
        database_name: Name of dataset
        lead_config: Optional dict specifying lead indices
    
    Returns:
        Array of shape (n_samples, 1) - single selected lead
    """
    if lead_config is None:
        # Default lead selection per dataset (clinically relevant)
        lead_config = {
            'mitdb': 0,        # MLII (modified limb lead II) - standard for arrhythmia
            'ptbdb': 1,        # Lead II - standard limb lead
            'ptb-xl': 1,       # Lead II - standard limb lead
            'challenge-2020': 1,  # Lead II
            'mimic3wdb': 0,    # First available (variable in ICU)
            'sleep-edf': 0,    # Single EEG channel (Fpz-Cz)
        }
    
    lead_idx = lead_config.get(database_name, 0)
    
    # Handle single-lead data (already 1D or 2D with 1 lead)
    if signal.ndim == 1:
        return signal.reshape(-1, 1)
    
    if signal.shape[1] == 1:
        return signal
    
    # Select specified lead from multi-lead signal
    if signal.shape[1] > lead_idx:
        return signal[:, lead_idx:lead_idx+1]
    else:
        # Fallback to first lead if specified lead not available
        print(f"Warning: Lead {lead_idx} not available, using lead 0")
        return signal[:, 0:1]


def load_physionet_dataset(database_name, record_id, lead_selection=True):
    """
    Downloads and loads a dataset record from PhysioNet with lead selection.
    
    Args:
        database_name: Name of PhysioNet database
        record_id: Record identifier
        lead_selection: If True, select primary lead; if False, use all leads
    
    Returns:
        Dict with signal (selected lead), annotations, fs, and metadata
    """
    wfdb.dl_database(database_name, dl_dir=database_name)
    record = wfdb.rdrecord(os.path.join(database_name, record_id))
    
    # Load annotations
    annotation = None
    try:
        annotation = wfdb.rdann(os.path.join(database_name, record_id), 'atr')
    except:
        try:
            annotation = wfdb.rdann(os.path.join(database_name, record_id), 'hypnogram')
        except:
            pass
    
    annotations = annotation.symbol if annotation else []
    
    # Get original signal
    signal = record.p_signal
    n_leads_original = signal.shape[1] if signal.ndim > 1 else 1
    
    # Select lead if requested
    if lead_selection:
        signal = select_lead(signal, database_name)
    
    return {
        'signal': signal,
        'annotations': annotations,
        'fs': record.fs,
        'fields': record.sig_name,
        'n_leads_original': n_leads_original,
        'n_leads_used': signal.shape[1] if signal.ndim > 1 else 1,
        'lead_selection': 'primary' if lead_selection else 'all'
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

def extract_patient_id(database_name, record_id):
    """
    Extract patient identifier from record ID for each dataset.
    Ensures consistent patient-level grouping to prevent data leakage.
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
    Split data at patient/record level to prevent data leakage.

    CRITICAL: All segments from a patient go to ONE split only.
    This prevents the model from seeing similar patterns in both training and test sets.

    Args:
        records_data: List of record dicts, each containing 'patient_id', 'X', 'y'
        train_ratio: Proportion of patients for training (default: 0.70)
        val_ratio: Proportion of patients for validation (default: 0.15)
        test_ratio: Proportion of patients for testing (default: 0.15)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        tuple: (train_data, val_data, test_data) as lists of records

    Raises:
        ValueError: If ratios don't sum to 1.0 or if insufficient patients
        AssertionError: If any patient appears in multiple splits (data leakage)
    """
    # Validate split ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    assert abs(total_ratio - 1.0) < 1e-6, f"Split ratios must sum to 1.0, got {total_ratio}"
    assert all(r > 0 for r in [train_ratio, val_ratio, test_ratio]), "All split ratios must be positive"

    np.random.seed(seed)
    print(f"\n=== Patient-Level Data Splitting (seed={seed}) ===")

    # Group records by patient
    patient_records = {}
    for record in records_data:
        pid = record['patient_id']
        if pid not in patient_records:
            patient_records[pid] = []
        patient_records[pid].append(record)

    # Shuffle patient IDs for unbiased splitting
    unique_patients = list(patient_records.keys())
    np.random.shuffle(unique_patients)

    # Calculate split indices
    n_patients = len(unique_patients)
    if n_patients < 10:
        raise ValueError(
            f"Insufficient patients ({n_patients}) for reliable splitting. "
            f"Minimum 10 patients recommended."
        )

    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    n_test = n_patients - n_train - n_val  # Ensure all patients are assigned

    print(f"  Total patients: {n_patients}")
    print(f"  Train patients: {n_train} ({n_train/n_patients*100:.1f}%)")
    print(f"  Val patients: {n_val} ({n_val/n_patients*100:.1f}%)")
    print(f"  Test patients: {n_test} ({n_test/n_patients*100:.1f}%)")

    # Split patient IDs
    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train:n_train + n_val])
    test_patients = set(unique_patients[n_train + n_val:])

    # CRITICAL: Verify no patient overlap (prevents data leakage)
    assert len(train_patients & val_patients) == 0, "Train-Val patient overlap detected!"
    assert len(train_patients & test_patients) == 0, "Train-Test patient overlap detected!"
    assert len(val_patients & test_patients) == 0, "Val-Test patient overlap detected!"
    print("  ✓ Verified: No patient overlap between splits (no data leakage)")

    # Assign records to splits
    train_data = []
    val_data = []
    test_data = []

    for pid in train_patients:
        train_data.extend(patient_records[pid])
    for pid in val_patients:
        val_data.extend(patient_records[pid])
    for pid in test_patients:
        test_data.extend(patient_records[pid])

    # Report final split statistics
    total_samples = len(train_data) + len(val_data) + len(test_data)
    print(f"  Total samples: {total_samples}")
    print(f"  Train samples: {len(train_data)} ({len(train_data)/total_samples*100:.1f}%)")
    print(f"  Val samples: {len(val_data)} ({len(val_data)/total_samples*100:.1f}%)")
    print(f"  Test samples: {len(test_data)} ({len(test_data)/total_samples*100:.1f}%)")

    return train_data, val_data, test_data


def verify_no_leakage(train_data, val_data, test_data):
    """Verify that no patient appears in multiple splits."""
    train_patients = set(r['patient_id'] for r in train_data)
    val_patients = set(r['patient_id'] for r in val_data)
    test_patients = set(r['patient_id'] for r in test_data)
    
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
    
    print("✓ No data leakage detected - patient-level splitting verified")
    return True


def train_combined_model(
    augment_prob=0.5,
    noise_factor=0.05,
    scale_range=(0.8, 1.2),
    seed=42,
    overlap=0.5,
):
    """
    Trains an LSTM classifier using multiple PhysioNet datasets.
    FIXED: Now uses patient-level splitting to prevent data leakage.
    """
    # Reproducibility
    np.random.seed(seed)
    
    print("\n" + "="*70)
    print("LOADING DATA WITH PATIENT-LEVEL TRACKING (Leakage Prevention)")
    print("="*70)
    
    # Using single-label datasets only (mutually exclusive classes)
    # Multi-label datasets (PTB-XL, Chapman) excluded to maintain consistent
    # single-label formulation with softmax activation
    datasets = [
        ('sleep-edf', 'slp01'),      # Single-label: Sleep stages (W, R, N1, N2, N3)
        ('mitdb', '100'),             # Single-label: Arrhythmia types (N, V, A, etc.)
        ('mimic3wdb', '3000003_0003') # Single-label: ICU waveforms
    ]

    # NOTE: Removed for single-label consistency:
    # - ptbdb: Can have overlapping pathologies
    # - ptb-xl: Multi-label (multiple diagnoses per patient)
    # - challenge-2020: Multi-label formulation
    
    # Load all records with patient tracking AND lead selection
    all_records = []
    for db, rec in datasets:
        try:
            print(f"Loading {db}/{rec}...")
            data = load_physionet_dataset(db, rec, lead_selection=True)
            signal = normalize(bandpass_filter(data['signal'], db, data['fs']))
            X, y = segment_signal_data(signal, data['annotations'], overlap=overlap)
            
            # Extract patient ID
            patient_id = extract_patient_id(db, rec)
            
            # Store record with patient tracking
            all_records.append({
                'patient_id': patient_id,
                'database': db,
                'record_id': rec,
                'segments': X,
                'labels': y,
                'n_leads_original': data['n_leads_original'],
                'n_leads_used': data['n_leads_used']
            })
            print(f"  ✓ Loaded {len(X)} segments from patient {patient_id}")
            print(f"    Original leads: {data['n_leads_original']}, Used: {data['n_leads_used']} (primary lead selected)")
            
        except Exception as e:
            print(f"  ✗ Failed to load {db}/{rec}: {e}")
    
    print(f"\n✓ Total records loaded: {len(all_records)}")
    print(f"✓ Unique patients: {len(set(r['patient_id'] for r in all_records))}")
    
    # CRITICAL: Patient-level split (prevents data leakage)
    print("\n" + "="*70)
    print("PERFORMING PATIENT-LEVEL SPLIT (70% train, 15% val, 15% test)")
    print("="*70)
    
    train_records, val_records, test_records = patient_level_split(
        all_records,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed
    )
    
    print(f"Train: {len(train_records)} records from {len(set(r['patient_id'] for r in train_records))} patients")
    print(f"Val:   {len(val_records)} records from {len(set(r['patient_id'] for r in val_records))} patients")
    print(f"Test:  {len(test_records)} records from {len(set(r['patient_id'] for r in test_records))} patients")
    
    # Verify no leakage
    verify_no_leakage(train_records, val_records, test_records)
    
    # Combine segments from all records in each split
    X_train = np.concatenate([r['segments'] for r in train_records], axis=0)
    y_train = np.concatenate([r['labels'] for r in train_records], axis=0)
    
    X_val = np.concatenate([r['segments'] for r in val_records], axis=0)
    y_val = np.concatenate([r['labels'] for r in val_records], axis=0)
    
    X_test = np.concatenate([r['segments'] for r in test_records], axis=0)
    y_test = np.concatenate([r['labels'] for r in test_records], axis=0)
    
    print(f"\nSegment counts:")
    print(f"  Train: {X_train.shape[0]} segments")
    print(f"  Val:   {X_val.shape[0]} segments")
    print(f"  Test:  {X_test.shape[0]} segments")
    
    # Encode labels
    le = LabelEncoder()
    all_labels = np.concatenate([y_train, y_val, y_test])
    le.fit(all_labels)
    
    y_train = to_categorical(le.transform(y_train))
    y_val = to_categorical(le.transform(y_val))
    y_test = to_categorical(le.transform(y_test))

    # Data augmentation: training set only
    # Implements temporal shifting, amplitude scaling, and additive Gaussian noise
    # as specified in paper Section III.C.1
    if augment_prob and noise_factor and scale_range:
        print(f"\nApplying data augmentation to {X_train.shape[0]} training samples...")
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

    # ========================================================================
    # MODEL HYPERPARAMETERS (See Table III in paper for complete specifications)
    # ========================================================================
    # Architecture:
    #   - LSTM Layer 1: 128 units, return_sequences=True
    #   - Dropout: 0.2
    #   - LSTM Layer 2: 64 units
    #   - Dropout: 0.2
    #   - Dense: 32 units, ReLU activation
    #   - Output: softmax (single-label classification)
    #
    # Optimization:
    #   - Optimizer: Adam (lr=0.001, β1=0.9, β2=0.999, ε=1e-7)
    #   - Batch size: 32
    #   - Epochs: 50 (configured below)
    #   - Loss: categorical_crossentropy
    #
    # Regularization:
    #   - Dropout rate: 0.2 (prevents overfitting)
    #   - Class weighting: balanced (handles class imbalance)
    #   - LR scheduler: ReduceLROnPlateau (factor=0.5, patience=10)
    #
    # Data Augmentation (applied above):
    #   - Probability: 0.5
    #   - Noise factor: 0.05
    #   - Amplitude scaling: [0.8, 1.2]
    #   - Temporal shift: ±10% of window
    # ========================================================================

    # Build model
    # Architecture: LSTM for single-lead, single-label classification
    # - Input: (batch, 3000, 1) where 1 = single selected lead per dataset
    # - Output: softmax over mutually exclusive classes (single-label)
    #
    # Lead selection per dataset (see select_lead function):
    #   - MIT-BIH: Lead 0 (MLII - modified limb lead II, standard for arrhythmia)
    #   - Sleep-EDF: Lead 0 (Fpz-Cz EEG channel)
    #   - MIMIC-III: Lead 0 (first available ICU lead)
    #
    # All datasets use single-label formulation (mutually exclusive classes)
    print("\n" + "="*70)
    print("BUILDING MODEL - Single-Lead, Single-Label Architecture")
    print(f"Input shape: ({X_train.shape[1]} timesteps, {X_train.shape[2]} channel)")
    print(f"  - Single lead selected per dataset (see lead_config)")
    print(f"Output: {y_train.shape[1]} classes (softmax - mutually exclusive)")
    print(f"  - Single-label classification (one class per sample)")
    print("="*70)
    
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))  # Softmax for single-label
    
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
    classes = np.arange(y_train.shape[1])
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_labels)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

    # Learning rate scheduler
    lr_cb = ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)
    
    # Comprehensive metrics callback (Section VI.B) - NOW USING VALIDATION SET
    metrics_cb = MetricsCallback(
        validation_data=(X_val, y_val),  # Using proper validation set
        n_classes=y_train.shape[1]
    )

    print("\n" + "="*60)
    print("Starting training with comprehensive evaluation metrics")
    print("Metrics computed per epoch: Accuracy, Sensitivity, Specificity, F1, AUC")
    print("Using VALIDATION set for monitoring (test set reserved for final eval)")
    print("="*60 + "\n")

    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_data=(X_val, y_val),  # Using validation set, not test
        class_weight=class_weight_dict,
        callbacks=[lr_cb, metrics_cb],
        verbose=1
    )
    
    # Final evaluation on TEST set (used only once)
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET (No Leakage)")
    print("="*60)
    
    y_test_pred_proba = model.predict(X_test, verbose=0)
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)
    y_test_true = np.argmax(y_test, axis=1)
    
    test_accuracy = accuracy_score(y_test_true, y_test_pred)
    test_sensitivity = recall_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    test_f1 = f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    
    try:
        test_auc = roc_auc_score(y_test, y_test_pred_proba, average='macro', multi_class='ovr')
    except ValueError:
        test_auc = 0.0
    
    # Compute specificity
    cm = confusion_matrix(y_test_true, y_test_pred)
    specificities = []
    for c in range(y_train.shape[1]):
        TP = cm[c, c] if c < cm.shape[0] and c < cm.shape[1] else 0
        FN = cm[c, :].sum() - TP if c < cm.shape[0] else 0
        FP = cm[:, c].sum() - TP if c < cm.shape[1] else 0
        TN = cm.sum() - (TP + FP + FN)
        denom = TN + FP
        spec = TN / denom if denom > 0 else 0.0
        specificities.append(spec)
    test_specificity = float(np.mean(specificities))
    
    print(f"\nTest Set Performance (True Generalization - No Leakage):")
    print(f"  Accuracy:     {test_accuracy:.4f}")
    print(f"  Sensitivity:  {test_sensitivity:.4f}")
    print(f"  Specificity:  {test_specificity:.4f}")
    print(f"  F1-score:     {test_f1:.4f}")
    print(f"  ROC AUC:      {test_auc:.4f}")
    print("="*60)
    
    print("\n✓ Training complete with patient-level splitting (no data leakage)")
    
    # Attach metrics history to model for later access
    model.metrics_history = metrics_cb.history
    
    return model

# Trains the model at runtime (can be moved to a startup script)
trained_health_model = train_combined_model()

