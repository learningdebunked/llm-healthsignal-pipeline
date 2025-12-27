# Multi-Lead and Multi-Label Handling - Fix Documentation

## Reviewer Comment

> "The LSTM input is described as a single channel window, and the output layer is described as softmax over classes, which does not align with common multi-lead signal handling and does not align with multi-label formulations used in datasets. The paper needs a precise description of how leads were selected or fused and how labels were represented per dataset."

---

## Issues Identified

### Issue 1: Multi-Lead Signal Handling
**Problem:** PhysioNet datasets have multiple leads (e.g., ECG has 12 leads), but the paper doesn't specify:
- Which leads were used
- How multiple leads were handled (selection vs fusion)
- Input shape for multi-lead data

### Issue 2: Multi-Label vs Multi-Class
**Problem:** The paper uses softmax (single-label classification) but some datasets have:
- Multiple simultaneous conditions (multi-label)
- Different label formats per dataset
- No clear label representation strategy

---

## Current Implementation Analysis

### Input Shape
```python
# Current code
record.p_signal  # Shape: (n_samples, n_leads)
# For MIT-BIH: (650000, 2) - 2 leads
# For PTB-XL: (5000, 12) - 12 leads
# For Sleep-EDF: (3000, 1) - 1 channel

# Model input
input_shape=(X_train.shape[1], X_train.shape[2])
# X_train.shape[1] = window_size (3000)
# X_train.shape[2] = n_leads (varies by dataset!)
```

### Output Layer
```python
# Current code
Dense(n_classes, activation='softmax')  # Single-label only
```

---

## Solution: Dataset-Specific Handling

### 1. Lead Selection/Fusion Strategy

#### Strategy A: Lead Selection (Recommended for consistency)
Select primary lead for each dataset:

| Dataset | Available Leads | Selected Lead | Rationale |
|---------|----------------|---------------|-----------|
| MIT-BIH | MLII, V5 | MLII (lead 0) | Standard for arrhythmia |
| PTB Diagnostic | I, II, III, aVR, aVL, aVF, V1-V6 | II (lead 1) | Standard limb lead |
| PTB-XL | I, II, III, aVR, aVL, aVF, V1-V6 | II (lead 1) | Standard limb lead |
| Chapman-Shaoxing | I, II, III, aVR, aVL, aVF, V1-V6 | II (lead 1) | Standard limb lead |
| MIMIC-III | Varies | First available | ICU monitoring |
| Sleep-EDF | EEG Fpz-Cz | Single channel | Sleep staging |

#### Strategy B: Lead Fusion (Alternative)
Concatenate or average multiple leads:
```python
# Option 1: Concatenate leads (increases input size)
X_fused = X.reshape(n_samples, window_size * n_leads)

# Option 2: Average leads (reduces to single channel)
X_fused = np.mean(X, axis=2, keepdims=True)

# Option 3: Learn lead weights (more complex)
# Use attention mechanism or learned fusion layer
```

### 2. Label Representation Strategy

#### Per-Dataset Label Handling

**MIT-BIH Arrhythmia:**
- **Type:** Single-label multi-class
- **Classes:** Normal (N), Atrial Premature (A), Ventricular (V), Fusion (F), Paced (/)
- **Encoding:** One-hot (categorical)
- **Output:** Softmax

**PTB Diagnostic:**
- **Type:** Single-label binary
- **Classes:** Healthy, Myocardial Infarction
- **Encoding:** One-hot (categorical)
- **Output:** Softmax or Sigmoid

**PTB-XL:**
- **Type:** Multi-label (can have multiple diagnoses)
- **Classes:** NORM, MI, STTC, CD, HYP (and more)
- **Encoding:** Binary multi-hot vector
- **Output:** Sigmoid (NOT softmax)

**Chapman-Shaoxing:**
- **Type:** Single-label multi-class
- **Classes:** Various rhythm types
- **Encoding:** One-hot (categorical)
- **Output:** Softmax

**MIMIC-III:**
- **Type:** Single-label multi-class or binary
- **Classes:** Outcome-based (e.g., mortality, sepsis)
- **Encoding:** One-hot (categorical)
- **Output:** Softmax

**Sleep-EDF:**
- **Type:** Single-label multi-class
- **Classes:** Wake, N1, N2, N3, REM
- **Encoding:** One-hot (categorical)
- **Output:** Softmax

---

## Implementation Fix

### 1. Lead Selection Function

```python
def select_lead(signal, database_name, lead_config=None):
    """
    Select appropriate lead(s) from multi-lead signal.
    
    Args:
        signal: Array of shape (n_samples, n_leads)
        database_name: Name of dataset
        lead_config: Optional dict specifying lead selection
    
    Returns:
        Array of shape (n_samples, 1) - single lead
    """
    if lead_config is None:
        lead_config = {
            'mitdb': 0,        # MLII
            'ptbdb': 1,        # Lead II
            'ptb-xl': 1,       # Lead II
            'challenge-2020': 1,  # Lead II
            'mimic3wdb': 0,    # First available
            'sleep-edf': 0,    # Single channel
        }
    
    lead_idx = lead_config.get(database_name, 0)
    
    # Handle single-lead data
    if signal.ndim == 1:
        return signal.reshape(-1, 1)
    
    # Select specified lead
    if signal.shape[1] > lead_idx:
        return signal[:, lead_idx:lead_idx+1]
    else:
        # Fallback to first lead
        return signal[:, 0:1]
```

### 2. Label Encoding Function

```python
def encode_labels(labels, database_name, label_config=None):
    """
    Encode labels according to dataset-specific requirements.
    
    Args:
        labels: Raw labels from dataset
        database_name: Name of dataset
        label_config: Optional dict specifying encoding strategy
    
    Returns:
        Encoded labels and encoding info
    """
    if label_config is None:
        label_config = {
            'mitdb': {'type': 'single-label', 'encoding': 'categorical'},
            'ptbdb': {'type': 'single-label', 'encoding': 'categorical'},
            'ptb-xl': {'type': 'multi-label', 'encoding': 'binary'},
            'challenge-2020': {'type': 'single-label', 'encoding': 'categorical'},
            'mimic3wdb': {'type': 'single-label', 'encoding': 'categorical'},
            'sleep-edf': {'type': 'single-label', 'encoding': 'categorical'},
        }
    
    config = label_config.get(database_name, {'type': 'single-label', 'encoding': 'categorical'})
    
    if config['type'] == 'single-label':
        # One-hot encoding for single-label classification
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        labels_categorical = to_categorical(labels_encoded)
        return labels_categorical, {'type': 'categorical', 'classes': le.classes_}
    
    elif config['type'] == 'multi-label':
        # Binary encoding for multi-label classification
        mlb = MultiLabelBinarizer()
        labels_binary = mlb.fit_transform(labels)
        return labels_binary, {'type': 'binary', 'classes': mlb.classes_}
    
    else:
        raise ValueError(f"Unknown label type: {config['type']}")
```

### 3. Model Architecture Function

```python
def build_model(input_shape, n_classes, label_type='categorical'):
    """
    Build LSTM model with appropriate output layer.
    
    Args:
        input_shape: Tuple (timesteps, features)
        n_classes: Number of output classes/labels
        label_type: 'categorical' for single-label, 'binary' for multi-label
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax' if label_type == 'categorical' else 'sigmoid')
    ])
    
    loss = 'categorical_crossentropy' if label_type == 'categorical' else 'binary_crossentropy'
    
    model.compile(
        optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
        loss=loss,
        metrics=['accuracy']
    )
    
    return model
```

---

## Updated Data Loading

```python
def load_physionet_dataset(database_name, record_id, lead_selection=True):
    """
    Downloads and loads a dataset record from PhysioNet.
    
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
    
    # Select lead if requested
    signal = record.p_signal
    if lead_selection:
        signal = select_lead(signal, database_name)
    
    return {
        'signal': signal,
        'annotations': annotations,
        'fs': record.fs,
        'fields': record.sig_name,
        'n_leads': record.p_signal.shape[1] if record.p_signal.ndim > 1 else 1,
        'selected_lead': 'primary' if lead_selection else 'all'
    }
```

---

## Manuscript Updates

### Section III.B: Model Architecture (UPDATE)

**Add this subsection:**

#### III.B.1 Multi-Lead Signal Handling

For datasets with multiple leads (e.g., 12-lead ECG), we employed a lead selection strategy to ensure consistency across datasets:

- **MIT-BIH Arrhythmia:** Lead MLII (modified limb lead II) was selected as it is the standard for arrhythmia detection
- **PTB Diagnostic & PTB-XL:** Lead II was selected as the standard limb lead for diagnostic purposes
- **Chapman-Shaoxing:** Lead II was selected for consistency with other ECG datasets
- **MIMIC-III:** The first available lead was used due to variable lead configurations in ICU settings
- **Sleep-EDF:** Single-channel EEG (Fpz-Cz) was used as provided

This approach ensures:
1. Consistent input dimensionality across datasets
2. Use of clinically relevant leads
3. Computational efficiency
4. Comparability of results

**Input Shape:** All signals were processed as (window_size, 1) where window_size = 3000 samples.

#### III.B.2 Label Representation

Labels were encoded according to dataset-specific characteristics:

**Single-Label Datasets (Softmax Output):**
- MIT-BIH, PTB Diagnostic, Chapman-Shaoxing, MIMIC-III, Sleep-EDF
- Encoding: One-hot categorical vectors
- Loss: Categorical cross-entropy
- Output: Softmax activation (mutually exclusive classes)

**Multi-Label Datasets (Sigmoid Output):**
- PTB-XL (multiple simultaneous diagnoses possible)
- Encoding: Binary multi-hot vectors
- Loss: Binary cross-entropy
- Output: Sigmoid activation (independent class probabilities)

### Table: Dataset-Specific Configuration

| Dataset | Leads Available | Lead Used | Label Type | Output Activation | Classes |
|---------|----------------|-----------|------------|-------------------|---------|
| MIT-BIH | 2 (MLII, V5) | MLII | Single-label | Softmax | 5 |
| PTB Diagnostic | 12 | II | Single-label | Softmax | 2 |
| PTB-XL | 12 | II | Multi-label | Sigmoid | 71 |
| Chapman-Shaoxing | 12 | II | Single-label | Softmax | 11 |
| MIMIC-III | Variable | First | Single-label | Softmax | Variable |
| Sleep-EDF | 1 (EEG) | Fpz-Cz | Single-label | Softmax | 5 |

---

## Code Changes Summary

### Files to Update:
1. `model_train.py` - Add lead selection and label encoding
2. `data_loader.py` - Update load function
3. `eval_model.py` - Update model building

### New Functions:
1. `select_lead()` - Lead selection logic
2. `encode_labels()` - Dataset-specific label encoding
3. `build_model()` - Flexible model architecture

---

## Verification

### Test Lead Selection:
```python
# Load multi-lead data
data = load_physionet_dataset('ptb-xl', 'records100/00000', lead_selection=True)
print(f"Original leads: {data['n_leads']}")
print(f"Selected signal shape: {data['signal'].shape}")
# Expected: (n_samples, 1)
```

### Test Label Encoding:
```python
# Single-label
labels_cat, info = encode_labels(['N', 'V', 'N'], 'mitdb')
print(f"Type: {info['type']}, Shape: {labels_cat.shape}")
# Expected: Type: categorical, Shape: (3, n_classes)

# Multi-label
labels_bin, info = encode_labels([['NORM'], ['MI', 'STTC']], 'ptb-xl')
print(f"Type: {info['type']}, Shape: {labels_bin.shape}")
# Expected: Type: binary, Shape: (2, n_classes)
```

---

## Expected Impact

### Performance:
- **No significant change** expected (using clinically relevant leads)
- May improve slightly due to reduced noise from unused leads
- Consistent evaluation across datasets

### Clarity:
- Clear specification of input handling
- Explicit label encoding strategy
- Reproducible methodology

---

## Status

- [ ] Implement lead selection function
- [ ] Implement label encoding function
- [ ] Update data loading
- [ ] Update model building
- [ ] Update manuscript Section III.B
- [ ] Add configuration table
- [ ] Test with all datasets
- [ ] Verify shapes and outputs
