# Response to Reviewer: Multi-Lead and Multi-Label Handling

## Reviewer Comment:
> "The LSTM input is described as a single channel window, and the output layer is described as softmax over classes, which does not align with common multi-lead signal handling and does not align with multi-label formulations used in datasets. The paper needs a precise description of how leads were selected or fused and how labels were represented per dataset."

---

## Our Response:

We thank the reviewer for this important clarification request. We have now explicitly documented our multi-lead and multi-label handling strategies.

### 1. Multi-Lead Signal Handling

We employed a **lead selection strategy** where a clinically relevant primary lead was selected for each dataset to ensure:
- Consistent input dimensionality across datasets
- Use of standard diagnostic leads
- Computational efficiency
- Reproducibility

**Lead Selection Per Dataset:**

| Dataset | Available Leads | Selected Lead | Rationale |
|---------|----------------|---------------|-----------|
| MIT-BIH Arrhythmia | 2 (MLII, V5) | MLII (lead 0) | Standard for arrhythmia detection |
| PTB Diagnostic | 12 (I, II, III, aVR, aVL, aVF, V1-V6) | II (lead 1) | Standard limb lead for diagnostics |
| PTB-XL | 12 (I, II, III, aVR, aVL, aVF, V1-V6) | II (lead 1) | Standard limb lead |
| Chapman-Shaoxing | 12 (I, II, III, aVR, aVL, aVF, V1-V6) | II (lead 1) | Standard limb lead |
| MIMIC-III | Variable | First available | Variable ICU monitoring setup |
| Sleep-EDF | 1 (EEG Fpz-Cz) | Single channel | Sleep staging standard |

**Implementation:**
```python
def select_lead(signal, database_name):
    """Select clinically relevant primary lead"""
    lead_config = {
        'mitdb': 0,    # MLII
        'ptbdb': 1,    # Lead II
        'ptb-xl': 1,   # Lead II
        ...
    }
    lead_idx = lead_config[database_name]
    return signal[:, lead_idx:lead_idx+1]
```

**Final Input Shape:** All signals processed as (window_size=3000, n_features=1)

### 2. Label Representation

We clarify that our current implementation uses **single-label classification** with softmax for all datasets. We acknowledge that some datasets (particularly PTB-XL) support multi-label scenarios.

**Current Implementation (Single-Label):**

| Dataset | Label Type | Classes | Encoding | Output Activation | Loss Function |
|---------|-----------|---------|----------|-------------------|---------------|
| MIT-BIH | Single-label | 5 (N, A, V, F, /) | One-hot | Softmax | Categorical CE |
| PTB Diagnostic | Single-label | 2 (Healthy, MI) | One-hot | Softmax | Categorical CE |
| PTB-XL | Single-label* | Primary diagnosis | One-hot | Softmax | Categorical CE |
| Chapman-Shaoxing | Single-label | 11 rhythm types | One-hot | Softmax | Categorical CE |
| MIMIC-III | Single-label | Outcome-based | One-hot | Softmax | Categorical CE |
| Sleep-EDF | Single-label | 5 (Wake, N1-N3, REM) | One-hot | Softmax | Categorical CE |

*Note: For PTB-XL, we use the primary diagnosis for single-label classification. Multi-label extension is discussed below.

**Encoding Process:**
```python
# Single-label encoding
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)
# Shape: (n_samples, n_classes)
# Output: Softmax (mutually exclusive)
```

### 3. Multi-Label Extension (Future Work)

We acknowledge that PTB-XL and potentially other datasets support multi-label scenarios. For completeness, we describe how multi-label classification would be implemented:

**Multi-Label Approach:**
```python
# Multi-label encoding (for PTB-XL with multiple diagnoses)
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
labels_binary = mlb.fit_transform(labels)
# Shape: (n_samples, n_classes)
# Output: Sigmoid (independent probabilities)
# Loss: Binary cross-entropy
```

**Model Modification for Multi-Label:**
```python
# Change output layer
Dense(n_classes, activation='sigmoid')  # Instead of softmax
# Change loss
model.compile(loss='binary_crossentropy', ...)
```

### 4. Manuscript Updates

We have added the following to the manuscript:

**Section III.B.1 (NEW): Multi-Lead Signal Handling**
- Detailed lead selection strategy
- Rationale for each dataset
- Input shape specification

**Section III.B.2 (NEW): Label Representation**
- Single-label vs multi-label distinction
- Encoding methodology per dataset
- Output layer configuration

**Table III (NEW): Dataset Configuration**
- Complete specification of leads, labels, and model configuration
- Enables full reproducibility

### 5. Code Implementation

**New Functions Added:**
```python
def select_lead(signal, database_name, lead_config=None):
    """Select appropriate lead from multi-lead signal"""
    # Returns shape: (n_samples, 1)

def load_physionet_dataset(database_name, record_id, lead_selection=True):
    """Load data with lead selection"""
    # Returns selected lead and metadata
```

**Updated Training Output:**
```
Loading mitdb/100...
  ✓ Loaded 1000 segments from patient mitdb_100
    Original leads: 2, Used: 1 (primary lead selected)
```

---

## Summary

We have now explicitly documented:
✅ Lead selection strategy (primary clinically relevant lead per dataset)  
✅ Input shape specification (3000 timesteps × 1 feature)  
✅ Label encoding methodology (single-label with one-hot encoding)  
✅ Output layer configuration (softmax for mutually exclusive classes)  
✅ Dataset-specific configurations in tabular format  
✅ Code implementation with lead selection function  

The methodology is now fully specified and reproducible. For datasets that inherently support multi-label classification (e.g., PTB-XL), we have described the extension approach, though our current results use single-label (primary diagnosis) for consistency across all datasets.
