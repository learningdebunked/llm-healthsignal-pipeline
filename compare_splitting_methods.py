#!/usr/bin/env python3
"""
Comparison: Segment-level vs Patient-level Splitting
Demonstrates the impact of data leakage on performance metrics
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulate patient data
np.random.seed(42)

# Create synthetic data: 10 patients, 100 segments each
n_patients = 10
segments_per_patient = 100
n_features = 50

print("="*70)
print("DATA LEAKAGE DEMONSTRATION")
print("="*70)

# Generate data with patient-specific patterns
all_segments = []
all_labels = []
patient_ids = []

for patient_id in range(n_patients):
    # Each patient has a unique "signature" pattern
    patient_signature = np.random.randn(n_features) * 2
    
    for seg in range(segments_per_patient):
        # Segment = patient signature + small noise
        segment = patient_signature + np.random.randn(n_features) * 0.5
        label = patient_id % 2  # Binary classification
        
        all_segments.append(segment)
        all_labels.append(label)
        patient_ids.append(patient_id)

X = np.array(all_segments)
y = np.array(all_labels)
patient_ids = np.array(patient_ids)

print(f"\nDataset: {len(X)} segments from {n_patients} patients")
print(f"Segments per patient: {segments_per_patient}")
print(f"Features per segment: {n_features}")

# ============================================================================
# METHOD 1: SEGMENT-LEVEL SPLIT (WRONG - HAS LEAKAGE)
# ============================================================================

print("\n" + "="*70)
print("METHOD 1: SEGMENT-LEVEL SPLIT (Current Implementation - WRONG)")
print("="*70)

X_train_seg, X_test_seg, y_train_seg, y_test_seg, pid_train, pid_test = train_test_split(
    X, y, patient_ids, test_size=0.2, random_state=42
)

# Check for patient overlap
train_patients_seg = set(pid_train)
test_patients_seg = set(pid_test)
overlap = train_patients_seg & test_patients_seg

print(f"\nTrain set: {len(X_train_seg)} segments")
print(f"Test set: {len(X_test_seg)} segments")
print(f"Unique patients in train: {len(train_patients_seg)}")
print(f"Unique patients in test: {len(test_patients_seg)}")
print(f"⚠️  PATIENT OVERLAP: {len(overlap)} patients appear in BOTH sets!")
print(f"   Overlapping patients: {sorted(overlap)}")

# Simple model: predict based on similarity to training data
def predict_with_leakage(X_train, y_train, X_test):
    """Predicts based on nearest neighbor (benefits from leakage)"""
    predictions = []
    for test_sample in X_test:
        # Find most similar training sample
        distances = np.linalg.norm(X_train - test_sample, axis=1)
        nearest_idx = np.argmin(distances)
        predictions.append(y_train[nearest_idx])
    return np.array(predictions)

y_pred_seg = predict_with_leakage(X_train_seg, y_train_seg, X_test_seg)
accuracy_seg = accuracy_score(y_test_seg, y_pred_seg)

print(f"\n📊 INFLATED ACCURACY: {accuracy_seg:.1%}")
print("   (High because test segments are similar to training segments from same patients)")

# ============================================================================
# METHOD 2: PATIENT-LEVEL SPLIT (CORRECT - NO LEAKAGE)
# ============================================================================

print("\n" + "="*70)
print("METHOD 2: PATIENT-LEVEL SPLIT (Fixed Implementation - CORRECT)")
print("="*70)

# Split patients first
unique_patients = np.unique(patient_ids)
np.random.seed(42)
np.random.shuffle(unique_patients)

n_train = int(len(unique_patients) * 0.8)
train_patients = set(unique_patients[:n_train])
test_patients = set(unique_patients[n_train:])

# Then assign segments based on patient
train_mask = np.array([pid in train_patients for pid in patient_ids])
test_mask = np.array([pid in test_patients for pid in patient_ids])

X_train_pat = X[train_mask]
y_train_pat = y[train_mask]
X_test_pat = X[test_mask]
y_test_pat = y[test_mask]

print(f"\nTrain set: {len(X_train_pat)} segments from {len(train_patients)} patients")
print(f"Test set: {len(X_test_pat)} segments from {len(test_patients)} patients")
print(f"Train patients: {sorted(train_patients)}")
print(f"Test patients: {sorted(test_patients)}")

# Verify no overlap
overlap_pat = train_patients & test_patients
print(f"✅ PATIENT OVERLAP: {len(overlap_pat)} patients (NONE - correct!)")

y_pred_pat = predict_with_leakage(X_train_pat, y_train_pat, X_test_pat)
accuracy_pat = accuracy_score(y_test_pat, y_pred_pat)

print(f"\n📊 TRUE ACCURACY: {accuracy_pat:.1%}")
print("   (Lower because test patients are truly unseen)")

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)

print(f"\nSegment-level split (WRONG):  {accuracy_seg:.1%} accuracy")
print(f"Patient-level split (CORRECT): {accuracy_pat:.1%} accuracy")
print(f"\n⚠️  INFLATION DUE TO LEAKAGE: {(accuracy_seg - accuracy_pat)*100:.1f} percentage points")

print("\n" + "="*70)
print("WHY THIS MATTERS")
print("="*70)
print("""
1. SEGMENT-LEVEL SPLIT (Wrong):
   - Test segments are similar to training segments from same patients
   - Model "memorizes" patient-specific patterns
   - Inflated performance doesn't reflect real-world generalization
   - Fails when deployed on new patients

2. PATIENT-LEVEL SPLIT (Correct):
   - Test patients are completely unseen during training
   - Model must learn generalizable patterns
   - True measure of clinical utility
   - Reflects real-world deployment performance

CONCLUSION:
The reviewer is correct - without patient-level splitting, reported
performance is materially inflated and does not reflect true generalization
capability. Our fixed implementation addresses this critical issue.
""")

print("="*70)
print("RECOMMENDATION")
print("="*70)
print("""
✅ Always use patient-level splitting for medical ML
✅ Verify no patient overlap between splits
✅ Document patient IDs in each split
✅ Report both validation and test performance
✅ Expect 3-10% performance drop (this is normal and correct!)
""")
