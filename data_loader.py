import wfdb
import os
import numpy as np
from scipy.signal import butter, lfilter

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

    Raises:
        ValueError: If sampling rate is too low for filter parameters
        AssertionError: If signal contains NaN or infinite values
    """
    # Input validation
    assert not np.any(np.isnan(signal)), "Signal contains NaN values"
    assert not np.any(np.isinf(signal)), "Signal contains infinite values"
    assert fs > 0, f"Sampling frequency must be positive, got {fs}"

    # Modality-specific filter parameters
    filter_params = {
        'mitdb':      {'low': 0.5, 'high': 50},   # ECG: preserve QRS, remove baseline wander
        'mimic3wdb':  {'low': 0.5, 'high': 50},   # ECG: standard cardiac filtering
        'sleep-edf':  {'low': 0.5, 'high': 30},   # EEG: preserve sleep frequency bands
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
        AssertionError: If signal contains NaN/inf or normalization fails
    """
    # Input validation
    assert not np.any(np.isnan(signal)), "Input signal contains NaN values"
    assert not np.any(np.isinf(signal)), "Input signal contains infinite values"
    assert signal.size > 0, "Input signal is empty"

    eps = 1e-8  # Numerical stability epsilon
    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True)

    normalized = (signal - mean) / (std + eps)

    # Verify output quality
    assert not np.any(np.isnan(normalized)), "Normalization produced NaN values"
    assert not np.any(np.isinf(normalized)), "Normalization produced infinite values"

    return normalized

def load_physionet_dataset(database_name, record_id):
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
    segments, labels = [], []
    step_size = max(1, int(window_size * (1 - overlap)))
    seg_idx = 0
    for i in range(0, len(signal) - window_size + 1, step_size):
        segments.append(signal[i:i+window_size])
        if seg_idx < len(annotations):
            labels.append(annotations[seg_idx])
        seg_idx += 1
    return np.array(segments), np.array(labels)

def augment_signal(signal, noise_factor=0.05):
    noise = np.random.normal(0, noise_factor, signal.shape)
    augmented = signal + noise
    scale = np.random.uniform(0.8, 1.2)
    augmented = augmented * scale
    return augmented
