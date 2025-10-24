import wfdb
import os
import numpy as np
from scipy.signal import butter, lfilter

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=250.0, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal, axis=0)

def normalize(signal):
    return 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1

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

def segment_signal_data(signal, annotations, window_size=3000):
    segments, labels = [], []
    for i in range(0, len(signal) - window_size, window_size):
        segments.append(signal[i:i+window_size])
        if i // window_size < len(annotations):
            labels.append(annotations[i // window_size])
    return np.array(segments), np.array(labels)

def augment_signal(signal, noise_factor=0.05):
    noise = np.random.normal(0, noise_factor, signal.shape)
    augmented = signal + noise
    scale = np.random.uniform(0.8, 1.2)
    augmented = augmented * scale
    return augmented
