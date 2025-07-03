from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical


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
    Normalizes signal values to range [-1, 1] to standardize input.
    """
    return 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1

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

def segment_signal_data(signal, annotations, window_size=3000):
    """
    Splits a long signal into fixed-size windows and assigns labels from annotations.
    """
    segments, labels = [], []
    for i in range(0, len(signal) - window_size, window_size):
        segments.append(signal[i:i+window_size])
        if i // window_size < len(annotations):
            labels.append(annotations[i // window_size])
    return np.array(segments), np.array(labels)

def train_combined_model():
    """
    Trains an LSTM classifier using multiple PhysioNet datasets.
    """
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
            X, y = segment_signal_data(signal, data['annotations'])
            all_segments.append(X)
            all_labels.append(y)
        except Exception as e:
            print(f"Failed to load {db}/{rec}: {e}")
    X = np.concatenate(all_segments, axis=0)
    y = np.concatenate(all_labels, axis=0)
    le = LabelEncoder()
    y_enc = to_categorical(le.fit_transform(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(y_enc.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
    return model

# Trains the model at runtime (can be moved to a startup script)
trained_health_model = train_combined_model()

