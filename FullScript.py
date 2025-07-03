# usethis if the modularized files are not working 

# Signal acquisition and lightweight preprocessing

import wfdb  # Library to load/download PhysioNet signal records
import os
import numpy as np
from scipy.signal import butter, lfilter  # Signal filtering functions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
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


# === 3.2 Cloud Inference Layer ===
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from your_model_zoo import load_cnn_model, load_rnn_model, load_transformer_classifier
from generative_models import complete_signal_with_gan, complete_signal_with_diffusion
from llm_explainer import get_diagnosis_explanation

# Load GPT-2 model for natural language response generation
llm_model_name = "gpt2"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
llm_pipeline = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer)

def complete_signal(signal, method="gan"):
    """
    Uses a generative model to complete or denoise an input signal.
    """
    if method == "gan":
        return complete_signal_with_gan(signal)
    elif method == "diffusion":
        return complete_signal_with_diffusion(signal)
    else:
        return signal

def classify_signal(signal, model_type="transformer"):
    """
    Uses a selected ML model to classify a given input signal.
    """
    if model_type == "cnn":
        model = load_cnn_model()
    elif model_type == "rnn":
        model = load_rnn_model()
    elif model_type == "transformer":
        model = load_transformer_classifier()
    return model.predict(signal)

def explain_with_llm(classification_result):
    """
    Uses LLM to provide natural language explanation of model results.
    """
    return get_diagnosis_explanation(classification_result)

def generate_prompt_based_response(prompt, max_tokens=100):
    """
    Sends a prompt to the LLM and returns generated natural language text.
    """
    response = llm_pipeline(prompt, max_length=max_tokens, do_sample=True, top_k=50)
    return response[0]['generated_text']


# === 3.3 Frontend ===
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route("/dashboard")
def dashboard():
    """
    Serves the dashboard HTML page (UI not included here).
    """
    return render_template("dashboard.html")

@app.route("/feedback", methods=['POST'])
def feedback():
    """
    Accepts user feedback in JSON format and logs it to a file.
    """
    user_feedback = request.json
    log_feedback(user_feedback)
    return jsonify({"status": "received"})

@app.route("/ask", methods=['POST'])
def ask():
    """
    Accepts a prompt from the frontend, runs it through the LLM, and returns the output.
    """
    user_input = request.json.get("prompt", "")
    response = generate_prompt_based_response(user_input)
    return jsonify({"response": response})

def log_feedback(feedback_data):
    """
    Appends feedback to a log file for monitoring and evaluation.
    """
    with open("feedback_log.json", "a") as f:
        f.write(str(feedback_data) + "\n")

if __name__ == "__main__":
    app.run(debug=True)
