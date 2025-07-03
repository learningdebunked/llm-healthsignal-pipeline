# 🩺 LLM Healthcare Pipeline

**A beginner-friendly AI pipeline to analyze ECG/EEG signals from real patient data using deep learning and GPT-2. Built for healthcare diagnostics, signal classification, and natural language explanations.**

---

## 🚀 What is This?

This project helps you:

✅ Download real medical data from PhysioNet  
✅ Preprocess ECG/EEG signals  
✅ Train a model to detect heart or brain abnormalities  
✅ Use GPT-2 to explain those results in plain English  
✅ Expose everything as a simple web API using Flask  

If you don’t know signal processing, machine learning, or LLMs—don’t worry. The code is heavily commented and the system is modular.

---

## 🧱 How it Works (Simple Breakdown)

| Layer         | What it does                                         |
|---------------|------------------------------------------------------|
| 1️⃣ Edge       | Loads raw signals, filters & normalizes them        |
| 2️⃣ Cloud AI   | Trains models + generates text explanations         |
| 3️⃣ Frontend   | Runs a Flask API you can talk to                    |

---

## 🧠 Data Used (from PhysioNet)

This script pulls real signals from:

- **Sleep-EDF** → EEG for sleep stage classification  
- **MIT-BIH** → ECG for arrhythmia detection  
- **PTB Diagnostic & PTB-XL** → ECG for heart issues  
- **Chapman/Ningbo** → 12-lead ECGs  
- **MIMIC-III ICU** → ECG from intensive care

---

## 🛠️ Setup Instructions (for Beginners)

### 1. Clone this repo
```bash
git clone https://github.com/yourname/llm-healthcare-pipeline.git
cd llm-healthcare-pipeline
```

### 2. Create a Python environment
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> If you don’t have a `requirements.txt`, here’s a start:
```txt
wfdb
numpy
scipy
flask
scikit-learn
tensorflow
transformers
```

---

## 🧪 Run the Application

```bash
python app.py
```

- This will:
  - Download all datasets
  - Preprocess the signals
  - Train an LSTM model on 6 datasets
  - Launch a Flask API at http://localhost:5000

---

## 📡 API Endpoints

### `/ask` (POST)
Ask medical questions based on your diagnosis.
```bash
curl -X POST http://localhost:5000/ask -H "Content-Type: application/json" \
     -d '{"prompt": "Explain this ECG result: LBBB"}'
```

### `/feedback` (POST)
Send feedback after reviewing a diagnosis.
```bash
curl -X POST http://localhost:5000/feedback -H "Content-Type: application/json" \
     -d '{"user": "DrSmith", "feedback": "This worked well for AFib"}'
```

### `/dashboard`
Loads an HTML dashboard (if you build one).

---

## 🧠 What the Code Does

- `bandpass_filter()` → Removes unwanted signal noise  
- `normalize()` → Scales signals from -1 to 1  
- `segment_signal_data()` → Splits time-series into training windows  
- `train_combined_model()` → Loads 6 real datasets and trains one LSTM  
- `generate_prompt_based_response()` → Uses GPT-2 to explain diagnosis  
- `classify_signal()` → Predicts signal class (e.g. sleep stage or arrhythmia)  
- `explain_with_llm()` → Translates results into plain English  
- `log_feedback()` → Saves user feedback locally

---

## 🧩 What’s Missing?

✅ GPT-2 works out of the box  
✅ Training on real signals is done automatically  
❌ No frontend UI (just a `/dashboard` route)  
❌ No model saving (`model.save()` not implemented)  
❌ No authentication or database storage

> Want to improve this? PRs welcome!

---

## 📚 Learn More (Suggested for Beginners)

- [PhysioNet Datasets](https://physionet.org/)
- [Time Series Classification with Keras](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Transformers by HuggingFace](https://huggingface.co/docs/transformers/index)
- [Butterworth Filter Basics](https://en.wikipedia.org/wiki/Butterworth_filter)

---

Edge Layer — Signal Acquisition & Preprocessing
python
Copy
Edit
import wfdb  # Library to load/download PhysioNet signal records
import os
import numpy as np  # For numerical computations
from scipy.signal import butter, lfilter  # For filtering noise from biomedical signals
These libraries help you:

Download & read medical datasets (WFDB format)

Filter out noise from physiological signals

Work with signals as arrays

python
Copy
Edit
from sklearn.model_selection import train_test_split  # Split data into training and test sets
from sklearn.preprocessing import LabelEncoder  # Converts text labels to numbers
from tensorflow.keras.models import Sequential  # For building ML models
from tensorflow.keras.layers import Dense, LSTM  # Fully connected + recurrent layers
from tensorflow.keras.utils import to_categorical  # Converts integer labels into one-hot vectors
This section brings in tools to:

Encode labels

Build & train deep learning models using Keras

Prompt Template for LLM
python
Copy
Edit
prebuilt_prompts = {
    "explain_ecg": "Explain this ECG result: {}",
    "next_steps": "Given the diagnosis '{}', what are the recommended next steps?",
    ...
}
Defines reusable prompts that can be filled in with actual values when interacting with the LLM.

python
Copy
Edit
def get_prompt(template_name, context):
    ...
Fetches the template and fills it with context, for example:

python
Copy
Edit
get_prompt("explain_ecg", "Atrial Fibrillation")
Signal Preprocessing
python
Copy
Edit
def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=250.0, order=5):
Applies a Butterworth filter to remove low (<0.5Hz) and high (>40Hz) frequency noise.

python
Copy
Edit
def normalize(signal):
Scales all values in the signal between -1 and 1 for consistent model training.

Dataset Loading
python
Copy
Edit
def load_physionet_dataset(database_name, record_id):
Downloads the dataset

Loads both the signal (.dat, .hea) and the annotation (.atr or hypnogram)

Returns a dictionary with signal, annotations, sample rate, and field names

Signal Segmentation
python
Copy
Edit
def segment_signal_data(signal, annotations, window_size=3000):
Splits the full-length signal into fixed-size chunks (window_size) and assigns each a label.

Model Training
python
Copy
Edit
def train_combined_model():
This is the core training function.

It:

Loads 6 real-world PhysioNet datasets

Applies filtering and normalization

Segments the data

Combines and encodes everything

Builds an LSTM model

Trains on the signals + labels

Returns the trained model

python
Copy
Edit
trained_health_model = train_combined_model()
Runs training immediately when the script is executed.

✅ 3.2 Cloud Inference Layer
LLM Setup
python
Copy
Edit
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
...
llm_pipeline = pipeline("text-generation", ...)
Loads the GPT-2 model and creates a pipeline to use it for text generation (explanations, summaries, next steps).

Signal Completion
python
Copy
Edit
def complete_signal(signal, method="gan"):
Fills in missing/low-quality signal data using either:

GAN

Diffusion model

Classification Logic
python
Copy
Edit
def classify_signal(signal, model_type="transformer"):
Accepts cnn, rnn, or transformer

Runs signal through corresponding model

Returns predictions

LLM Explanation & Prompt Completion
python
Copy
Edit
def explain_with_llm(classification_result):
Explains model prediction using a large language model (like GPT-2)

python
Copy
Edit
def generate_prompt_based_response(prompt, max_tokens=100):
Sends a prompt to the LLM and returns the generated text

✅ 3.3 Frontend (Flask App)
python
Copy
Edit
from flask import Flask, request, jsonify, render_template
app = Flask(__name__)
Sets up a Flask server with basic routes.

/dashboard
python
Copy
Edit
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")
Serves a frontend page (like a web app UI)

/feedback
python
Copy
Edit
@app.route("/feedback", methods=['POST'])
def feedback():
Receives feedback in JSON and appends it to a file called feedback_log.json.

/ask
python
Copy
Edit
@app.route("/ask", methods=['POST'])
def ask():
Receives a user prompt → runs it through the GPT model → returns natural language response.

Log Writer
python
Copy
Edit
def log_feedback(feedback_data):
Saves user feedback to a file

Can be used to improve the model later (human-in-the-loop training)

Run Server
python
Copy
Edit
if __name__ == "__main__":
    app.run(debug=True)
Starts the Flask server.

🎯 Summary
This script builds a complete AI stack for biomedical signal classification:

Layer	What It Does
Edge Layer	Loads, cleans, and prepares signal data
AI Layer	Trains LSTM + uses GPT-2 for explanations
Frontend	Provides API endpoints for prediction & natural language

## 📄 License

MIT License. Use freely with attribution.
