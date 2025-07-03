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

## 📄 License

MIT License. Use freely with attribution.
