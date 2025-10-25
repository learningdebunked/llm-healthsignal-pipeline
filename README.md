# 🩺 LLM Healthcare Pipeline

Welcome to the **LLM Healthcare Pipeline** project! This guide is for **new developers**, or anyone curious about how AI can help with health data like ECGs and EEGs.

---

## 🤖 What This Project Does (In Simple Terms)

Imagine you have a machine that can:

- Read heart and brain signals (like ECG and EEG)
- Clean them up (remove noise)
- Break them into pieces
- Learn from them using a smart brain (AI)
- Then explain what it found in plain English using GPT-2

That's what this project does, step by step.

---

## 💡 Technologies Used

| Tool                   | What it does                                |
| ---------------------- | ------------------------------------------- |
| `wfdb`                 | Downloads ECG/EEG data from PhysioNet       |
| `numpy`                | Math with arrays (like Excel but for code)  |
| `scipy`                | Helps filter out noise from signals         |
| `keras` / `tensorflow` | Trains and runs AI models (like LSTM)       |
| `transformers`         | Lets us use GPT-2 to write natural language |
| `flask`                | Turns our code into a web app with buttons  |
| `sklearn`              | Helps prepare data and split it             |

---

## 📂 File Descriptions

### 1. `data_loader.py`

Loads health signal data and prepares it.

- `bandpass_filter()`: Removes noise from raw data
- `normalize()`: Scales values between -1 and 1 (helps model learn better)
- `load_physionet_dataset()`: Downloads and loads ECG/EEG signals
- `segment_signal_data()`: Breaks a long signal into smaller pieces

### 2. `model_train.py`

Trains an AI model (LSTM) using cleaned signal data.

- It loads multiple datasets (ECG, EEG, etc.)
- Cleans and splits the data
- Builds a neural network using Keras
- Trains it to classify heartbeats or sleep stages

### 3. `inference.py`

Uses the model to:

- Predict what's happening in a signal
- Explain it using GPT-2 with **structured prompt templates**
- Support both base and **fine-tuned medical GPT-2 models**
- Generate clinical interpretations conditioned on classification confidence
- Fill in missing signal data using GANs or diffusion

### 4. `api.py`

Runs a REST API web server with endpoints:

- `/`: API information and available endpoints
- `/dashboard`: web interface (if template available)
- `/ask`: medical queries with optional classification context
- `/classify`: signal classification + LLM interpretation
- `/feedback`: saves user feedback to a file

### 5. `finetune_gpt2.py` ⭐ NEW

Fine-tunes GPT-2 on medical domain data:

- Trains on ECG/EEG interpretations and clinical guidelines
- Implements paper's hyperparameters (lr=5e-5, warmup=500)
- Supports multiple corpus formats (JSONL, TXT)
- Includes sample corpus for demonstration
- Command-line interface with full configuration

### 6. `eval_model.py`

Evaluates model performance:

- Computes accuracy, sensitivity, specificity, F1-score, ROC-AUC
- Runs on test split from PhysioNet datasets
- Generates comprehensive metrics report

---

## 🧠 How It Works Step-by-Step

1. **Download data** from PhysioNet using `wfdb`
2. **Filter noise** using a bandpass filter
3. **Normalize** the values so they fit a consistent scale
4. **Split the signal** into chunks of 3000 units
5. **Label each chunk** with what it represents (like "AFib" or "REM sleep")
6. **Feed it into an LSTM model**
7. **Train** that model to predict future data
8. **Use GPT-2** to explain the predictions in English
9. **Provide a web API** to interact with this pipeline

---

## 📊 Supported Datasets

This project supports 6 real medical datasets from [https://physionet.org](https://physionet.org):

- MIT-BIH Arrhythmia Dataset
- PTB Diagnostic ECG Database
- PTB-XL (Extended ECG)
- Chapman-Shaoxing ECG
- MIMIC-III ICU Waveforms
- Sleep-EDF (for EEG sleep signals)

---

## 🔌 How to Run It

### Basic Setup

1. ✅ Install Python 3.8+
2. ✅ Clone the repository:

```bash
git clone https://github.com/learningdebunked/llm-healthsignal-pipeline.git
cd llm-healthsignal-pipeline
```

3. ✅ Install required packages:

```bash
pip install -r requirements.txt
```

### Option A: Run with Base GPT-2 (Quick Start)

```bash
python3 api.py
```

Server starts at: [http://localhost:3333](http://localhost:3333)

### Option B: Run with Fine-tuned Medical GPT-2 (Recommended)

1. **Fine-tune the model** (one-time setup):

```bash
# With your medical corpus
python3 finetune_gpt2.py \
    --data_dir ./medical_corpus \
    --output_dir ./medical-gpt2 \
    --epochs 3

# OR use demo mode (sample data)
python3 finetune_gpt2.py \
    --data_dir ./nonexistent \
    --output_dir ./demo-gpt2 \
    --epochs 1
```

2. **Set environment variable**:

```bash
export MEDICAL_GPT2_PATH=./medical-gpt2
```

3. **Run the API**:

```bash
python3 api.py
```

You should see: `✓ Fine-tuned medical GPT-2 model configured`

---

## 🧪 API Examples

### 1. General Medical Query

```bash
curl -X POST http://localhost:3333/ask \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain atrial fibrillation"}'
```

**Response:**
```json
{
  "response": "Atrial Fibrillation (AFib) is an irregular and often rapid heart rhythm..."
}
```

### 2. Signal Classification with Interpretation ⭐ NEW

```bash
curl -X POST http://localhost:3333/classify \
  -H "Content-Type: application/json" \
  -d '{
    "signal": [0.1, 0.2, 0.15, ...],
    "signal_type": "ECG",
    "clinical_context": "Patient with palpitations"
  }'
```

**Response:**
```json
{
  "classification": "Atrial Fibrillation",
  "confidence": 0.92,
  "interpretation": "Analysis of ECG signal:\n\n1. Finding: The signal has been classified as 'Atrial Fibrillation' with high confidence (92.0%).\n\n2. Clinical Significance: This finding requires immediate attention and specialist review.\n\n3. Recommended Actions: Immediate cardiology consultation recommended.",
  "signal_type": "ECG"
}
```

### 3. Query with Classification Context ⭐ NEW

```bash
curl -X POST http://localhost:3333/ask \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the treatment options?",
    "classification": {
      "prediction": "Atrial Fibrillation",
      "confidence": 0.92
    }
  }'
```

**Response:**
```json
{
  "response": "For Atrial Fibrillation detected with 92% confidence...\n\nTreatment options include rate control, rhythm control, and anticoagulation...",
  "classification": "Atrial Fibrillation",
  "confidence": 0.92
}
```

### 4. Submit Feedback

```bash
curl -X POST http://localhost:3333/feedback \
  -H "Content-Type: application/json" \
  -d '{"rating": 5, "comment": "Very helpful interpretation"}'
```

---

## 📚 Additional Documentation

- **[GPT2_FINETUNING.md](GPT2_FINETUNING.md)** - Complete guide to fine-tuning GPT-2 on medical data
- **[CHANGES_GPT2_INTEGRATION.md](CHANGES_GPT2_INTEGRATION.md)** - Detailed changelog of GPT-2 improvements
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Quick reference for implementation details

## 🎯 Key Features

### ✅ Structured Prompt Engineering
- Implements paper Section IV.A prompt templates
- Classification-conditioned generation
- Confidence-based clinical recommendations
- Context-aware medical explanations

### ✅ Fine-tuned Medical Models
- Support for domain-specific GPT-2 models
- Training pipeline with paper's hyperparameters
- Environment-based model selection
- Graceful fallback to base model

### ✅ Enhanced API
- `/classify` endpoint for end-to-end classification + interpretation
- `/ask` endpoint with classification context support
- Structured JSON responses with confidence scores
- Comprehensive error handling

### ✅ Comprehensive Evaluation
- Accuracy, sensitivity, specificity metrics
- F1-score and ROC-AUC computation
- Per-class performance analysis
- Confusion matrix generation

## 🧪 Running Evaluation

```bash
# Evaluate model on test data
python3 eval_model.py
```

**Output:**
```
==== Evaluation Metrics (Test Set) ====
Accuracy:     0.9234
Sensitivity:  0.8976 (macro recall)
Specificity:  0.9145 (macro)
F1-score:     0.9012 (macro)
ROC AUC:      0.9456 (macro OVR)
```

## 📘 Glossary (For Beginners)

| Term     | Meaning                                      |
| -------- | -------------------------------------------- |
| ECG      | Electrical signal from the heart             |
| EEG      | Electrical signal from the brain             |
| Signal   | Time-series data (changing values over time) |
| Filter   | Removes noise or unwanted parts              |
| LSTM     | A type of AI good at learning sequences      |
| GPT-2    | A text-generating AI (like ChatGPT)          |
| Fine-tuning | Training a pre-trained model on specific data |
| Prompt   | Structured input text to guide LLM generation |
| Confidence | Model's certainty about its prediction (0-1) |
| Classify | Predict a label for input data               |
| GAN      | An AI that can create realistic fake data    |

---

## 📄 License

MIT – free to use, just give credit.

---

## 🚀 Quick Start Examples

### Python Usage

```python
from inference import explain_with_llm, generate_prompt_based_response

# Example 1: Explain classification result
result = {"prediction": "Atrial Fibrillation", "confidence": 0.92}
interpretation = explain_with_llm(result, signal_type="ECG")
print(interpretation)

# Example 2: Ask question with context
response = generate_prompt_based_response(
    "What are treatment options?",
    classification_result=result
)
print(response)
```

### Training Your Own Model

```bash
# 1. Prepare your medical corpus
mkdir medical_corpus
# Add ecg_interpretations.jsonl, eeg_reports.jsonl, etc.

# 2. Fine-tune GPT-2
python3 finetune_gpt2.py \
    --data_dir ./medical_corpus \
    --output_dir ./my-medical-gpt2 \
    --model_name gpt2-medium \
    --epochs 3 \
    --batch_size 4

# 3. Use your model
export MEDICAL_GPT2_PATH=./my-medical-gpt2
python3 api.py
```

## 🔬 Model Architecture

**LSTM Classifier:**
- LSTM(128, return_sequences=True) → Dropout(0.2)
- LSTM(64) → Dropout(0.2)
- Dense(32, relu) → Dense(n_classes, softmax)
- Class weights for imbalanced data
- ReduceLROnPlateau scheduler

**GPT-2 Integration:**
- Base: `gpt2` (117M parameters)
- Supported: `gpt2-medium` (345M), `gpt2-large` (774M)
- Fine-tuning: lr=5e-5, warmup=500, gradient_accumulation=4
- Generation: temperature=0.7, top_k=50, top_p=0.92

## 📊 Performance Metrics

As reported in evaluation:

| Dataset | Accuracy | Sensitivity | Specificity | F1-Score | AUC |
|---------|----------|-------------|-------------|----------|-----|
| MIT-BIH | 92.3% | 89.7% | 94.1% | 0.91 | 0.95 |
| PTB Diagnostic | 94.7% | 93.2% | 95.8% | 0.94 | 0.97 |
| Sleep-EDF | 87.3% | 84.6% | 89.7% | 0.86 | 0.91 |

## ❤️ Need Help?

Open an issue or message me. Happy to help non-ML folks too!

<img width="1187" height="778" alt="Healthcare_AI_model_comparision" src="https://github.com/user-attachments/assets/c01a7264-aced-4da2-9185-cc9ffe308ada" />
<img width="1238" height="849" alt="compare_plot" src="https://github.com/user-attachments/assets/f0428b57-ddec-4d56-a7e6-115c8176c20d" />
<img width="1016" height="717" alt="health-signal-board-1" src="https://github.com/user-attachments/assets/971684f7-528b-4f58-83be-d1ed48bcd1d5" />
<img width="1055" height="814" alt="health-signal-board" src="https://github.com/user-attachments/assets/9f3c2619-dde6-4212-83d4-9eccedc100d0" />
<img width="698" height="634" alt="model-output" src="https://github.com/user-attachments/assets/5903af5a-d550-41b3-b130-21eda279406f" />
[IEEE_Format_Document_with_Tables_and_Figures.docx](https://github.com/user-attachments/files/21958476/IEEE_Format_Document_with_Tables_and_Figures.docx)
folks too!
