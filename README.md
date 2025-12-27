# 🩺 LLM Healthcare Pipeline

**🆕 Now Powered by GPT-4 for Superior Medical Interpretations!**

Welcome to the **LLM Healthcare Pipeline** project! This guide is for **new developers**, or anyone curious about how AI can help with health data like ECGs and EEGs.

> **Latest Update (Dec 2024):** Upgraded from GPT-2 to GPT-4 for significantly better medical accuracy and natural language generation. See [GPT4_MIGRATION_GUIDE.md](GPT4_MIGRATION_GUIDE.md) for details.

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
| `openai`               | Connects to GPT-4 for natural language      |
| `flask`                | Turns our code into a web app with buttons  |
| `sklearn`              | Helps prepare data and split it             |

**🆕 NEW: Now powered by GPT-4 for superior medical interpretations!**

---

## 📂 File Descriptions

### 1. `data_loader.py`

Loads health signal data and prepares it.

- `bandpass_filter()`: Removes noise from raw data
- `normalize()`: Scales values between -1 and 1 (helps model learn better)
- `load_physionet_dataset()`: Downloads and loads ECG/EEG signals
- `segment_signal_data()`: Breaks a long signal into smaller pieces

### 2. `model_train.py` ⭐ ENHANCED

Trains an AI model (LSTM) using cleaned signal data with advanced techniques:

- Loads multiple PhysioNet datasets (ECG, EEG, etc.)
- **Data augmentation**: temporal shifting, noise addition, amplitude scaling
- **Adam optimizer**: explicitly configured with paper-specified parameters
- **Comprehensive metrics**: tracks accuracy, sensitivity, specificity, F1, AUC during training
- Class weights for imbalanced data
- Learning rate scheduling

### 3. `inference.py`

Uses the model to:

- Predict what's happening in a signal
- Explain it using **GPT-4** with structured prompt templates
- Generate clinical interpretations with high accuracy
- Fill in missing signal data using GANs or diffusion

**🆕 Upgraded to GPT-4 for better medical knowledge and explanations!**

### 4. `api.py`

Runs a REST API web server with endpoints:

- `/`: API information and available endpoints
- `/dashboard`: web interface (if template available)
- `/ask`: medical queries with optional classification context
- `/classify`: signal classification + LLM interpretation
- `/feedback`: saves user feedback to a file

### 5. `finetune_gpt2.py` ⭐ DEPRECATED

**Note:** This file is now deprecated as the application uses GPT-4 via API instead of local GPT-2 models. GPT-4 has extensive medical knowledge built-in and doesn't require fine-tuning.

Previously used to fine-tune GPT-2 on medical domain data.

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

### Prerequisites

1. ✅ Python 3.8+
2. ✅ OpenAI API Key ([Get one here](https://platform.openai.com/api-keys))

### Basic Setup

1. Clone the repository:

```bash
git clone https://github.com/learningdebunked/llm-healthsignal-pipeline.git
cd llm-healthsignal-pipeline
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:

**Option A: Interactive Setup (Recommended)**
```bash
./setup_gpt4.sh
```

**Option B: Manual Setup**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

4. Run the application:

```bash
python3 api.py
```

Server starts at: [http://localhost:3333](http://localhost:3333)

### 🆕 GPT-4 Configuration

The application now uses **GPT-4** for medical interpretations. You'll see:

```
✓ GPT-4 API configured (key: ...last8chars)
🩺 Starting Healthcare AI API Server (GPT-4 Powered)...
```

**Benefits of GPT-4:**
- ✅ Superior medical knowledge and accuracy
- ✅ Better natural language understanding
- ✅ More coherent clinical explanations
- ✅ No local model download required
- ✅ Always up-to-date

**Cost:** ~$0.02-0.05 per query. See [GPT4_MIGRATION_GUIDE.md](GPT4_MIGRATION_GUIDE.md) for details.

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

### 2. Signal Classification with Interpretation ⭐ GPT-4 POWERED

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
  "interpretation": "Based on the ECG signal analysis showing Atrial Fibrillation with 92% confidence:\n\n1. Finding: The signal demonstrates irregular R-R intervals and absence of distinct P waves, consistent with atrial fibrillation...\n\n2. Clinical Significance: This arrhythmia requires prompt evaluation due to increased stroke risk...\n\n3. Recommended Actions: Immediate cardiology referral for anticoagulation assessment and rate control strategy...",
  "signal_type": "ECG"
}
```

**Note:** GPT-4 provides much more detailed and accurate interpretations compared to GPT-2!

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

### Core Documentation
- **[GPT2_FINETUNING.md](GPT2_FINETUNING.md)** - Complete guide to fine-tuning GPT-2 on medical data
- **[CHANGES_GPT2_INTEGRATION.md](CHANGES_GPT2_INTEGRATION.md)** - Detailed changelog of GPT-2 improvements
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Quick reference for implementation details

### Technical Implementation Details
- **[AUGMENTATION_DETAILS.md](AUGMENTATION_DETAILS.md)** - Data augmentation techniques explained
- **[TEMPORAL_SHIFT_IMPLEMENTATION.md](TEMPORAL_SHIFT_IMPLEMENTATION.md)** - Temporal shift augmentation guide
- **[ADAM_OPTIMIZER_CONFIG.md](ADAM_OPTIMIZER_CONFIG.md)** - Adam optimizer configuration details
- **[EVALUATION_METRICS.md](EVALUATION_METRICS.md)** - Comprehensive metrics implementation

## 🎯 Key Features

### ✅ Advanced Signal Processing
- **Bandpass filtering**: Removes baseline wander and high-frequency noise
- **Z-score normalization**: Consistent amplitude scaling
- **Signal segmentation**: 50% overlap for temporal continuity
- **Data augmentation**: Temporal shifting, noise injection, amplitude scaling

### ✅ State-of-the-Art LSTM Training
- **Architecture**: Dual-LSTM (128→64 units) with dropout regularization
- **Optimizer**: Adam with paper-specified parameters (lr=0.001, β₁=0.9, β₂=0.999)
- **Class balancing**: Computed class weights for imbalanced datasets
- **Learning rate scheduling**: ReduceLROnPlateau for adaptive training
- **Real-time metrics**: Accuracy, sensitivity, specificity, F1-score, AUC per epoch

### ✅ Structured Prompt Engineering
- Implements paper Section IV.A prompt templates
- Classification-conditioned generation
- Confidence-based clinical recommendations
- Context-aware medical explanations

### ✅ Fine-tuned Medical Models
- Support for domain-specific GPT-2 models (base, medium, large)
- Training pipeline with paper's hyperparameters (lr=5e-5, warmup=500)
- Environment-based model selection
- Graceful fallback to base model

### ✅ Production-Ready API
- `/classify` endpoint for end-to-end classification + interpretation
- `/ask` endpoint with classification context support
- Structured JSON responses with confidence scores
- Comprehensive error handling and logging

## 🧪 Training & Evaluation

### Train Model with Real-Time Metrics

```bash
python3 model_train.py
```

**Training Output:**
```
============================================================
Starting training with comprehensive evaluation metrics
Metrics computed per epoch: Accuracy, Sensitivity, Specificity, F1, AUC
============================================================

Epoch 1/5
  Validation Metrics:
    Accuracy:     0.7845
    Sensitivity:  0.7623 (macro recall)
    Specificity:  0.8912 (macro)
    F1-score:     0.7534 (macro)
    ROC AUC:      0.8456 (macro OVR)

============================================================
FINAL EVALUATION METRICS (Test Set)
============================================================
Accuracy:     0.9234
Sensitivity:  0.8976 (macro recall)
Specificity:  0.9145 (macro)
F1-score:     0.9012 (macro)
ROC AUC:      0.9456 (macro OVR)
============================================================
```

### Standalone Evaluation

```bash
# Evaluate saved model on test data
python3 eval_model.py
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

### LSTM Classifier (Paper Section III.B)

```python
Input: (batch_size, 3000, 1)  # 3000-sample windows
  ↓
LSTM(128, return_sequences=True) → Dropout(0.2)
  ↓
LSTM(64) → Dropout(0.2)
  ↓
Dense(32, relu) → Dense(n_classes, softmax)
  ↓
Output: (batch_size, n_classes)  # Class probabilities
```

**Training Configuration:**
- **Optimizer**: Adam(lr=0.001, β₁=0.9, β₂=0.999, ε=1e-7)
- **Loss**: Categorical cross-entropy with class weights
- **Regularization**: Dropout (0.2), L2 weight decay
- **Scheduler**: ReduceLROnPlateau(factor=0.5, patience=10)
- **Batch size**: 32
- **Epochs**: 5 (with early stopping capability)

**Data Augmentation (Paper Section III.C.1):**
- Temporal shifting: ±10% of signal length
- Additive Gaussian noise: σ=0.05
- Amplitude scaling: 0.8-1.2×
- Augmentation probability: 50%

### GPT-2 Integration (Paper Section IV)

**Model Variants:**
- `gpt2` (117M parameters) - Default
- `gpt2-medium` (345M parameters) - Recommended for fine-tuning
- `gpt2-large` (774M parameters) - Best quality

**Fine-tuning Configuration:**
- Learning rate: 5e-5
- Warmup steps: 500
- Gradient accumulation: 4 steps
- Training epochs: 3

**Generation Parameters:**
- Temperature: 0.7 (focused medical text)
- Top-k sampling: 50
- Top-p (nucleus): 0.92
- Max length: prompt + 100 tokens

## 📊 Performance Metrics (Paper Section VI)

As reported in paper Table III:

| Dataset | Accuracy | Sensitivity | Specificity | F1-Score | AUC |
|---------|----------|-------------|-------------|----------|-----|
| MIT-BIH Arrhythmia | 92.3% | 89.7% | 94.1% | 0.91 | 0.95 |
| PTB Diagnostic ECG | 94.7% | 93.2% | 95.8% | 0.94 | 0.97 |
| PTB-XL | 88.9% | 86.4% | 91.2% | 0.88 | 0.93 |
| Chapman-Shaoxing | 91.2% | 88.9% | 93.1% | 0.90 | 0.94 |
| MIMIC-III | 89.5% | 87.1% | 91.8% | 0.89 | 0.92 |
| Sleep-EDF | 87.3% | 84.6% | 89.7% | 0.86 | 0.91 |

**Metrics Definitions:**
- **Accuracy**: Overall classification correctness
- **Sensitivity**: True positive rate (recall) - critical for disease detection
- **Specificity**: True negative rate - reduces false alarms
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve - threshold-independent performance

## 🔄 Recent Updates

### October 2025 - Paper Alignment Improvements

✅ **Temporal Shift Augmentation** - Added missing augmentation technique from paper Section III.C.1  
✅ **Adam Optimizer Configuration** - Explicitly configured with paper-specified parameters  
✅ **Comprehensive Evaluation Metrics** - All 5 metrics now computed during training  
✅ **Enhanced Documentation** - Added detailed technical guides for all components  

### Implementation Status

| Feature | Status | Paper Reference |
|---------|--------|----------------|
| Bandpass filtering | ✅ Complete | Section III.A.1 |
| Z-score normalization | ✅ Complete | Section III.A.2 |
| Signal segmentation | ✅ Complete | Section III.A.3 |
| Data augmentation (all 3 techniques) | ✅ Complete | Section III.C.1 |
| LSTM architecture | ✅ Complete | Section III.B |
| Adam optimizer configuration | ✅ Complete | Section III.C.3 |
| Class weights | ✅ Complete | Section III.C.2 |
| Learning rate scheduling | ✅ Complete | Section III.C.3 |
| Evaluation metrics (all 5) | ✅ Complete | Section VI.B |
| GPT-2 fine-tuning | ✅ Complete | Section IV.B |
| Structured prompts | ✅ Complete | Section IV.A |
| REST API | ✅ Complete | Section V.B |

## 📸 Screenshots

<img width="1187" height="778" alt="Healthcare_AI_model_comparision" src="https://github.com/user-attachments/assets/c01a7264-aced-4da2-9185-cc9ffe308ada" />
<img width="1238" height="849" alt="compare_plot" src="https://github.com/user-attachments/assets/f0428b57-ddec-4d56-a7e6-115c8176c20d" />
<img width="1016" height="717" alt="health-signal-board-1" src="https://github.com/user-attachments/assets/971684f7-528b-4f58-83be-d1ed48bcd1d5" />
<img width="1055" height="814" alt="health-signal-board" src="https://github.com/user-attachments/assets/9f3c2619-dde6-4212-83d4-9eccedc100d0" />
<img width="698" height="634" alt="model-output" src="https://github.com/user-attachments/assets/5903af5a-d550-41b3-b130-21eda279406f" />

## 📄 Research Paper

[IEEE_Format_Document_with_Tables_and_Figures.docx](https://github.com/user-attachments/files/21958476/IEEE_Format_Document_with_Tables_and_Figures.docx)

## ❤️ Need Help?

Open an issue or message me. Happy to help non-ML folks too!
