# GPT-2 Fine-tuning for Medical Domain

This document describes the GPT-2 fine-tuning implementation as outlined in Section IV.B of the paper.

## Overview

The system supports loading fine-tuned GPT-2 models trained on medical corpora to provide domain-specific clinical interpretations. The fine-tuning process adapts the base GPT-2 model to generate accurate, accessible medical explanations suitable for non-specialist healthcare workers.

## Fine-tuning Dataset

As described in the paper, the fine-tuning dataset comprises:

- **10,000 annotated ECG interpretations** from cardiology textbooks
- **5,000 EEG reports** with corresponding clinical summaries
- **Clinical guidelines** from professional medical associations
- **Simplified explanations** designed for patient education

## Preparing Your Training Data

Create a directory structure with your medical corpus:

```
medical_corpus/
├── ecg_interpretations.jsonl
├── eeg_reports.jsonl
├── clinical_guidelines.txt
└── patient_education.txt
```

### Format for JSONL files

Each line should be a JSON object with the following structure:

```json
{"classification": "Atrial Fibrillation", "confidence": "92%", "interpretation": "Atrial Fibrillation (AFib) is characterized by..."}
```

### Format for text files

Plain text with sections separated by double newlines (`\n\n`).

## Running Fine-tuning

### Basic Usage

```bash
python finetune_gpt2.py \
    --data_dir ./medical_corpus \
    --output_dir ./medical-gpt2 \
    --epochs 3
```

### Advanced Configuration

```bash
python finetune_gpt2.py \
    --data_dir ./medical_corpus \
    --output_dir ./medical-gpt2 \
    --model_name gpt2-medium \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --max_length 512 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 500
```

### Parameters (as per paper Section IV.B)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `gpt2` | Base model (gpt2, gpt2-medium, gpt2-large) |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `4` | Per-device batch size |
| `--learning_rate` | `5e-5` | Learning rate (paper: 5×10⁻⁵) |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation steps |
| `--warmup_steps` | `500` | Number of warmup steps |
| `--max_length` | `512` | Maximum sequence length |

## Using the Fine-tuned Model

### Set Environment Variable

After fine-tuning, configure the system to use your model:

```bash
export MEDICAL_GPT2_PATH=/path/to/medical-gpt2
```

### Verify Model Loading

Start the API server and check the startup message:

```bash
python api.py
```

You should see:
```
✓ Fine-tuned medical GPT-2 model configured: /path/to/medical-gpt2
```

### Programmatic Usage

```python
from inference import initialize_llm, explain_with_llm, generate_prompt_based_response

# Initialize with fine-tuned model
initialize_llm(model_path="/path/to/medical-gpt2")

# Generate clinical interpretation
classification_result = {
    "prediction": "Atrial Fibrillation",
    "confidence": 0.92
}

interpretation = explain_with_llm(
    classification_result,
    signal_type="ECG",
    clinical_context="Patient presents with palpitations"
)

print(interpretation)
```

## Structured Prompt Templates

The system implements structured prompt engineering as described in the paper (Section IV.A):

### Clinical Interpretation Prompt

```
Medical Signal Analysis Report:
Signal Type: {signal_type}
Classification: {classification}
Confidence: {confidence:.2%}
Clinical Context: {context}

Provide a detailed clinical interpretation suitable for primary care practitioners, including:
1. Explanation of the finding
2. Clinical significance
3. Recommended follow-up actions

Interpretation:
```

### General Medical Query Prompt

```
Medical Question: {prompt}

Provide an educational explanation suitable for healthcare practitioners, covering:
- Key medical concepts
- Clinical relevance
- Important considerations

Answer:
```

### Clinical Query with Context

```
Clinical Query: {prompt}
Diagnostic Finding: {prediction}
Confidence Level: {confidence:.1%}

Provide a clear, accessible explanation for healthcare workers addressing:
- What this finding means
- Clinical implications
- Recommended next steps

Response:
```

## API Endpoints with Fine-tuned Model

### `/classify` - Signal Classification with Interpretation

```bash
curl -X POST http://localhost:3333/classify \
  -H "Content-Type: application/json" \
  -d '{
    "signal": [0.1, 0.2, ...],
    "signal_type": "ECG",
    "clinical_context": "Patient with chest pain"
  }'
```

Response:
```json
{
  "classification": "Atrial Fibrillation",
  "confidence": 0.92,
  "interpretation": "Atrial Fibrillation (AFib) is characterized by...",
  "signal_type": "ECG"
}
```

### `/ask` - Medical Query with Classification Context

```bash
curl -X POST http://localhost:3333/ask \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the treatment options?",
    "classification": {
      "prediction": "Atrial Fibrillation",
      "confidence": 0.92
    },
    "signal_type": "ECG"
  }'
```

## Model Quality Assessment

As reported in the paper (Table V), the fine-tuned model achieves:

| Metric | Score (1-5) | Inter-rater Agreement |
|--------|-------------|----------------------|
| Clinical Accuracy | 4.3 | 0.87 |
| Relevance | 4.5 | 0.91 |
| Clarity | 4.6 | 0.89 |
| Actionability | 4.2 | 0.85 |

## Fallback Behavior

If no fine-tuned model is available, the system:

1. Attempts to use base GPT-2 for generation
2. Falls back to knowledge-base responses for common conditions
3. Provides structured template-based interpretations using confidence levels

## Hardware Requirements

### Minimum (CPU-only)
- 8 GB RAM
- 10 GB disk space

### Recommended (GPU)
- NVIDIA GPU with 8+ GB VRAM
- 16 GB system RAM
- 20 GB disk space

### Paper Configuration (for reference)
- 16 nodes × 8× NVIDIA A100 (80GB) GPUs
- Mixed-precision training
- Bayesian hyperparameter optimization

## Troubleshooting

### Model not loading

```
⚠ Using base GPT-2 model (no fine-tuning). Set MEDICAL_GPT2_PATH for domain-specific model.
```

**Solution**: Verify the path exists and contains `config.json`, `pytorch_model.bin`, and tokenizer files.

### Out of memory during fine-tuning

**Solutions**:
- Reduce `--batch_size` (try 2 or 1)
- Reduce `--max_length` (try 256)
- Increase `--gradient_accumulation_steps`
- Use a smaller base model (`gpt2` instead of `gpt2-medium`)

### Poor generation quality

**Solutions**:
- Increase training data size and diversity
- Train for more epochs
- Adjust temperature (lower = more focused, higher = more creative)
- Verify training data quality and formatting

## Sample Training Data

If you don't have medical corpus data, the script includes a small sample corpus for demonstration. To use it:

```bash
python finetune_gpt2.py --data_dir ./nonexistent_dir --output_dir ./demo-medical-gpt2
```

The script will automatically create sample training examples and proceed with fine-tuning.

## Next Steps

1. **Collect domain-specific data**: Gather ECG/EEG interpretations from medical literature
2. **Fine-tune the model**: Run the training script with your corpus
3. **Evaluate quality**: Test interpretations with medical professionals
4. **Deploy**: Set `MEDICAL_GPT2_PATH` and restart the API server
5. **Monitor**: Collect feedback via `/feedback` endpoint for continuous improvement

## References

- Paper Section IV: Natural Language Generation
- Paper Section IV.A: Prompt Engineering
- Paper Section IV.B: Fine-tuning Strategy
- Paper Table V: Natural Language Generation Quality Assessment
