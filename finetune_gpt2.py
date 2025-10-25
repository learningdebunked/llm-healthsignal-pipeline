"""
GPT-2 Fine-tuning Script for Medical Domain
Implements the fine-tuning strategy described in Section IV.B of the paper.

This script fine-tunes GPT-2 on medical corpora including:
- ECG interpretations from cardiology textbooks
- EEG reports with clinical summaries
- Clinical guidelines from professional medical associations
- Simplified explanations for patient education

Usage:
    python finetune_gpt2.py --data_dir ./medical_corpus --output_dir ./medical-gpt2 --epochs 3
"""

import os
import argparse
import json
from pathlib import Path
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, load_dataset
import numpy as np


def prepare_medical_corpus(data_dir):
    """
    Prepare medical training corpus from various sources.
    
    Expected directory structure:
        data_dir/
            ecg_interpretations.jsonl
            eeg_reports.jsonl
            clinical_guidelines.txt
            patient_education.txt
    
    Returns:
        Dataset object ready for training
    """
    texts = []
    
    # Load ECG interpretations
    ecg_file = Path(data_dir) / "ecg_interpretations.jsonl"
    if ecg_file.exists():
        print(f"Loading ECG interpretations from {ecg_file}")
        with open(ecg_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Format: classification -> interpretation
                text = f"Medical Signal Analysis Report:\nSignal Type: ECG\nClassification: {data['classification']}\nConfidence: {data.get('confidence', 'N/A')}\n\nInterpretation: {data['interpretation']}"
                texts.append(text)
        print(f"Loaded {len(texts)} ECG interpretations")
    
    # Load EEG reports
    eeg_file = Path(data_dir) / "eeg_reports.jsonl"
    if eeg_file.exists():
        print(f"Loading EEG reports from {eeg_file}")
        eeg_count = 0
        with open(eeg_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                text = f"Medical Signal Analysis Report:\nSignal Type: EEG\nClassification: {data['classification']}\nConfidence: {data.get('confidence', 'N/A')}\n\nInterpretation: {data['interpretation']}"
                texts.append(text)
                eeg_count += 1
        print(f"Loaded {eeg_count} EEG reports")
    
    # Load clinical guidelines
    guidelines_file = Path(data_dir) / "clinical_guidelines.txt"
    if guidelines_file.exists():
        print(f"Loading clinical guidelines from {guidelines_file}")
        with open(guidelines_file, 'r') as f:
            content = f.read()
            # Split into chunks (each guideline section)
            chunks = content.split('\n\n')
            texts.extend([chunk.strip() for chunk in chunks if len(chunk.strip()) > 100])
        print(f"Loaded {len(chunks)} guideline sections")
    
    # Load patient education materials
    education_file = Path(data_dir) / "patient_education.txt"
    if education_file.exists():
        print(f"Loading patient education materials from {education_file}")
        with open(education_file, 'r') as f:
            content = f.read()
            chunks = content.split('\n\n')
            texts.extend([chunk.strip() for chunk in chunks if len(chunk.strip()) > 100])
        print(f"Loaded {len(chunks)} education sections")
    
    if not texts:
        print("WARNING: No training data found. Creating sample medical corpus...")
        texts = create_sample_corpus()
    
    print(f"Total training examples: {len(texts)}")
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    return dataset


def create_sample_corpus():
    """
    Create a small sample corpus for demonstration purposes.
    In production, replace with actual medical literature.
    """
    sample_texts = [
        """Medical Signal Analysis Report:
Signal Type: ECG
Classification: Atrial Fibrillation
Confidence: 92%

Interpretation: Atrial Fibrillation (AFib) is characterized by irregular and often rapid heart rhythm. The ECG shows absence of distinct P waves and irregularly irregular R-R intervals. This arrhythmia increases stroke risk due to potential blood clot formation in the atria. Immediate anticoagulation therapy should be considered, and rate control or rhythm control strategies should be evaluated based on patient symptoms and hemodynamic stability. Refer to cardiology for comprehensive management.""",
        
        """Medical Signal Analysis Report:
Signal Type: ECG
Classification: Normal Sinus Rhythm
Confidence: 96%

Interpretation: The ECG demonstrates normal sinus rhythm with regular P waves preceding each QRS complex. Heart rate is within normal range (60-100 bpm), PR interval is normal (120-200 ms), and QRS duration is normal (<120 ms). No ST segment abnormalities or T wave inversions are noted. This represents normal cardiac electrical activity. Continue routine monitoring and maintain cardiovascular health through lifestyle modifications.""",
        
        """Medical Signal Analysis Report:
Signal Type: ECG
Classification: Ventricular Tachycardia
Confidence: 89%

Interpretation: Ventricular Tachycardia (VT) is a potentially life-threatening arrhythmia characterized by rapid ventricular rate (>100 bpm) with wide QRS complexes (>120 ms). This rhythm originates from the ventricles rather than the normal conduction system. Immediate assessment of hemodynamic stability is critical. Unstable patients require immediate cardioversion. Stable patients should receive antiarrhythmic medications and urgent cardiology consultation. Investigate underlying causes including ischemia, electrolyte abnormalities, and structural heart disease.""",
        
        """Medical Signal Analysis Report:
Signal Type: EEG
Classification: Sleep Stage N3 (Deep Sleep)
Confidence: 91%

Interpretation: The EEG shows characteristic high-amplitude, low-frequency delta waves (0.5-2 Hz) comprising more than 20% of the epoch, consistent with Stage N3 sleep (slow-wave sleep). This is the deepest stage of non-REM sleep, associated with restorative physiological processes. Normal N3 sleep is essential for physical recovery, immune function, and memory consolidation. Adequate deep sleep duration indicates healthy sleep architecture.""",
        
        """Medical Signal Analysis Report:
Signal Type: ECG
Classification: Premature Ventricular Contractions
Confidence: 88%

Interpretation: Premature Ventricular Contractions (PVCs) are early heartbeats originating from the ventricles, appearing as wide QRS complexes occurring before the expected normal beat. Isolated PVCs are common and usually benign in individuals without structural heart disease. However, frequent PVCs (>10% of total beats) or complex patterns may warrant further evaluation including echocardiography and Holter monitoring. Assess for underlying triggers such as caffeine, stress, electrolyte imbalances, or cardiac pathology.""",
        
        """Clinical Guideline: Management of Atrial Fibrillation
Atrial fibrillation management requires a comprehensive approach addressing rate control, rhythm control, and stroke prevention. CHA2DS2-VASc score should be calculated to assess stroke risk and guide anticoagulation decisions. Rate control can be achieved with beta-blockers, calcium channel blockers, or digoxin. Rhythm control strategies include antiarrhythmic drugs or catheter ablation for selected patients. All patients should receive education about symptom recognition and when to seek emergency care.""",
        
        """Patient Education: Understanding Your ECG Results
An electrocardiogram (ECG) is a test that measures the electrical activity of your heart. It helps doctors identify irregular heart rhythms, heart attacks, and other cardiac conditions. The test is painless and takes only a few minutes. Small electrodes are placed on your chest, arms, and legs to record your heart's electrical signals. Your doctor will review the results and explain any findings. If abnormalities are detected, additional tests or treatments may be recommended.""",
    ]
    
    return sample_texts


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize text examples for training"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on medical corpus")
    parser.add_argument("--data_dir", type=str, default="./medical_corpus",
                        help="Directory containing medical training data")
    parser.add_argument("--output_dir", type=str, default="./medical-gpt2",
                        help="Directory to save fine-tuned model")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Base model to fine-tune (gpt2, gpt2-medium, gpt2-large)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GPT-2 Medical Domain Fine-tuning")
    print("=" * 60)
    print(f"Base model: {args.model_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)
    
    # Load tokenizer and model
    print("\nLoading tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have pad token by default
    
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    print(f"Model loaded: {model.num_parameters():,} parameters")
    
    # Prepare dataset
    print("\nPreparing training dataset...")
    dataset = prepare_medical_corpus(args.data_dir)
    
    # Split into train/validation
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")
    
    # Tokenize datasets
    print("\nTokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"]
    )
    tokenized_eval = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"]
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 uses causal language modeling, not masked LM
    )
    
    # Training arguments (as per paper Section IV.B)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to="none",  # Disable wandb/tensorboard for simplicity
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting fine-tuning...")
    print("=" * 60)
    trainer.train()
    
    # Save final model
    print("\nSaving fine-tuned model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("=" * 60)
    print(f"Fine-tuning complete! Model saved to: {args.output_dir}")
    print("\nTo use the fine-tuned model, set environment variable:")
    print(f"export MEDICAL_GPT2_PATH={os.path.abspath(args.output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
