# === 3.2 Cloud Inference Layer ===
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import os

# Initialize LLM pipeline with support for fine-tuned models
llm_pipeline = None
llm_model = None
llm_tokenizer = None

# Path to fine-tuned medical GPT-2 model (if available)
FINETUNED_MODEL_PATH = os.environ.get('MEDICAL_GPT2_PATH', 'gpt2')  # defaults to base gpt2

def initialize_llm(model_path=None):
    """Initialize the LLM pipeline when needed
    
    Args:
        model_path: Path to fine-tuned model. If None, uses FINETUNED_MODEL_PATH env var or base gpt2
    """
    global llm_pipeline, llm_model, llm_tokenizer
    if llm_pipeline is None:
        model_to_load = model_path or FINETUNED_MODEL_PATH
        print(f"Initializing GPT-2 model from: {model_to_load}...")
        try:
            # Try loading as fine-tuned model first
            if os.path.isdir(model_to_load) and model_to_load != 'gpt2':
                print(f"Loading fine-tuned medical model from {model_to_load}")
                llm_model = GPT2LMHeadModel.from_pretrained(model_to_load)
                llm_tokenizer = GPT2Tokenizer.from_pretrained(model_to_load)
                llm_pipeline = pipeline(
                    "text-generation",
                    model=llm_model,
                    tokenizer=llm_tokenizer,
                    device=-1  # Use CPU; set to 0 for GPU
                )
            else:
                # Fall back to base model
                print("Loading base GPT-2 model (no fine-tuning detected)")
                llm_pipeline = pipeline(
                    "text-generation", 
                    model="gpt2",
                    tokenizer="gpt2",
                    device=-1  # Use CPU
                )
            print("GPT-2 model initialized successfully!")
        except Exception as e:
            print(f"Failed to initialize GPT-2: {e}")
            raise e

def complete_signal(signal, method="gan"):
    """
    Uses a generative model to complete or denoise an input signal.
    """
    # Placeholder implementation
    return signal

def classify_signal(signal, model_type="transformer"):
    """
    Uses a selected ML model to classify a given input signal.
    """
    # Placeholder implementation - returns mock classification
    return {"prediction": "normal", "confidence": 0.85}

def explain_with_llm(classification_result, signal_type="ECG", clinical_context=None):
    """
    Uses LLM to provide natural language explanation of model results.
    Implements structured prompt engineering as described in the paper.
    
    Args:
        classification_result: Dict with 'prediction', 'confidence', and optionally 'probabilities'
        signal_type: Type of biomedical signal (ECG, EEG, etc.)
        clinical_context: Optional additional clinical information
    
    Returns:
        Natural language interpretation suitable for non-specialist healthcare workers
    """
    prediction = classification_result.get("prediction", "unknown")
    confidence = classification_result.get("confidence", 0.0)
    probabilities = classification_result.get("probabilities", {})
    
    # Build structured prompt template as per paper Section IV.A
    prompt_template = f"""Medical Signal Analysis Report:
Signal Type: {signal_type}
Classification: {prediction}
Confidence: {confidence:.2%}
Clinical Context: {clinical_context or 'Routine screening'}

Provide a detailed clinical interpretation suitable for primary care practitioners, including:
1. Explanation of the finding
2. Clinical significance
3. Recommended follow-up actions

Interpretation:"""
    
    # Try GPT-2 generation with structured prompt
    try:
        initialize_llm()
        response = llm_pipeline(
            prompt_template,
            max_length=len(prompt_template.split()) + 100,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            temperature=0.7,  # Lower temperature for more focused medical text
            pad_token_id=50256,
            num_return_sequences=1
        )
        generated = response[0]['generated_text']
        
        # Extract interpretation section
        if "Interpretation:" in generated:
            interpretation = generated.split("Interpretation:")[-1].strip()
            if len(interpretation) > 30:  # Ensure substantial response
                return interpretation
    except Exception as e:
        print(f"GPT-2 generation failed: {e}")
    
    # Fallback: structured template-based response
    confidence_level = "high" if confidence > 0.85 else "moderate" if confidence > 0.65 else "low"
    return f"""Analysis of {signal_type} signal:

1. Finding: The signal has been classified as '{prediction}' with {confidence_level} confidence ({confidence:.1%}).

2. Clinical Significance: {'This finding requires immediate attention and specialist review.' if confidence > 0.85 and prediction.lower() not in ['normal', 'normal sinus rhythm'] else 'This finding should be reviewed in the context of patient history and symptoms.'}

3. Recommended Actions: {'Immediate cardiology/neurology consultation recommended.' if confidence > 0.85 and prediction.lower() not in ['normal', 'normal sinus rhythm'] else 'Continue monitoring and correlate with clinical presentation. Consider specialist referral if symptoms persist.'}

Note: This AI-assisted interpretation should be reviewed by a qualified healthcare professional."""

def generate_prompt_based_response(prompt, max_tokens=150, classification_result=None):
    """
    Sends a prompt to the LLM and returns generated natural language text.
    Implements prompt engineering strategies from paper Section IV.A.
    
    Args:
        prompt: User query or context
        max_tokens: Maximum tokens to generate
        classification_result: Optional dict with classification outputs to condition response
    
    Returns:
        Natural language response
    """
    # Medical knowledge base for common queries (fallback)
    medical_responses = {
        "atrial fibrillation": "Atrial Fibrillation (AFib) is an irregular and often rapid heart rhythm that can lead to blood clots in the heart. In AFib, the heart's two upper chambers (atria) beat chaotically and irregularly, out of sync with the two lower chambers (ventricles). This can cause symptoms like palpitations, shortness of breath, and fatigue. It's important to monitor and treat AFib as it increases the risk of stroke and heart failure.",
        "normal sinus rhythm": "Normal Sinus Rhythm indicates a healthy heart rhythm originating from the sinoatrial (SA) node. The heart rate is typically between 60-100 beats per minute with regular intervals between beats. This is the ideal heart rhythm pattern.",
        "ventricular tachycardia": "Ventricular Tachycardia (VT) is a fast heart rhythm that starts in the ventricles. It can be life-threatening if sustained, as it may prevent the heart from pumping blood effectively. Immediate medical attention is often required.",
        "bradycardia": "Bradycardia is a slower than normal heart rate, typically below 60 beats per minute. While it can be normal in athletes, it may indicate underlying heart problems in others and can cause dizziness, fatigue, or fainting.",
        "premature ventricular contraction": "Premature Ventricular Contractions (PVCs) are extra heartbeats that begin in the ventricles. They are common and usually harmless, but frequent PVCs may require evaluation.",
        "sleep apnea": "Sleep apnea is a disorder where breathing repeatedly stops and starts during sleep. It can lead to daytime fatigue, cardiovascular problems, and other health issues if untreated."
    }
    
    # If classification result provided, use structured clinical prompt
    if classification_result:
        prediction = classification_result.get("prediction", "unknown")
        confidence = classification_result.get("confidence", 0.0)
        
        structured_prompt = f"""Clinical Query: {prompt}
Diagnostic Finding: {prediction}
Confidence Level: {confidence:.1%}

Provide a clear, accessible explanation for healthcare workers addressing:
- What this finding means
- Clinical implications
- Recommended next steps

Response:"""
    else:
        # General medical query prompt
        structured_prompt = f"""Medical Question: {prompt}

Provide an educational explanation suitable for healthcare practitioners, covering:
- Key medical concepts
- Clinical relevance
- Important considerations

Answer:"""
    
    # Try GPT-2 generation with structured prompt
    try:
        initialize_llm()
        response = llm_pipeline(
            structured_prompt, 
            max_length=len(structured_prompt.split()) + max_tokens,
            do_sample=True, 
            top_k=50,
            top_p=0.92,
            temperature=0.7,  # Balanced for medical accuracy
            pad_token_id=50256,
            num_return_sequences=1
        )
        generated = response[0]['generated_text']
        
        # Extract response section
        for delimiter in ["Response:", "Answer:"]:
            if delimiter in generated:
                result = generated.split(delimiter)[-1].strip()
                if len(result) > 30:  # Ensure substantial response
                    return result
    except Exception as e:
        print(f"GPT-2 generation error: {e}")
    
    # Fallback to knowledge base
    prompt_lower = prompt.lower()
    for condition, explanation in medical_responses.items():
        if condition in prompt_lower:
            # Enhance with classification context if available
            if classification_result:
                confidence = classification_result.get("confidence", 0.0)
                return f"{explanation}\n\nCurrent Analysis: Detected with {confidence:.1%} confidence. Please correlate with patient history and clinical presentation."
            return explanation
    
    # Generic medical response
    if any(term in prompt_lower for term in ["ecg", "eeg", "heart", "cardiac", "rhythm", "signal", "arrhythmia", "sleep"]):
        return f"""Medical Signal Analysis Query: {prompt}

This healthcare AI system provides educational information about biomedical signals and diagnostic patterns. The query relates to physiological signal analysis which typically requires:

1. Professional medical interpretation in clinical context
2. Correlation with patient symptoms and history  
3. Confirmation through additional diagnostic tests when indicated

For specific medical advice and treatment decisions, please consult with a qualified healthcare professional.

Note: This system is designed as a decision support tool for healthcare workers in underserved regions, not as a replacement for clinical judgment."""
    
    return f"""Healthcare AI Response: {prompt}

This system is designed to provide educational information about biomedical signals and medical conditions. For personalized medical advice, diagnosis, or treatment recommendations, please consult with a qualified healthcare professional.

The AI-assisted diagnostic pipeline integrates signal processing and natural language generation to support healthcare delivery in resource-constrained environments."""

