# === 3.2 Cloud Inference Layer ===
from openai import OpenAI
import numpy as np
import os

# Initialize OpenAI client
client = None
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

def initialize_llm(api_key=None):
    """Initialize the OpenAI GPT-4 client
    
    Args:
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var
    """
    global client, OPENAI_API_KEY
    if client is None:
        key = api_key or OPENAI_API_KEY
        if not key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable.\n"
                "Get your API key from: https://platform.openai.com/api-keys"
            )
        client = OpenAI(api_key=key)
        print("✓ GPT-4 client initialized successfully!")
        return True
    return True

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
    Uses GPT-4 to provide natural language explanation of model results.
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
    prompt_template = f"""You are a medical AI assistant helping primary care practitioners interpret biomedical signal analysis results.

Medical Signal Analysis Report:
- Signal Type: {signal_type}
- Classification: {prediction}
- Confidence: {confidence:.2%}
- Clinical Context: {clinical_context or 'Routine screening'}

Provide a detailed clinical interpretation including:
1. Explanation of the finding
2. Clinical significance
3. Recommended follow-up actions

Keep the response professional, clear, and actionable for healthcare workers in underserved regions."""
    
    # Try GPT-4 generation with structured prompt
    try:
        initialize_llm()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant specializing in biomedical signal interpretation. Provide clear, accurate clinical guidance suitable for primary care practitioners."},
                {"role": "user", "content": prompt_template}
            ],
            temperature=0.7,
            max_tokens=300,
            top_p=0.9
        )
        interpretation = response.choices[0].message.content.strip()
        if len(interpretation) > 30:
            return interpretation
    except Exception as e:
        print(f"GPT-4 generation failed: {e}")
        # Fallback: structured template-based response
        confidence_level = "high" if confidence > 0.85 else "moderate" if confidence > 0.65 else "low"
        return f"""Analysis of {signal_type} signal:

1. Finding: The signal has been classified as '{prediction}' with {confidence_level} confidence ({confidence:.1%}).

2. Clinical Significance: {'This finding requires immediate attention and specialist review.' if confidence > 0.85 and prediction.lower() not in ['normal', 'normal sinus rhythm'] else 'This finding should be reviewed in the context of patient history and symptoms.'}

3. Recommended Actions: {'Immediate cardiology/neurology consultation recommended.' if confidence > 0.85 and prediction.lower() not in ['normal', 'normal sinus rhythm'] else 'Continue monitoring and correlate with clinical presentation. Consider specialist referral if symptoms persist.'}

Note: This AI-assisted interpretation should be reviewed by a qualified healthcare professional."""

def generate_prompt_based_response(prompt, max_tokens=150, classification_result=None):
    """
    Sends a prompt to GPT-4 and returns generated natural language text.
    Implements prompt engineering strategies from paper Section IV.A.
    
    Args:
        prompt: User query or context
        max_tokens: Maximum tokens to generate
        classification_result: Optional dict with classification outputs to condition response
    
    Returns:
        Natural language response
    """
    # If classification result provided, use structured clinical prompt
    if classification_result:
        prediction = classification_result.get("prediction", "unknown")
        confidence = classification_result.get("confidence", 0.0)
        
        user_message = f"""Clinical Query: {prompt}

Diagnostic Context:
- Finding: {prediction}
- Confidence Level: {confidence:.1%}

Please provide a clear, accessible explanation for healthcare workers addressing:
- What this finding means
- Clinical implications
- Recommended next steps"""
    else:
        # General medical query prompt
        user_message = f"""Medical Question: {prompt}

Provide an educational explanation suitable for healthcare practitioners, covering:
- Key medical concepts
- Clinical relevance
- Important considerations"""
    
    # Try GPT-4 generation with structured prompt
    try:
        initialize_llm()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant providing educational information about biomedical signals and medical conditions for healthcare workers in underserved regions. Provide clear, accurate, and actionable guidance."},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=0.9
        )
        result = response.choices[0].message.content.strip()
        if len(result) > 30:
            return result
    except Exception as e:
        print(f"GPT-4 generation error: {e}")
        # Fallback to knowledge base
        return fallback_medical_response(prompt, classification_result)
    
    return fallback_medical_response(prompt, classification_result)

def fallback_medical_response(prompt, classification_result=None):
    """Fallback responses when GPT-4 is unavailable"""
    # Medical knowledge base for common queries
    medical_responses = {
        "atrial fibrillation": "Atrial Fibrillation (AFib) is an irregular and often rapid heart rhythm that can lead to blood clots in the heart. In AFib, the heart's two upper chambers (atria) beat chaotically and irregularly, out of sync with the two lower chambers (ventricles). This can cause symptoms like palpitations, shortness of breath, and fatigue. It's important to monitor and treat AFib as it increases the risk of stroke and heart failure.",
        "normal sinus rhythm": "Normal Sinus Rhythm indicates a healthy heart rhythm originating from the sinoatrial (SA) node. The heart rate is typically between 60-100 beats per minute with regular intervals between beats. This is the ideal heart rhythm pattern.",
        "ventricular tachycardia": "Ventricular Tachycardia (VT) is a fast heart rhythm that starts in the ventricles. It can be life-threatening if sustained, as it may prevent the heart from pumping blood effectively. Immediate medical attention is often required.",
        "bradycardia": "Bradycardia is a slower than normal heart rate, typically below 60 beats per minute. While it can be normal in athletes, it may indicate underlying heart problems in others and can cause dizziness, fatigue, or fainting.",
        "premature ventricular contraction": "Premature Ventricular Contractions (PVCs) are extra heartbeats that begin in the ventricles. They are common and usually harmless, but frequent PVCs may require evaluation.",
        "sleep apnea": "Sleep apnea is a disorder where breathing repeatedly stops and starts during sleep. It can lead to daytime fatigue, cardiovascular problems, and other health issues if untreated."
    }
    
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

