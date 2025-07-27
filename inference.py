# === 3.2 Cloud Inference Layer ===
from transformers import pipeline
import numpy as np

# Initialize a simple text generation pipeline
llm_pipeline = None

def initialize_llm():
    """Initialize the LLM pipeline when needed"""
    global llm_pipeline
    if llm_pipeline is None:
        print("Initializing GPT-2 model...")
        try:
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

def explain_with_llm(classification_result):
    """
    Uses LLM to provide natural language explanation of model results.
    """
    # Simple explanation based on classification result
    prediction = classification_result.get("prediction", "unknown")
    confidence = classification_result.get("confidence", 0.0)
    return f"The signal appears to be {prediction} with {confidence*100:.1f}% confidence."

def generate_prompt_based_response(prompt, max_tokens=100):
    """
    Sends a prompt to the LLM and returns generated natural language text.
    """
    # Medical knowledge base for common queries
    medical_responses = {
        "atrial fibrillation": "Atrial Fibrillation (AFib) is an irregular and often rapid heart rhythm that can lead to blood clots in the heart. In AFib, the heart's two upper chambers (atria) beat chaotically and irregularly, out of sync with the two lower chambers (ventricles). This can cause symptoms like palpitations, shortness of breath, and fatigue. It's important to monitor and treat AFib as it increases the risk of stroke and heart failure.",
        "normal sinus rhythm": "Normal Sinus Rhythm indicates a healthy heart rhythm originating from the sinoatrial (SA) node. The heart rate is typically between 60-100 beats per minute with regular intervals between beats. This is the ideal heart rhythm pattern.",
        "ventricular tachycardia": "Ventricular Tachycardia (VT) is a fast heart rhythm that starts in the ventricles. It can be life-threatening if sustained, as it may prevent the heart from pumping blood effectively. Immediate medical attention is often required.",
        "bradycardia": "Bradycardia is a slower than normal heart rate, typically below 60 beats per minute. While it can be normal in athletes, it may indicate underlying heart problems in others and can cause dizziness, fatigue, or fainting."
    }
    
    # Try to use GPT-2 first, fall back to knowledge base
    try:
        initialize_llm()
        medical_prompt = f"Medical explanation: {prompt}\n\nExplanation:"
        response = llm_pipeline(
            medical_prompt, 
            max_length=len(medical_prompt.split()) + 50,
            do_sample=True, 
            top_k=40, 
            temperature=0.8,
            pad_token_id=50256
        )
        generated = response[0]['generated_text']
        if "Explanation:" in generated:
            result = generated.split("Explanation:")[-1].strip()
            if len(result) > 20:  # Only return if we got a substantial response
                return result
    except Exception as e:
        print(f"GPT-2 Error: {e}")
    
    # Fallback to knowledge base
    prompt_lower = prompt.lower()
    for condition, explanation in medical_responses.items():
        if condition in prompt_lower:
            return explanation
    
    # Generic medical response
    if any(term in prompt_lower for term in ["ecg", "eeg", "heart", "cardiac", "rhythm", "signal"]):
        return f"This appears to be a medical signal analysis query about: {prompt}. The system has detected cardiac or neurological signal patterns that would typically require professional medical interpretation. Please consult with a healthcare provider for proper diagnosis and treatment recommendations."
    
    return f"I understand you're asking about: {prompt}. This healthcare AI system is designed to provide educational information about medical signals and conditions. For specific medical advice, please consult with a qualified healthcare professional."

