# === 3.3 Frontend ===
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Print model configuration on startup
medical_model_path = os.environ.get('MEDICAL_GPT2_PATH', 'gpt2')
if medical_model_path != 'gpt2' and os.path.isdir(medical_model_path):
    print(f"✓ Fine-tuned medical GPT-2 model configured: {medical_model_path}")
else:
    print("⚠ Using base GPT-2 model (no fine-tuning). Set MEDICAL_GPT2_PATH for domain-specific model.")

from flask import Flask, request, jsonify, render_template
from inference import generate_prompt_based_response, explain_with_llm, classify_signal
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    """
    Root route that redirects to dashboard
    """
    try:
        return render_template("dashboard.html")
    except:
        # Fallback if template not found
        return jsonify({
            "message": "Healthcare AI API Server",
            "endpoints": {
                "/ask": "POST - General medical queries with optional classification context",
                "/classify": "POST - Signal classification with LLM interpretation",
                "/feedback": "POST - User feedback collection",
                "/dashboard": "GET - Web interface (if template available)"
            },
            "version": "1.0",
            "model": os.environ.get('MEDICAL_GPT2_PATH', 'gpt2 (base)')
        })

@app.route("/dashboard")
def dashboard():
    """
    Serves the dashboard HTML page (UI not included here).
    """
    try:
        return render_template("dashboard.html")
    except:
        return jsonify({
            "error": "Dashboard template not found",
            "message": "Use API endpoints directly or create templates/dashboard.html"
        }), 404

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
    Accepts a prompt from the frontend, optionally with signal data and classification results.
    Returns LLM-generated interpretation using structured prompts.
    """
    data = request.json
    user_input = data.get("prompt", "")
    
    # Check if signal data and classification provided
    signal_data = data.get("signal", None)
    classification_result = data.get("classification", None)
    signal_type = data.get("signal_type", "ECG")
    clinical_context = data.get("clinical_context", None)
    
    # If signal provided but no classification, run classification
    if signal_data is not None and classification_result is None:
        try:
            signal_array = np.array(signal_data)
            classification_result = classify_signal(signal_array)
        except Exception as e:
            print(f"Classification error: {e}")
            classification_result = None
    
    # Generate response with classification context if available
    if classification_result:
        # Use explain_with_llm for structured clinical interpretation
        if not user_input or user_input.lower() in ["explain", "interpret", "what does this mean"]:
            response = explain_with_llm(
                classification_result,
                signal_type=signal_type,
                clinical_context=clinical_context
            )
        else:
            # User has specific question, use generate_prompt_based_response with context
            response = generate_prompt_based_response(
                user_input,
                classification_result=classification_result
            )
        
        return jsonify({
            "response": response,
            "classification": classification_result.get("prediction"),
            "confidence": classification_result.get("confidence")
        })
    else:
        # No classification data, general medical query
        response = generate_prompt_based_response(user_input)
        return jsonify({"response": response})

def log_feedback(feedback_data):
    """
    Appends feedback to a log file for monitoring and evaluation.
    """
    with open("feedback_log.json", "a") as f:
        f.write(str(feedback_data) + "\n")

@app.route("/classify", methods=['POST'])
def classify():
    """
    Endpoint for signal classification with LLM interpretation.
    Accepts signal data and returns classification + natural language explanation.
    """
    data = request.json
    signal_data = data.get("signal", None)
    signal_type = data.get("signal_type", "ECG")
    clinical_context = data.get("clinical_context", None)
    
    if signal_data is None:
        return jsonify({"error": "No signal data provided"}), 400
    
    try:
        # Convert to numpy array and classify
        signal_array = np.array(signal_data)
        classification_result = classify_signal(signal_array)
        
        # Generate natural language interpretation
        interpretation = explain_with_llm(
            classification_result,
            signal_type=signal_type,
            clinical_context=clinical_context
        )
        
        return jsonify({
            "classification": classification_result.get("prediction"),
            "confidence": classification_result.get("confidence"),
            "interpretation": interpretation,
            "signal_type": signal_type
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🩺 Starting Healthcare AI API Server...")
    print("📊 Dashboard available at: http://localhost:3333")
    print("🔗 API endpoints:")
    print("   - /ask: General medical queries with optional classification context")
    print("   - /classify: Signal classification with LLM interpretation")
    print("   - /feedback: User feedback collection")
    print("   - /dashboard: Web interface")
    print("\n💡 Tip: Set MEDICAL_GPT2_PATH environment variable to use fine-tuned model")
    app.run(debug=True, port=3333, host='127.0.0.1')