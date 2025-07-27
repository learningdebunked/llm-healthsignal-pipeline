# === 3.3 Frontend ===
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from flask import Flask, request, jsonify, render_template
from inference import generate_prompt_based_response

app = Flask(__name__)

@app.route("/")
def index():
    """
    Root route that redirects to dashboard
    """
    return render_template("dashboard.html")

@app.route("/dashboard")
def dashboard():
    """
    Serves the dashboard HTML page (UI not included here).
    """
    return render_template("dashboard.html")

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
    Accepts a prompt from the frontend, runs it through the LLM, and returns the output.
    """
    user_input = request.json.get("prompt", "")
    response = generate_prompt_based_response(user_input)
    return jsonify({"response": response})

def log_feedback(feedback_data):
    """
    Appends feedback to a log file for monitoring and evaluation.
    """
    with open("feedback_log.json", "a") as f:
        f.write(str(feedback_data) + "\n")

if __name__ == "__main__":
    print("🩺 Starting Healthcare AI API Server...")
    print("📊 Dashboard available at: http://localhost:3333")
    print("🔗 API endpoints: /ask, /feedback, /dashboard")
    app.run(debug=True, port=3333, host='127.0.0.1')