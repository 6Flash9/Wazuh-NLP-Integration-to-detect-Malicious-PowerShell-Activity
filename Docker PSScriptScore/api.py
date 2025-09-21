# api.py

from flask import Flask, request, jsonify
import joblib

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- Load the Model and Vectorizer ---
# These are loaded only once when the API starts up for efficiency.
print("Loading model and vectorizer...")
try:
    model = joblib.load('model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    print("✅ Model and vectorizer loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'model.joblib' or 'vectorizer.joblib' not found.")
    print("Please make sure the files are in the same directory as api.py.")
    # In a real app, you might want to exit or handle this more gracefully.
    model = None
    vectorizer = None


# --- Define the Prediction Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze_command():
    """
    Analyzes a command string and returns its classification.
    Expects a JSON payload like: {"command": "some command string"}
    """
    if not model or not vectorizer:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    # Get the JSON data from the request
    data = request.get_json()

    # Ensure the 'command' key exists
    if not data or 'command' not in data:
        return jsonify({"error": "Missing 'command' key in JSON payload."}), 400

    command_to_analyze = data['command']

    # 1. Transform the input command using the loaded vectorizer
    command_tfidf = vectorizer.transform([command_to_analyze])

    # 2. Predict the class (0 for benign, 1 for malicious)
    prediction = model.predict(command_tfidf)[0]

    # 3. Get the confidence score (probability of being malicious)
    confidence_score = model.predict_proba(command_tfidf)[0][1]

    # 4. Prepare the JSON response
    result = "malicious" if prediction == 1 else "benign"
    response = {
        "command": command_to_analyze,
        "class": result,
        "score": round(confidence_score, 4) # Round for a cleaner output
    }

    return jsonify(response)


# --- Health Check Endpoint (Good Practice) ---
@app.route('/health', methods=['GET'])
def health_check():
    """A simple endpoint to check if the API is running."""
    return jsonify({"status": "ok"}), 200


# --- Run the App ---
if __name__ == '__main__':
    # Use host='0.0.0.0' to make the API accessible from other machines
    # on your network, like your Wazuh manager VM.
    app.run(host='0.0.0.0', port=5000, debug=True)