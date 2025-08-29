# app.py

from flask import Flask, request, jsonify, render_template
from transformers import pipeline

# --- Load a NEW, more reliable Classification Model ---
print("Loading text classification model... This may take a moment.")
try:
    # MODIFIED: Switched to a standard, PyTorch-based toxicity model.
    classifier = pipeline("text-classification", model="unitary/toxic-bert")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

# Create the Flask application
app = Flask(__name__)

# --- Main Page Route ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Analysis API Route ---
@app.route('/analyze', methods=['POST'])
def analyze():
    if classifier is None:
        return jsonify({"error": "Model is not available."}), 500

    data = request.get_json()
    prompt = data.get('text', '')
    if not prompt:
        return jsonify({"error": "Invalid input. 'text' field is missing."}), 400

    try:
        results = classifier(prompt)
        top_result = results[0]
        label = top_result['label'].upper()
        
        stats = {"neutral": 0, "equality": 0, "bias": 0}
        
        # MODIFIED: The new model's label for bias is 'TOXIC'.
        if label == 'TOXIC' and top_result['score'] > 0.7: # Only flag if confident
            stats['bias'] = 1
        else: 
            stats['neutral'] = 1
            stats['equality'] = 1 

        return jsonify({"stats": stats})
    except Exception as e:
        return jsonify({"error": f"An error occurred during analysis: {str(e)}"}), 500

# Run the server
if __name__ == '__main__':
    app.run(debug=True)

