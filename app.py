# app.py

from flask import Flask, request, jsonify, render_template
from transformers import pipeline

# --- Load a NEW, more specific Bias Detection Model ---
print("Loading bias detection model... This may take a moment.")
try:
    # MODIFIED: Switched to a model specifically trained for bias detection.
    classifier = pipeline("text-classification", model="valurank/distilroberta-bias")
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
        score = top_result['score']
        
        stats = {"neutral": 0, "equality": 0, "bias": 0}
        
        # MODIFIED: The new model's label for bias is 'BIASED'.
        if label == 'BIASED':
            stats['bias'] = 1
        else: # The other label is 'NOT BIASED'
            stats['neutral'] = 1
            stats['equality'] = 1 

        classification_data = {
            # MODIFIED: Use the correct label name from the new model
            "label": "Biased" if label == 'BIASED' else "Neutral",
            "score": score
        }

        return jsonify({
            "stats": stats,
            "classification": classification_data 
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred during analysis: {str(e)}"}), 500

# Run the server
if __name__ == '__main__':
    app.run(debug=True)

