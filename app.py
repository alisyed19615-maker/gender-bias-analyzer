# app.py

from flask import Flask, request, jsonify, render_template
from transformers import pipeline

# --- Load the More Robust Classification Model ---
print("Loading toxicity detection model...")
try:
    # This model is better at detecting insults and overt bias.
    classifier = pipeline("text-classification", model="unitary/toxic-bert")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading classifier: {e}")
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
        return jsonify({"error": "AI model is not available."}), 500

    data = request.get_json()
    prompt = data.get('text', '')
    if not prompt:
        return jsonify({"error": "Invalid input."}), 400

    try:
        # --- Step 1: Classify the text ---
        results = classifier(prompt)
        top_result = results[0]
        # This model uses the label 'toxic' for negative content
        label = top_result['label'].upper()
        score = top_result['score']
        
        stats = {"neutral": 0, "equality": 0, "bias": 0}
        is_truly_biased = False

        # --- Check if the model confidently flags the text as TOXIC ---
        # We'll treat "TOXIC" as our indicator for "Biased"
        if label == 'TOXIC' and score > 0.75:
            is_truly_biased = True
            stats['bias'] = 1
        else:
            stats['neutral'] = 1
            stats['equality'] = 1 

        classification_data = {
            "label": "Biased" if is_truly_biased else "Neutral",
            "score": score
        }

        # --- Return the data to the frontend ---
        return jsonify({
            "stats": stats,
            "classification": classification_data
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)