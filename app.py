
from flask import Flask, request, jsonify, render_template
from transformers import pipeline

# --- Load the Classification Model ---
print("Loading bias detection model... This may take a moment.")
try:
    # This model is specifically trained to classify text as biased or not.
    classifier = pipeline("text-classification", model="d4data/bias-detection-model")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

# Create the Flask application
app = Flask(__name__)

# --- Main Page Route ---
@app.route('/')
def home():
    """
    Renders the main HTML page from the 'templates' folder.
    """
    return render_template('index.html')

# --- Analysis API Route ---
@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Receives text from the frontend, classifies it using the AI model,
    and returns a stats object for the UI grid.
    """
    if classifier is None:
        return jsonify({"error": "Model is not available."}), 500

    data = request.get_json()
    prompt = data.get('text', '')
    if not prompt:
        return jsonify({"error": "Invalid input. 'text' field is missing."}), 400

    try:
        # Run the classification model on the user's text
        results = classifier(prompt)
        top_result = results[0]
        label = top_result['label'].upper()
        
        # Create the stats object that the frontend expects
        stats = {"neutral": 0, "equality": 0, "bias": 0}
        
        if label == 'BIASED':
            stats['bias'] = 1
        else: # Assumes the other label is 'NEUTRAL' or similar
            stats['neutral'] = 1
            # For this demo, we can say that neutral text promotes equality
            stats['equality'] = 1 

        # We are not using chart data anymore, so we only return the stats
        return jsonify({
            "stats": stats
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred during analysis: {str(e)}"}), 500

# Run the server
if __name__ == '__main__':
    app.run(debug=True)

