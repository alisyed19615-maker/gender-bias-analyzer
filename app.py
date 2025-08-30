from flask import Flask, request, jsonify, render_template
from transformers import pipeline

# --- Load Model 1: The Fast Bias Classifier ---
print("Loading bias detection model...")
try:
    classifier = pipeline("text-classification", model="valurank/distilroberta-bias")
    print("Classifier loaded successfully!")
except Exception as e:
    print(f"Error loading classifier: {e}")
    classifier = None

# --- Load Model 2: The Bias Explainer (LLM) ---
print("Loading text generation model for explanations...")
try:
    # This is a small language model that can generate text.
    explainer = pipeline("text-generation", model="distilgpt2")
    print("Explainer model loaded successfully!")
except Exception as e:
    print(f"Error loading explainer model: {e}")
    explainer = None

app = Flask(__name__)

# --- Main Page Route ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Analysis API Route ---
@app.route('/analyze', methods=['POST'])
def analyze():
    if classifier is None or explainer is None:
        return jsonify({"error": "One or more AI models are not available."}), 500

    data = request.get_json()
    prompt = data.get('text', '')
    if not prompt:
        return jsonify({"error": "Invalid input."}), 400

    try:
        # --- Step 1: Fast Check with the Classifier ---
        results = classifier(prompt)
        top_result = results[0]
        label = top_result['label'].upper()
        score = top_result['score']
        
        stats = {"neutral": 0, "equality": 0, "bias": 0}
        explanation = None # Default explanation is None

        # --- Step 2: If Biased, ask the LLM for an Explanation ---
        if label == 'BIASED':
            stats['bias'] = 1
            
            # Create a specific prompt for the explainer model
            explainer_prompt = f"The sentence '{prompt}' is considered biased because"
            
            # Generate the explanation
            output = explainer(explainer_prompt, max_length=50, num_return_sequences=1)
            generated_text = output[0]['generated_text']
            
            # Clean up the output to get just the explanation part
            explanation = generated_text.replace(explainer_prompt, "").strip()

        else: # The other label is 'NOT BIASED'
            stats['neutral'] = 1
            stats['equality'] = 1 

        classification_data = {
            "label": "Biased" if label == 'BIASED' else "Neutral",
            "score": score
        }

        # --- Send all data back to the frontend ---
        return jsonify({
            "stats": stats,
            "classification": classification_data,
            "explanation": explanation # Will be the text or None
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

