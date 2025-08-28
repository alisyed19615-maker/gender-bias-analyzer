# app.py

from flask import Flask, request, jsonify, render_template, url_for
from transformers import pipeline
import re # Using the regular expression library

# --- Load Model ---
print("Loading AI model... This may take a moment.")
try:
    generator = pipeline("text-generation", model="gpt2")
    print("Model loaded successfully!")
except Exception as e:
    generator = None

app = Flask(__name__)

# --- Define Pronouns ---
MALE_PRONOUNS = ['he', 'him', 'his', 'himself']
FEMALE_PRONOUNS = ['she', 'her', 'hers', 'herself']

# --- Main Page Route ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Analysis API Route ---
@app.route('/analyze', methods=['POST'])
def analyze():
    if generator is None:
        return jsonify({"error": "Model is not available."}), 500

    data = request.get_json()
    prompt = data.get('text', '')
    if not prompt:
        return jsonify({"error": "Invalid input."}), 400

    try:
        # --- Run AI Model ---
        output = generator(prompt, max_length=40, num_return_sequences=1)
        completed_text = output[0]['generated_text']

        # --- Count Pronouns ---
        words = re.findall(r'\b\w+\b', completed_text.lower())
        male_count = sum(1 for word in words if word in MALE_PRONOUNS)
        female_count = sum(1 for word in words if word in FEMALE_PRONOUNS)

        # --- Create Chart Data Object ---
        chart_data = {
            "male": male_count,
            "female": female_count
        }

        # --- Send Response with text and chart data ---
        return jsonify({
            "completed_text": completed_text,
            "chart_data": chart_data
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)