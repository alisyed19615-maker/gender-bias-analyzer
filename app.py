# app.py

import os
from flask import Flask, request, jsonify, render_template, url_for
from transformers import pipeline
import re
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
import time

# --- Load Model ---
print("Loading AI model...")
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
        
        # --- NEW: Generate Chart with Matplotlib ---
        chart_url = None
        # Only create a chart if there are pronouns to show
        if male_count > 0 or female_count > 0:
            labels = ['Male Pronouns', 'Female Pronouns']
            sizes = [male_count, female_count]
            colors = ['#36A2EB', '#FF6384'] # Blue, Red

            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.title('Pronoun Distribution')
            
            # Create a unique filename to avoid browser caching issues
            timestamp = int(time.time())
            filename = f'chart_{timestamp}.png'
            filepath = os.path.join('static', filename)
            plt.savefig(filepath)
            plt.close(fig) # Close the figure to free memory
            
            # Get the URL for the frontend
            chart_url = url_for('static', filename=filename)

        # --- Send Response ---
        return jsonify({
            "completed_text": completed_text,
            "chart_url": chart_url # This will be the URL or None
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)