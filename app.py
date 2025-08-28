# app.py

# Step 1: Import necessary libraries
from flask import Flask, request, jsonify, render_template
from transformers import pipeline

# Step 2: Load the AI model
# This loads the GPT-2 model for text generation. It happens once when the server starts.
print("Loading AI model... This may take a moment.")
try:
    generator = pipeline("text-generation", model="gpt2")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    generator = None

# Step 3: Create the Flask application
app = Flask(__name__)

# Step 4: Define the route for the frontend
# This will serve your main HTML page.
@app.route('/')
def home():
    """
    Renders the main HTML page for the user interface.
    Flask will look for 'index.html' in a 'templates' folder.
    """
    return render_template('index.html')

# Step 5: Define the API route for analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Receives a text prompt from the frontend, runs it through the AI model,
    and returns the generated text completion.
    """
    # Check if the model loaded correctly
    if generator is None:
        return jsonify({"error": "Model is not available."}), 500

    # Get the JSON data sent from the frontend
    data = request.get_json()

    # Validate the input
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid input. 'text' field is missing."}), 400
    
    prompt = data['text']

    # Run the AI model
    try:
        # Generate text based on the prompt
        output = generator(prompt, max_length=40, num_return_sequences=1)
        # Extract the completed text from the model's response
        completed_text = output[0]['generated_text']
        # Send the result back to the frontend
        return jsonify({"completed_text": completed_text})
    except Exception as e:
        return jsonify({"error": f"An error occurred during analysis: {str(e)}"}), 500

# Step 6: Run the server
if __name__ == '__main__':
    # debug=True allows the server to auto-reload when you make code changes
    app.run(debug=True)