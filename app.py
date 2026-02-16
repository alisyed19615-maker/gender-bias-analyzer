from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import sqlite3
import datetime
import re
import os

print("Loading toxicity detection model...")
try:
    classifier = pipeline("text-classification", model="unitary/toxic-bert")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading classifier: {e}")
    classifier = None

app = Flask(__name__)

# --- Configuration & Constants ---
DB_NAME = "bias_history.db"

# A basic dictionary for demonstration. 
BIAS_DICTIONARY = {
    r"\bmankind\b": "humanity",
    r"\bmanpower\b": "workforce",
    r"\bfireman\b": "firefighter",
    r"\bpoliceman\b": "police officer",
    r"\bstewardess\b": "flight attendant",
    r"\bchairman\b": "chairperson",
    r"\bmailman\b": "mail carrier",
    r"\bhis\b": "their",
    r"\bhe\b": "they",
    r"\bhim\b": "them",
    r"\bguys\b": "everyone",
    r"\bladies\b": "everyone",
    r"\bgirls\b": "everyone"
}

# --- Database Setup ---
def init_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS history
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      text TEXT,
                      label TEXT,
                      score REAL,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()
        print("Database initialized.")
    except Exception as e:
        print(f"Database error: {e}")

init_db()

# --- Helper Functions ---
def analyze_text_issues(text):
    """
    Finds problematic words based on the dictionary and suggests a rewrite.
    Returns:
        highlights (list): List of words found in the text that match the dictionary.
        rewrite_suggestion (str): The text with substitutions made, or None if no changes.
    """
    highlights = []
    rewrite_text = text
    
    # Check for matches and replace
    found_issue = False
    for pattern, replacement in BIAS_DICTIONARY.items():
        # Find all matches for highlighting
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            highlights.append(match.group(0))
            found_issue = True
        
        # Perform substitution for rewrite
        rewrite_text = re.sub(pattern, replacement, rewrite_text, flags=re.IGNORECASE)
        
    regex_highlights = list(set(highlights))
    
    return regex_highlights, rewrite_text if found_issue else None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if classifier is None:
        return jsonify({"error": "AI model is not available. Please check server logs."}), 500

    data = request.get_json()
    prompt = data.get('text', '')
    if not prompt:
        return jsonify({"error": "Invalid input."}), 400

    try:
        # 1. AI Classification
        results = classifier(prompt)
        top_result = results[0]
        label = top_result['label'].upper()
        score = top_result['score']
        
        stats = {"neutral": 0, "equality": 0, "bias": 0}
        
        # Determine if biased based on score threshold
        final_label = "Neutral"
        is_toxic = False
        
        if label == 'TOXIC' and score > 0.75:
            is_toxic = True
            stats['bias'] = 1
            final_label = "Biased"
        else:
            stats['neutral'] = 1
            stats['equality'] = 1 

        classification_data = {
            "label": final_label,
            "score": score
        }
        
        # 2. Dictionary-based Highlighting & Rewriting
        highlights, rewrite_suggestion = analyze_text_issues(prompt)
        
        # If AI says toxic but regex found nothing, we can't offer a specific rewrite
        if is_toxic and not rewrite_suggestion:
             rewrite_suggestion = "This text was flagged as potentially toxic by AI, but no specific gendered terms were identified for replacement."

        # 3. Save to History
        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("INSERT INTO history (text, label, score) VALUES (?, ?, ?)", 
                      (prompt, final_label, score))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving to history: {e}")

        return jsonify({
            "stats": stats,
            "classification": classification_data,
            "highlights": highlights,
            "rewrite": rewrite_suggestion
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT text, label, score, timestamp FROM history ORDER BY id DESC LIMIT 5")
        rows = c.fetchall()
        conn.close()
        
        history_data = []
        for row in rows:
            history_data.append({
                "text": row["text"],
                "label": row["label"],
                "score": row["score"],
                "timestamp": row["timestamp"]
            })
            
        return jsonify(history_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)