from flask import Flask, request, render_template, redirect, url_for, session, flash
from generative_ai import generate_text  # Google Generative AI integration
from nltk_utils import tokenize, stem, bag_of_words  # NLTK helper functions
import json
import numpy as np
import random
import nltkfix

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key for session management

# This dictionary simulates a simple user database for demo purposes
users_db = {}

# Load intents from the JSON file
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Prepare vocabulary (all words) and classes (tags)
all_words = []
tags = []
patterns = []

# Initialize conversation history
history = []

for intent in intents['intents']:
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
        tokenized_words = tokenize(pattern)
        all_words.extend(tokenized_words)
        patterns.append((tokenized_words, intent['tag']))

# Stem and remove duplicates
all_words = sorted(set(stem(w) for w in all_words if w.isalpha()))

def classify_intent(user_input):
    """Classifies user input into an intent based on patterns in the intents.json."""
    tokenized_input = tokenize(user_input)
    bow = bag_of_words(tokenized_input, all_words)

    # Compare input against all known patterns
    similarities = np.array([np.dot(bow, bag_of_words(tokenized_pattern, all_words)) for tokenized_pattern, _ in patterns])
    
    best_match_idx = np.argmax(similarities)
    if similarities[best_match_idx] > 0:  # Ensure some words match
        matched_intent = patterns[best_match_idx][1]
        return matched_intent
    
    return None

def get_recent_context(history, limit=6):
    """Get recent conversation context."""
    recent_history = history[-limit:]  # Get the last 'limit' messages
    context = "\n".join(recent_history)  # Join them as a string for context
    return context

@app.route('/')
def index():
    """Render the index page and redirect to login by default."""
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login logic."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Check if the username and password match the saved credentials
        if username in users_db and users_db[username] == password:
            session['username'] = username  # Save username in session
            return redirect(url_for('chat_page'))  # Redirect to the chat page
        else:
            flash('Invalid credentials, please try again.')  # Flash error message
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Simple user registration logic (you might want to use a database)
        if username not in users_db:
            users_db[username] = password  # Save user in the simulated database
            return redirect(url_for('login'))  # Redirect to login after successful registration
        else:
            return "Username already exists, please choose a different one."
    
    return render_template('register.html')

@app.route('/chat')
def chat_page():
    """Render the chat page after successful login."""
    return render_template('chat.html')

@app.route('/get', methods=['POST'])
def chat():
    """Handle chat messages from the user."""
    user_input = request.form.get('msg', '').strip()

    # Store user input in history
    history.append(f"user: {user_input}")

    # Classify user input to determine intent
    intent_tag = classify_intent(user_input)

    # Use recent context for a more intelligent response
    recent_context = get_recent_context(history)

    if intent_tag:
        for intent in intents['intents']:
            if intent['tag'] == intent_tag:
                response = random.choice(intent['responses'])
                
                # Add the bot's response to history
                history.append(f"bot: {response}")
                return response
    
    # If no intent matches, use Google Generative AI with recent context
    response = generate_text(f"Context: {recent_context}\nUser: {user_input}")
    
    # Add bot's response to the history
    history.append(f"bot: {response}")
    return response

if __name__ == '__main__':
    app.run(debug=True)
