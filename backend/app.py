from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import pickle
import os

app = Flask(__name__)

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords', quiet=True)

# Function to clean tweets
def tweet_to_words(tweet):
    letters_only = re.sub("[^a-zA-Z]", " ", tweet)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)

# Load model and vectorizer
def load_model():
    # Check if model files exist, otherwise return None
    if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
        model = pickle.load(open('model.pkl', 'rb'))
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        return model, vectorizer
    return None, None

@app.route('/')
def home():
    return """
    <h1>Airline Sentiment Analysis API</h1>
    <p>A simple API for predicting sentiment of airline tweets.</p>
    <h2>Available Endpoints:</h2>
    <ul>
        <li><b>POST /predict</b> - Predict sentiment for a tweet</li>
        <li><b>GET /airlines</b> - Get list of airlines in the dataset</li>
        <li><b>GET /stats</b> - Get statistics about the model</li>
    </ul>
    """

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is loaded
    model, vectorizer = load_model()
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500
    
    # Get tweet from request
    data = request.get_json(force=True)
    
    if 'tweet' not in data:
        return jsonify({"error": "No tweet provided"}), 400
    
    tweet = data['tweet']
    
    # Clean tweet
    clean_tweet = tweet_to_words(tweet)
    
    # Convert to document-term matrix
    tweet_dtm = vectorizer.transform([clean_tweet])
    
    # Predict sentiment
    prediction = model.predict(tweet_dtm)[0]
    
    # Return prediction
    return jsonify({
        "tweet": tweet,
        "sentiment": prediction,
        "sentiment_label": "positive" if prediction == 1 else "negative"
    })

@app.route('/airlines', methods=['GET'])
def airlines():
    # In a real implementation, this would query your dataset
    airlines_list = ['US Airways', 'United', 'American', 'Southwest', 'Delta', 'Virgin America']
    return jsonify({"airlines": airlines_list})

@app.route('/stats', methods=['GET'])
def stats():
    # In a real implementation, this would be based on your actual model metrics
    return jsonify({
        "model_type": "Support Vector Machine (SVM)",
        "accuracy": 0.83,  # This should be your actual model accuracy
        "dataset_size": 14640  # This should be your actual dataset size
    })

@app.route('/train', methods=['POST'])
def train_model():
    """Endpoint to train and save the model from CSV data"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        # Load data
        data = pd.read_csv(file)
        
        # Basic preprocessing (similar to what's in the notebook)
        # Drop neutral sentiment
        data = data[data['airline_sentiment'] != 'neutral']
        
        # Clean tweets
        data['clean_tweet'] = data['text'].apply(lambda x: tweet_to_words(x))
        
        # Prepare features and target
        x = data.clean_tweet
        y = data.airline_sentiment
        
        # Vectorize
        vectorizer = CountVectorizer()
        x_dtm = vectorizer.fit_transform(x)
        
        # Train model
        model = SVC(kernel='linear', random_state=10)
        model.fit(x_dtm, y)
        
        # Save model and vectorizer
        pickle.dump(model, open('model.pkl', 'wb'))
        pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
        
        return jsonify({"success": "Model trained and saved successfully!"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)