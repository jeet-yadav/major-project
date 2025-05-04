
from flask import Flask, request, jsonify
from flask_cors import CORS
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
CORS(app)

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

model = None
vectorizer = None

def load_existing_model():
    global model, vectorizer
    try:
        if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
            model = pickle.load(open('model.pkl', 'rb'))
            vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
            print("Model and vectorizer loaded successfully!")
            return True
        else:
            print("Model or vectorizer not found.")
            return False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# Try to load model at startup, or train if files don't exist
if not load_existing_model():
    print("Attempting to train model from default dataset...")
    try:
        # Import the training function
        from train_model import train_model
        # Try to train using the default Tweets.csv
        if os.path.exists('Tweets.csv'):
            train_model('Tweets.csv')
            load_existing_model()
        else:
            print("Tweets.csv not found in the current directory.")
    except Exception as e:
        print(f"Error training model: {str(e)}")

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
    global model, vectorizer
    
    if model is None or vectorizer is None:
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
    try:
        # Load the dataset if it exists
        if os.path.exists('Tweets.csv'):
            data = pd.read_csv('Tweets.csv')
            airlines_list = data['airline'].unique().tolist()
            return jsonify({"airlines": airlines_list})
        else:
            # Fallback to hardcoded list if dataset not available
            airlines_list = ['US Airways', 'United', 'American', 'Southwest', 'Delta', 'Virgin America']
            return jsonify({"airlines": airlines_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    global model, vectorizer
    
    try:
        # Load the dataset if it exists
        if os.path.exists('Tweets.csv'):
            data = pd.read_csv('Tweets.csv')
            # Filter out neutral sentiment as we're only handling binary classification
            data = data[data['airline_sentiment'] != 'neutral']
            
            stats_data = {
                "model_type": "Support Vector Machine (SVM)",
                "dataset_size": len(data),
                "airlines_count": len(data['airline'].unique()),
                "positive_tweets": len(data[data['airline_sentiment'] == 'positive']),
                "negative_tweets": len(data[data['airline_sentiment'] == 'negative']),
                "accuracy": 0.83  # This is placeholder - in a real implementation you'd use cross-validation
            }
            
            # Add model info if model is loaded
            if model is not None:
                stats_data["model_loaded"] = True
            else:
                stats_data["model_loaded"] = False
                
            return jsonify(stats_data)
        else:
            # Return basic info if dataset not available
            return jsonify({
                "model_type": "Support Vector Machine (SVM)",
                "accuracy": 0.83,  # Placeholder accuracy
                "dataset_size": 14640,  # Placeholder dataset size
                "model_loaded": model is not None
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model_endpoint():
    """Endpoint to train and save the model from local CSV data"""
    global model, vectorizer
    
    try:
        # Check if file was uploaded
        if 'file' in request.files:
            file = request.files['file']
            # Save the uploaded file temporarily
            temp_file_path = 'temp_upload.csv'
            file.save(temp_file_path)
            file_path = temp_file_path
        else:
            # Use the local Tweets.csv file
            file_path = 'Tweets.csv'
            if not os.path.exists(file_path):
                return jsonify({"error": f"No file uploaded and {file_path} not found"}), 400
        
        # Load data
        data = pd.read_csv(file_path)
        
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
        
        # Clean up temporary file if it exists
        if 'file' in request.files and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return jsonify({"success": "Model trained and saved successfully!"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)