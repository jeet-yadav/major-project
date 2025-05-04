import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import pickle

# Download NLTK stopwords
print("Downloading NLTK stopwords...")
nltk.download('stopwords')

# Function to clean tweets
def tweet_to_words(tweet):
    letters_only = re.sub("[^a-zA-Z]", " ", tweet)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)

def train_model(csv_path='Tweets.csv'):
    print(f"Loading data from {csv_path}...")
    # Load data
    data = pd.read_csv(csv_path)
    
    # Drop neutral sentiment
    print("Preprocessing data...")
    data = data[data['airline_sentiment'] != 'neutral']
    
    # Clean tweets
    print("Cleaning tweets...")
    data['clean_tweet'] = data['text'].apply(lambda x: tweet_to_words(x))
    
    # Prepare features and target
    x = data.clean_tweet
    y = data.airline_sentiment
    
    # Vectorize
    print("Vectorizing text...")
    vectorizer = CountVectorizer()
    x_dtm = vectorizer.fit_transform(x)
    
    # Train model
    print("Training SVM model...")
    model = SVC(kernel='linear', random_state=10)
    model.fit(x_dtm, y)
    
    # Save model and vectorizer
    print("Saving model and vectorizer...")
    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
    
    print("Done! Model and vectorizer saved.")

if __name__ == "__main__":
    train_model()