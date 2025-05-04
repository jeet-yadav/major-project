from flask import Flask, request, jsonify
# from flask_cors import CORS
import joblib

app = Flask(__name__)
# CORS(app)

# Load model and vectorizer
model = joblib.load("model/svm.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    tweet = data.get("tweet", "")

    if not tweet:
        return jsonify({"error": "Tweet is required"}), 400

    # Transform and predict
    features = vectorizer.transform([tweet])
    prediction = model.predict(features)[0]  # Should return a string like "positive"

    label = prediction.capitalize()

    return jsonify({"sentiment": label})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
