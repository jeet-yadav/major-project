const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { load } = require('joblib-node'); // Load Python model

const app = express();
app.use(cors());
app.use(bodyParser.json());

const model = load('./sentiment_model.pkl'); // Load trained model

app.post('/predict', async (req, res) => {
    const text = req.body.text;
    const prediction = await model.predict([text]); // Predict sentiment
    res.json({ sentiment: prediction[0] });
});

app.listen(5000, () => console.log('Server running on port 5000'));
