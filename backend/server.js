const express = require('express');
const cors = require('cors');
const Sentiment = require('sentiment');

const app = express();
const sentiment = new Sentiment();

app.use(cors());
app.use(express.json());

app.post('/api/sentiment', (req, res) => {
  const { tweet } = req.body;
  if (!tweet) {
    return res.status(400).json({ error: 'Tweet text is required.' });
  }

  const result = sentiment.analyze(tweet);
  res.json({ score: result.score, comparative: result.comparative });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
