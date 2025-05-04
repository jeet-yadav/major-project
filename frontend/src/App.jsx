import React, { useState } from "react";
import axios from "axios";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState("");

  const analyzeSentiment = async () => {
    try {
      const response = await axios.post("http://localhost:5000/api/sentiment", {
        tweet: text,
      });
      setResult(`Score: ${response.data.score}, Comparative: ${response.data.comparative}`);
    } catch (error) {
      console.error("Error analyzing sentiment:", error);
      setResult("Error occurred");
    }
  };

  return (
    <div className="h-screen bg-black text-white flex justify-center items-center flex-col">
      <h1 className="text-5xl font-bold mb-4">Sentiment Analysis</h1>
      <input
        className="bg-slate-700 text-white p-2 rounded-full w-1/2"
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <button
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold px-4 py-2 m-4 rounded-full cursor-pointer"
        onClick={analyzeSentiment}
      >
        Analyze
      </button>
      <div className="bg-slate-700 p-4 rounded-full">
        <p>Sentiment: {result}</p>
      </div>
    </div>
  );
}

export default App;
