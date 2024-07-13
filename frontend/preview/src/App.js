import React, { useState } from "react";
import "./App.css";

function App() {
  const [currentImage, setCurrentImage] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchImage = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch("http://localhost:5000/api/image");
      if (!response.ok) {
        throw new Error("Failed to fetch image");
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setCurrentImage(url);
    } catch (error) {
      setError(error.message);
      console.error("Error fetching image:", error);
    } finally {
      setLoading(false);
    }
  };

  const generateImage = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch("http://localhost:5000/api/generate", {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error("Failed to generate image");
      }
      const result = await response.json();
      console.log(result.message);
      fetchImage(); // Refresh the displayed image after generating
    } catch (error) {
      setError(error.message);
      console.error("Error generating image:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="image-container">
          <h3>Current Image</h3>
          <div className="image-placeholder">
            {loading ? (
              <p>Loading...</p>
            ) : currentImage ? (
              <img src={currentImage} alt="Current" />
            ) : (
              <p>No image available</p>
            )}
          </div>
          {error && <p className="error-message">{error}</p>}
        </div>
        <button onClick={generateImage} disabled={loading}>
          {loading ? "Generating..." : "Generate Image"}
        </button>
      </header>
    </div>
  );
}

export default App;
