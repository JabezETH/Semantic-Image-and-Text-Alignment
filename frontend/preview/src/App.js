import React, { useState } from "react";
import "./App.css";

function App() {
  const [currentImage, setCurrentImage] = useState("");

  const fetchImage = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/image");
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setCurrentImage(url);
    } catch (error) {
      console.error("Error fetching image:", error);
    }
  };

  const generateImage = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/generate", {
        method: "POST",
      });
      const result = await response.json();
      console.log(result.message);
      fetchImage(); // Refresh the displayed image after generating
    } catch (error) {
      console.error("Error generating image:", error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="image-container">
          <h3>Current Image</h3>
          <div className="image-placeholder">
            {currentImage && <img src={currentImage} alt="Current" />}
          </div>
        </div>
        <button onClick={generateImage}>Generate Image</button>
      </header>
    </div>
  );
}

export default App;
