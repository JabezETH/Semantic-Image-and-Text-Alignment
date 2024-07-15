import React, { useState } from "react";
import "./App.css"; // Import CSS file for styling (create this file with your styles)

const App = () => {
  const [largerImageFile, setLargerImageFile] = useState(null);
  const [smallImageFiles, setSmallImageFiles] = useState([]);
  const [comparisonImageFile, setComparisonImageFile] = useState(null);
  const [message, setMessage] = useState("");
  const [generatedImage, setGeneratedImage] = useState(null);

  const handleLargerImageChange = (event) => {
    const file = event.target.files[0];
    setLargerImageFile(file);
  };

  const handleSmallImageChange = (event, index) => {
    const file = event.target.files[0];
    const newSmallImageFiles = [...smallImageFiles];
    newSmallImageFiles[index] = file;
    setSmallImageFiles(newSmallImageFiles);
  };

  const handleComparisonImageChange = (event) => {
    const file = event.target.files[0];
    setComparisonImageFile(file);
  };

  const handleUploadAll = async () => {
    const formData = new FormData();
    formData.append("large_image", largerImageFile);

    smallImageFiles.forEach((file, index) => {
      if (file) {
        formData.append(`small_image${index + 1}`, file); // Ensure each small image file is appended correctly
      }
    });

    try {
      const response = await fetch("http://localhost:5000/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Error uploading images");
      }

      const data = await response.json();
      setMessage(data.message);
    } catch (error) {
      console.error("Error:", error);
      setMessage("Failed to upload images");
    }
  };

  const handleGenerateImage = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/generate", {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error("Error generating image");
      }

      const data = await response.json();
      setMessage(data.message);

      // Fetch the generated image after generating it
      await fetchGeneratedImage();
    } catch (error) {
      console.error("Error:", error);
      setMessage("Failed to generate image");
    }
  };

  const handleUploadComparisonImage = async () => {
    const formData = new FormData();
    formData.append("comparison_image", comparisonImageFile);

    try {
      const response = await fetch(
        "http://localhost:5000/api/upload-comparison",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("Error uploading comparison image");
      }

      const data = await response.json();
      setMessage(data.message);
    } catch (error) {
      console.error("Error:", error);
      setMessage("Failed to upload comparison image");
    }
  };

  const fetchGeneratedImage = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/image");
      if (!response.ok) {
        throw new Error("Error fetching image");
      }

      const blob = await response.blob();
      setGeneratedImage(URL.createObjectURL(blob));
    } catch (error) {
      console.error("Error:", error);
      setMessage("Failed to fetch generated image");
    }
  };

  const handleDeleteImages = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/delete-images", {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error("Error deleting images");
      }

      const data = await response.json();
      setMessage(data.message);

      // Reset state to allow new uploads
      setLargerImageFile(null);
      setSmallImageFiles([]);
      setComparisonImageFile(null);
      setGeneratedImage(null);

      // Optionally, reload the page
      window.location.reload();
    } catch (error) {
      console.error("Error:", error);
      setMessage("Failed to delete images");
    }
  };

  return (
    <div className="app-container">
      <h1>Update Image Paths</h1>
      <div className="upload-section">
        <label>
          Upload Larger Image:
          <input
            type="file"
            accept="image/*"
            onChange={handleLargerImageChange}
          />
        </label>
      </div>
      <div className="upload-section">
        <h2>Upload Small Images:</h2>
        {smallImageFiles.map((file, index) => (
          <div key={index} className="small-image-upload">
            <label>
              Upload Small Image {index + 1}:
              <input
                type="file"
                accept="image/*"
                onChange={(e) => handleSmallImageChange(e, index)}
              />
            </label>
          </div>
        ))}
        <button
          className="add-small-image-btn"
          onClick={() => setSmallImageFiles([...smallImageFiles, null])}
        >
          Add Another Small Image
        </button>
      </div>
      <div className="upload-section">
        <h2>Upload Comparison Image:</h2>
        <label>
          Upload Comparison Image:
          <input
            type="file"
            accept="image/*"
            onChange={handleComparisonImageChange}
          />
        </label>
        <button
          className="upload-comparison-btn"
          onClick={handleUploadComparisonImage}
          disabled={!comparisonImageFile}
        >
          Upload Comparison Image
        </button>
      </div>
      <div className="image-preview-section">
        <h2>Image Previews:</h2>
        {largerImageFile && (
          <div className="image-preview">
            <img
              src={URL.createObjectURL(largerImageFile)}
              alt="Preview of larger image"
            />
          </div>
        )}
        {smallImageFiles.map(
          (file, index) =>
            file && (
              <div key={index} className="image-preview">
                <img
                  src={URL.createObjectURL(file)}
                  alt={`Preview of small image ${index + 1}`}
                />
              </div>
            )
        )}
        {generatedImage && (
          <div className="image-preview">
            <h3>Generated Image:</h3>
            <img src={generatedImage} alt="Generated Image" />
          </div>
        )}
      </div>
      <div className="button-group">
        <button
          className="upload-all-btn"
          onClick={handleUploadAll}
          disabled={!largerImageFile || smallImageFiles.length === 0}
        >
          Upload All Images
        </button>
        <button
          className="generate-image-btn"
          onClick={handleGenerateImage}
          disabled={!largerImageFile || smallImageFiles.some((file) => !file)}
        >
          Generate Image
        </button>
        <button className="delete-images-btn" onClick={handleDeleteImages}>
          Delete Images and Reload
        </button>
      </div>
      {message && (
        <p className={`message ${message.includes("Failed") ? "error" : ""}`}>
          {message}
        </p>
      )}
    </div>
  );
};

export default App;
