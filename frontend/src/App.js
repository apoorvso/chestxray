import React, { useState, useRef } from "react";
import axios from "axios";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const fileInputRef = useRef(); // To reset <input type="file" />

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
    }
  };

  const handleRemoveImage = () => {
    setImage(null);
    setPreview(null);
    setPrediction(null);

    // Clear the actual file input
    if (fileInputRef.current) {
      fileInputRef.current.value = null;
    }
  };

  const handleSubmit = async () => {
    if (!image) {
      alert("Please upload an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await axios.post("http://127.0.0.1:5000/classify", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      const result = response.data;
      setPrediction(result);

    } catch (error) {
      console.error("Error uploading image:", error);
      alert("Failed to get prediction.");
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h2>🖼️ Upload an Image</h2>
  
      <input
        type="file"
        accept="image/*"
        onChange={handleImageChange}
        ref={fileInputRef}
      />
  
      {preview && (
        <div style={{ marginTop: "20px" }}>
          <img src={preview} alt="preview" width="300px" />
          <br />
          <button
            onClick={handleRemoveImage}
            style={{
              marginTop: "10px",
              padding: "6px 12px",
              background: "#ff4d4f",
              color: "white",
              border: "none",
              borderRadius: "5px"
            }}
          >
            ❌ Remove Image
          </button>
        </div>
      )}
  
      <br />
      <button
        onClick={handleSubmit}
        style={{
          marginTop: "20px",
          padding: "10px 20px",
          fontSize: "16px",
          borderRadius: "5px"
        }}
      >
        Submit
      </button>
  
      {prediction && (
        <div style={{ marginTop: "20px" }}>
          <h3>🧠 Prediction Result</h3>
          <p>
            <strong>Prediction:</strong> {prediction.prediction}
          </p>
          <ul style={{ listStyleType: "disc", textAlign: "left", display: "inline-block" }}>
            {prediction.labels.map((label, index) => (
              <li key={index}>
                {label}: {prediction.probabilities[index]}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
  
}

export default App;
