import React, { useState } from "react";
import axios from "axios";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
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
        headers: {
          "Content-Type": "multipart/form-data"
        }
      });

      const result = response.data;
      console.log(result)
      // Build a message string from predictions
      const message = Object.entries(result)
        .map(([label, prob]) => `${label}: ${prob}%`)
        .join("\n");

      alert("✅ Prediction Result:\n\n" + message);

    } catch (error) {
      console.error("Error uploading image:", error);
      alert("Failed to get prediction.");
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h2>🖼️ Upload an Image</h2>
      <input type="file" accept="image/*" onChange={handleImageChange} />

      {preview && (
        <div style={{ marginTop: "20px" }}>
          <img src={preview} alt="preview" width="300px" />
        </div>
      )}

      <br />
      <button onClick={handleSubmit} style={{ marginTop: "20px", padding: "10px 20px" }}>
        Submit
      </button>
    </div>
  );
}

export default App;
