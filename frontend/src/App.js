import React, { useState, useRef } from "react";
import axios from "axios";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [report, setReport] = useState(null);
  const fileInputRef = useRef();

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setReport(null);
    }
  };

  const handleRemoveImage = () => {
    setImage(null);
    setPreview(null);
    setPrediction(null);
    setReport(null);
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
      setPrediction(response.data);
    } catch (error) {
      console.error("Error uploading image:", error);
      alert("Failed to get prediction.");
    }
  };

  const handleGenerateReport = async () => {
    if (!prediction || !prediction.labels) {
      alert("Run prediction first.");
      return;
    }

    try {
      const response = await axios.post("http://127.0.0.1:5000/generate_report", {
        age: 67,
        gender: "Male",
        view: "PA",
        findings: prediction.labels
      });
      setReport(response.data.report);
    } catch (error) {
      console.error("Error generating report:", error);
      alert("Failed to generate report.");
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h2> Upload Chest X-Ray</h2>
      <input type="file" accept="image/*" onChange={handleImageChange} ref={fileInputRef} />
      {preview && (
        <div style={{ marginTop: "20px" }}>
          <img src={preview} alt="preview" width="300px" />
          <br />
          <button
            onClick={handleRemoveImage}
            style={{ marginTop: "10px", padding: "6px 12px", background: "#ff4d4f", color: "white", border: "none", borderRadius: "5px" }}
          >
             Remove Image
          </button>
        </div>
      )}
      <br />
      <button
        onClick={handleSubmit}
        style={{ marginTop: "20px", padding: "10px 20px", fontSize: "16px", borderRadius: "5px" }}
      >
        Submit
      </button>
      {prediction && (
  <div style={{ marginTop: "20px" }}>
    <h3> Prediction Result</h3>
    <p><strong>Prediction:</strong> {prediction.prediction}</p>
    
    <div style={{ display: "inline-block", textAlign: "left" }}>
      <ul style={{ listStyleType: "disc", paddingLeft: "20px" }}>
        {prediction.labels.map((label, index) => (
          <li key={index}>{label}: {prediction.probabilities[index]}</li>
        ))}
      </ul>

      <div style={{ textAlign: "center", marginTop: "10px" }}>
        <button
          onClick={handleGenerateReport}
          style={{
            padding: "8px 16px",
            background: "#1890ff",
            color: "white",
            border: "none",
            borderRadius: "5px"
          }}
        >
          📄 Generate Report
        </button>
      </div>
    </div>
  </div>
)}

      {report && (
        <div style={{ marginTop: "30px", padding: "20px", border: "1px solid #ccc", maxWidth: "600px", marginInline: "auto" }}>
          <h3> Radiology Report</h3>
          <pre style={{ textAlign: "left", whiteSpace: "pre-wrap" }}>{report}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
