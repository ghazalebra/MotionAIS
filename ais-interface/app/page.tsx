"use client";

import React, { useEffect, useState } from "react";
import FileUpload from "./components/FileUpload";
import "./App.css"; // Import the CSS file
import VisSequence3D from "./components/VisSequence3D";
import data from "./sample_data";

const App: React.FC = () => {
  // console.log("hello");
  const handleSave = (files: File[]) => {
    console.log("Files to save:", files);
  };

  const [sequence, setSequence] = useState([]);

  const [sequenceReady, setSequenceReady] = useState(false);
  const [sequenceResults, setSequenceResults] = useState([]);

  // options
  const [showSegments, setShowSegments] = useState(false);
  const [showLandmarks, setShowLandmarks] = useState(false);
  const [showMotion, setShowMotion] = useState(false);

  useEffect(() => {
    fetchSequence();
  }, []);

  const fetchSequence = async () => {
    const response = await fetch("http://127.0.0.1:5000/sequence");
    const data = await response.json();
    setSequence(data.sequence);
    setSequenceResults(data.results);
    setSequenceReady(true);
    // console.log(data.sequence);
  };
  return (
    <div>
      <h1 className="header">Motion AIS</h1>
      <div className="upload">
        <FileUpload onSave={handleSave} />
      </div>
      <div className="task-bar">
        <input
          type="checkbox"
          id="show-segment-checkbox"
          checked={showSegments}
          onChange={(event) => {
            setShowSegments(event.target.checked);
          }}
        />
        <label htmlFor="show-segment-checkbox">Show Segments</label>

        <input
          type="checkbox"
          id="show-landmarks-checkbox"
          checked={showLandmarks}
          onChange={(event) => {
            setShowLandmarks(event.target.checked);
          }}
        />
        <label htmlFor="show-motion-checkbox">Show Landmarks</label>
        <input
          type="checkbox"
          id="show-motion-checkbox"
          checked={showMotion}
          onChange={(event) => {
            setShowMotion(event.target.checked);
          }}
        />
        <label htmlFor="show-motion-checkbox">Show Motion</label>
      </div>
      <div className="content">
        <div className="sequence-panel">
          {sequenceReady ? (
            <VisSequence3D sequence={sequence} results={sequenceResults} />
          ) : (
            <p>not ready!</p>
          )}
        </div>
        <div className="analysis-panel">the analysis</div>
      </div>
    </div>
  );
};

export default App;
