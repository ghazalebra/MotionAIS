"use client";

import React, { useEffect, useState } from "react";
import FileUpload from "./components/SequenceUpload";
import "./App.css";
import VisSequence3D from "./components/VisSequence3D";
import axios from "axios";
import Collapsible from "react-collapsible";

const App: React.FC = () => {
  const [uploadSequence, setUploadSequence] = useState(false);
  const [sequenceList, setSequenceList] = useState<string[]>([]);
  const [sequenceName, setSequenceName] = useState("");
  const [sequence, setSequence] = useState<any[]>([]);
  const [sequenceReady, setSequenceReady] = useState(false);
  const [sequenceResults, setSequenceResults] = useState<any[]>([]);
  const [showSegments, setShowSegments] = useState(false);
  const [showLandmarks, setShowLandmarks] = useState(false);
  const [showMotion, setShowMotion] = useState(false);
  const [plots, setPlots] = useState("");

  useEffect(() => {
    fetchSequenceList();
  }, []);

  const fetchSequenceList = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/sequencelist");
      const data = await response.json();
      if (response.ok) {
        setSequenceList(data.sequencelist);
      } else {
        console.error("Failed to fetch the sequences:", data.error);
      }
    } catch (error) {
      console.error("Error fetching the sequence list:", error);
    }
  };

  const handleSequenceSelect = (
    event: React.ChangeEvent<HTMLSelectElement>
  ) => {
    setSequenceName(event.target.value);
  };

  const handleSave = (files: File[], sequenceName: string) => {
    console.log("Files and sequence name:", files, sequenceName);
    setSequenceName(sequenceName);
  };

  const fetchSequence = async (sequenceName?: string) => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/sequence", {
        params: { sequenceName },
      });
      const data = response.data;
      setSequence(data.sequence);
      setSequenceReady(true);
    } catch (error) {
      console.error("Error fetching the sequence:", error);
    }
  };

  const fetchLandmarks = async (sequenceName?: string) => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/landmarks", {
        params: { sequenceName },
      });
      const data = response.data;
      setSequenceResults(data.results);
    } catch (error) {
      console.error("Error fetching the landmarks:", error);
    }
  };

  const fetchPlots = async (sequenceName?: string) => {
    const url = new URL("http://127.0.0.1:5000/analyze");
    if (sequenceName) {
      url.searchParams.append("sequenceName", sequenceName);
    }

    try {
      const response = await fetch(url.toString());
      const blob = await response.blob();
      const plotUrl = URL.createObjectURL(blob);
      console.log(plotUrl);
      setPlots(plotUrl);
    } catch (error) {
      console.error("Error fetching the plots:", error);
    }
  };

  return (
    <div>
      <h1 className="header">Motion AIS</h1>
      <div className="choose-sequence">
        <div>
          <h3>Select or Upload a Sequence:</h3>
          <select onChange={handleSequenceSelect} defaultValue="">
            <option value="" disabled>
              Select a sequence
            </option>
            {sequenceList.map((seq, index) => (
              <option key={index} value={seq}>
                {seq}
              </option>
            ))}
          </select>
        </div>
        <Collapsible
          trigger={
            <div
              className="collapsible-trigger"
              //  key={someStableKey}
            >
              Upload Sequence
              <span className={`arrow ${uploadSequence ? "open" : ""}`}>
                &#9654;
              </span>
            </div>
          }
          onOpening={() => setUploadSequence(true)}
          onClosing={() => setUploadSequence(false)}
        >
          {uploadSequence && <FileUpload onSave={handleSave} />}
        </Collapsible>
        <button onClick={() => fetchSequence(sequenceName)}>
          Visualize Sequence
        </button>
      </div>
      {/* <div className="task-bar">
        <input
          type="checkbox"
          id="show-segment-checkbox"
          checked={showSegments}
          onChange={(event) => setShowSegments(event.target.checked)}
        />
        <label htmlFor="show-segment-checkbox">Show Segments</label>
        <input
          type="checkbox"
          id="show-landmarks-checkbox"
          checked={showLandmarks}
          onChange={(event) => setShowLandmarks(event.target.checked)}
        />
        <label htmlFor="show-landmarks-checkbox">Show Landmarks</label>
        <input
          type="checkbox"
          id="show-motion-checkbox"
          checked={showMotion}
          onChange={(event) => setShowMotion(event.target.checked)}
        />
        <label htmlFor="show-motion-checkbox">Show Motion</label>
      </div> */}
      <div className="tasks">
        <button onClick={() => fetchLandmarks(sequenceName)}>
          Track Landmarks
        </button>
        <button onClick={() => fetchPlots(sequenceName)}>
          Show Motion Analysis
        </button>
      </div>
      <div className="content">
        <div className="sequence-panel">
          {sequenceReady ? (
            <VisSequence3D sequence={sequence} results={sequenceResults} />
          ) : sequenceName.length ? (
            <p>On it...</p>
          ) : (
            <p>Please select or upload a sequence!</p>
          )}
        </div>
        <div className="analysis-panel">
          <h1>Trunk Motion Analysis</h1>
          {plots && <img src={plots} alt="Motion Analysis" />}
        </div>
      </div>
    </div>
  );
};

export default App;
