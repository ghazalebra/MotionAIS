"use client";

import React, { useEffect, useState } from "react";
import FileUpload from "../components/SequenceUpload";
import "../styles/Demo.css";
import VisSequence3D from "../components/VisSequence3D";
import axios from "axios";
import Collapsible from "react-collapsible";

const Demo: React.FC = () => {
  const [uploadSequence, setUploadSequence] = useState(false);
  const [showSequenceSelect, setShowSequenceSelect] = useState(true);
  const [sequenceList, setSequenceList] = useState<string[]>([]);
  const [sequenceName, setSequenceName] = useState("");
  const [sequence, setSequence] = useState<any[]>([]);
  const [sequenceReady, setSequenceReady] = useState(false);
  const [isFetching, setIsFetching] = useState(false);
  const [sequenceResults, setSequenceResults] = useState<any[]>([]);
  const [plots, setPlots] = useState({});
  const [LoadingPlots, setLoadingPlots] = useState(false);

  useEffect(() => {
    fetchSequenceList();
    setShowSequenceSelect(true);
  }, []);

  const toggleShowSequenceSelect = () => {
    setShowSequenceSelect((prev) => !prev);
  };

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
    // setSequenceReady(false);
  };

  const handleSave = (files: File[], sequenceName: string) => {
    console.log("Files and sequence name:", files, sequenceName);
    setSequenceName(sequenceName);
  };

  const fetchSequence = async (sequenceName?: string) => {
    setSequenceReady(false);
    setIsFetching(true);
    setSequenceResults([]);
    setPlots({});
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
    setIsFetching(false);
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
    // console.log("hey");
    setPlots({});
    setLoadingPlots(true);
    // const url = new URL("http://127.0.0.1:5000/analyze");
    // if (sequenceName) {
    //   url.searchParams.append("sequenceName", sequenceName);
    // }
    const response = await axios.get("http://127.0.0.1:5000/analyze", {
      params: { sequenceName },
    });
    const data = response.data;

    try {
      const response = await axios.get("http://127.0.0.1:5000/analyze", {
        params: { sequenceName },
      });
      const plotUrls = response.data;
      // console.log("Plot ready");
      // console.log(plotUrls);
      // console.log(Object.keys(plotUrls).length);
      setPlots(plotUrls);
    } catch (error) {
      console.error("Error fetching the plots:", error);
    }
    setLoadingPlots(false);
  };

  return (
    <div>
      {/* <h1 className="header">Motion AIS</h1> */}
      <div className="choose-sequence">
        {showSequenceSelect && (
          <div>
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
            <Collapsible
              trigger={
                <div className="collapsible-trigger">
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
          </div>
        )}

        <div className="tasks">
          <button onClick={() => fetchSequence(sequenceName)}>
            Visualize Sequence
          </button>
          <button onClick={() => fetchLandmarks(sequenceName)}>
            Track Landmarks
          </button>
          <button onClick={() => fetchPlots(sequenceName)}>
            Show Motion Analysis
          </button>
        </div>
      </div>

      <div className="content">
        <div className="sequence-panel">
          {sequenceReady ? (
            <VisSequence3D sequence={sequence} results={sequenceResults} />
          ) : sequenceName.length && isFetching ? (
            <p className="loading-message">Loading the sequence...</p>
          ) : (
            <p className="placeholder">Please select or upload a sequence!</p>
          )}
        </div>

        <div className="analysis-panel">
          {plots && Object.keys(plots).length ? (
            <div className="plot-grid">
              {Object.entries(plots).map(([metricName, url]) => (
                <div className="plot-item" key={metricName}>
                  {/* <h3>{metricName}</h3> */}
                  <img
                    src={"http://127.0.0.1:5000/" + url}
                    alt={`Plot for ${metricName}`}
                  />
                </div>
              ))}
            </div>
          ) : LoadingPlots ? (
            <div className="placeholder">
              <p>Loading the graphs...</p>
            </div>
          ) : (
            <div className="placeholder">
              <p>Analysis graphs will appear here once available.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Demo;
