"use client";

import React, { useEffect, useState } from "react";
import FileUpload from "./components/SequenceUpload";
import "./App.css";
import VisSequence3D from "./components/VisSequence3D";
import axios from "axios";
// import { io } from "socket.io-client";

const App: React.FC = () => {
  //   const socket = io(
  //     "http://127.0.0.1:5000"
  // {
  //       transports: ["websocket"],
  //       cors: {
  //         origin: "http://localhost:3000/",
  //       },
  // }
  // );

  // console.log("hello");

  const [uploadSequence, setUploadSequence] = useState(false);
  const [sequenceList, setSequenceList] = useState<string[]>([]);
  const [sequenceName, setSequenceName] = useState("");

  const [sequence, setSequence] = useState([]);

  const [sequenceReady, setSequenceReady] = useState(false);
  const [sequenceResults, setSequenceResults] = useState([]);

  // options
  const [showSegments, setShowSegments] = useState(false);
  const [showLandmarks, setShowLandmarks] = useState(false);
  const [showMotion, setShowMotion] = useState(false);

  const [plots, setPlots] = useState("");

  // const [progress, setProgress] = useState(1);

  useEffect(() => {
    fetchSequenceList();
  }, []);

  // const startLoop = async () => {
  //   try {
  //     const response = await axios.post("http://127.0.0.1:5000/start_loop");
  //     console.log(response.data.message);
  //   } catch (error) {
  //     console.error("Error starting loop:", error);
  //   }
  // };

  // useEffect(() => {
  //   console.log("here");
  //   socket.on("progress", (data) => {
  //     setProgress(data.progress);
  //     console.log(data.progress);
  //   });

  //   socket.on("connect", () => {
  //     console.log("Connected to SocketIO server");
  //   });

  //   return () => {
  //     socket.off("progress");
  //     socket.off("connect");
  //   };
  // }, []);

  const fetchSequenceList = async () => {
    const response = await fetch(
      "http://127.0.0.1:5000/sequencelist"
      // "http://127.0.0.1:5000/sequencelist?path=data"
    );
    const data = await response.json();
    if (response.ok) {
      setSequenceList(data.sequencelist);
    } else {
      console.error("Failed to fetch the sequences:", data.error);
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
    const response = await axios.get("http://127.0.0.1:5000/sequence", {
      params: { sequenceName },
    });
    const data = response.data;
    setSequence(data.sequence);
    setSequenceResults(data.results);
    setSequenceReady(true);
  };

  const fetchplots = async (sequenceName?: string) => {
    const url = new URL("http://127.0.0.1:5000/analyze");
    if (sequenceName) {
      url.searchParams.append("sequenceName", sequenceName);
    }

    fetch(url)
      .then((response) => response.blob())
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        setPlots(url);
      })
      .catch((error) => {
        console.error("Error fetching the plots:", error);
      });
  };

  return (
    <div>
      <h1 className="header">Motion AIS</h1>
      <div className="specify-sequence">
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
        <button
          onClick={() => {
            setUploadSequence(!uploadSequence);
          }}
        >
          Upload Sequence
        </button>
        {uploadSequence && <FileUpload onSave={handleSave} />}
        <button
          onClick={() => {
            fetchSequence(sequenceName);
          }}
        >
          Visualize Sequence
        </button>

        {/* <div> */}
        {/* <h1>Progress Tracker</h1> */}
        {/* <button onClick={startLoop}>Start Loop</button> */}
        {/* <progress value={progress} max="100"></progress>
        <p>{progress}%</p> */}
        {/* </div> */}
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
          ) : sequenceName.length ? (
            <p>On it...</p>
          ) : (
            <p>Please select or upload a sequence!</p>
          )}
        </div>
        <div className="analysis-panel">
          <h1>Trunk Motion Analysis</h1>
          <button
            onClick={() => {
              fetchplots(sequenceName);
            }}
          >
            Show Motion Analysis
          </button>
          {plots && <img src={plots} alt="Motion Analysis" />}
        </div>
      </div>
    </div>
  );
};

export default App;
