// src/Page.tsx

import React from "react";
import { Link } from "react-router-dom";
import "../styles/Home.css";

const Home: React.FC = () => {
  return (
    <div className="page-container">
      <div className="left-column">
        {/* <h1 className="title">Motion AIS</h1> */}
        {/* <img src="spine.png" alt="Spine" className="spine-image" /> */}
        <p className="description">
          An AI-powered toolbox for tracking and analyzing the 3D trunk motion
          in patients with scoliosis.
        </p>
        <Link to="/demo" className="demo-button">
          Try the Demo
        </Link>{" "}
        {/* Button for demo */}
      </div>
      <div className="right-column">
        {/* You can add content for the right half here, if needed */}
      </div>
    </div>
  );
};

export default Home;
