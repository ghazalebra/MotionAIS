// src/About.tsx

import React from "react";
import "../styles/About.css";

const About: React.FC = () => {
  return (
    <div className="about-container">
      {/* <h1 className="about-title">About Motion AIS</h1> */}
      <p className="about-description">
        Adolescent idiopathic scoliosis (AIS) is a 3D deformity of the spine and
        the ribcage, that affects the general appearance of the trunk. If left
        untreated, AIS can progress during growth spurt and reduce patients’
        quality of life. Existing methods for assessing scoliosis, whether
        through radiographs or surface topography are all based on static
        acquisitions. However, the spine is an articulated structure allowing
        for the mobility of the trunk. The evaluation of the trunk under
        kinematic conditions provides insights on the rigidity of scoliosis. The
        self-correction exercise and the lateral bending test are two examples
        of motion used by physiotherapists and orthopedic surgeons respectively
        to evaluate the flexibility of the trunk. This information, although
        mostly qualitative, is considered for treatment planning. This project
        is the first to explore dynamic surface topography for the quantitative
        evaluation of trunk mobility. Using pre-trained 3D deep learning models,
        we develop a toolbox for an automatic quantification of trunk motion
        during sequences of lateral bending and self-correction exercises. Our
        proposed toolbox provides functionalities such as back surface
        anatomical landmark detection, back surface anatomical segmentation,
        quantifying trunk surface parameters over time, and visualizing
        landmarks’ trajectories during a motion sequence. Despite some
        limitations, our experiments show promising results regarding the
        potential clinical application of the proposed toolbox. Overall, this
        work is the first step toward automating dynamic quantitative evaluation
        of the trunk motion and provides a basis for future work in this
        direction.
      </p>

      <a
        href="thesis.pdf"
        target="_blank"
        rel="noopener noreferrer"
        className="pdf-link"
      >
        Download our PDF overview
      </a>
    </div>
  );
};

export default About;
