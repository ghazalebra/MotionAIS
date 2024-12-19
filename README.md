# MotionAIS
### Overview
This repository contains the code for my master's research project. This project is the first step toward automatic and markerless dynamic analysis of 3D trunk motion for patients with AIS (Adolescent Idiopathic Scoliosis), an essential factor in scoliosis treatment planning. We have trained a model (PointNet++) to detect the 3D positions of the anatomical landmarks on a given point cloud of the back surface. By applying the landmark detection model on all the frames in a 3D motion sequence, we can find the trajectories of the landmarks. Using the detected landmarks, trunk motion metrics can be calculated for all the frames and monitored throughout a movement. We have collected a dataset of dynamic 3D acquisitions of three different movements performed by 26 AIS patients and annotated them with the actual positions of the landmarks. In addition, we have developed a pipeline for semi-automized annotation. Our experiments show a significant correlation between the motion analysis derived from the ground truth and the automatically detected landmarks.
### Demo
https://github.com/user-attachments/assets/13df05b3-1ec0-466b-9b96-0e7b6a18a488

