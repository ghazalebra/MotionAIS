"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import Stats from "three/examples/jsm/libs/stats.module";
import "../App.css";

function VisSequence3D({ sequence, results }) {
  const [frame, setFrame] = useState(1);
  const canvasRef = useRef(null);
  const sceneRef = useRef(null);
  const pointsRef = useRef(null);
  const landmarksRef = useRef(null);

  useEffect(() => {
    console.log("Initializing scene");

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050505);
    scene.fog = new THREE.Fog(0x050505, 2000, 3500);

    const camera = new THREE.PerspectiveCamera(
      50,
      canvasRef.current.clientWidth / canvasRef.current.clientHeight,
      1,
      5000
    );
    camera.position.z = 2000;

    const renderer = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
      antialias: true,
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(
      canvasRef.current.clientWidth,
      canvasRef.current.clientHeight
    );

    const controls = new OrbitControls(camera, renderer.domElement);

    const stats = Stats();
    document.body.appendChild(stats.dom);

    sceneRef.current = scene;

    const animate = () => {
      stats.update();
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    };
    animate();

    const handleResize = () => {
      camera.aspect =
        canvasRef.current.clientWidth / canvasRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(
        canvasRef.current.clientWidth,
        canvasRef.current.clientHeight
      );
    };

    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      document.body.removeChild(stats.dom);
    };
  }, []);

  useEffect(() => {
    if (!sceneRef.current) return;

    const scene = sceneRef.current;

    if (pointsRef.current) {
      scene.remove(pointsRef.current);
    }
    if (landmarksRef.current) {
      scene.remove(landmarksRef.current);
    }

    const body = sequence[frame - 1]["points"];
    const bodyColors = sequence[frame - 1]["colors"];
    const landmarks = results
      .slice(0, frame)
      .reduce((acc, result) => acc.concat(result.centers.slice(0, -1)), []);

    // console.log(landmarks);
    const bodyVertices = [];
    const bodyColorsArray = [];
    const landmarkVertices = [];

    // Add body points
    body.forEach((coord, index) => {
      bodyVertices.push(coord[1], coord[0], coord[2]);
      bodyColorsArray.push(
        bodyColors[index][0],
        bodyColors[index][1],
        bodyColors[index][2]
      );
    });

    // Add landmarks
    landmarks.forEach((coord) => {
      landmarkVertices.push(coord[1], coord[0], coord[2] + 5);
    });

    // Body geometry and material
    const bodyGeometry = new THREE.BufferGeometry();
    bodyGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(bodyVertices, 3)
    );
    bodyGeometry.setAttribute(
      "color",
      new THREE.Float32BufferAttribute(bodyColorsArray, 3)
    );

    const bodyMaterial = new THREE.PointsMaterial({
      size: 15,
      vertexColors: true,
    });

    const bodyPoints = new THREE.Points(bodyGeometry, bodyMaterial);
    scene.add(bodyPoints);
    pointsRef.current = bodyPoints;

    // Landmark geometry and material
    const landmarkGeometry = new THREE.BufferGeometry();
    landmarkGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(landmarkVertices, 3)
    );

    const landmarkMaterial = new THREE.PointsMaterial({
      size: 15,
      color: 0x00ff00, // Red color for landmarks
    });

    const landmarkPoints = new THREE.Points(landmarkGeometry, landmarkMaterial);
    scene.add(landmarkPoints);
    landmarksRef.current = landmarkPoints;
  }, [frame, sequence, results]);

  return (
    <div className="visualization-container">
      <canvas id="showSequence3D" ref={canvasRef} className="showsequence3D" />
      <div className="slider-container">
        <input
          className="slider"
          type="range"
          min={1}
          max={sequence.length}
          onChange={(e) => setFrame(parseInt(e.target.value))}
          value={frame}
          id="slider"
        />
        <span className="frame-indicator">{frame}</span>
      </div>
    </div>
  );
}

export default VisSequence3D;
