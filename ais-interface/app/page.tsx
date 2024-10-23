"use client";

import React from "react";
import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";
import About from "./pages/about";
import Demo from "./pages/demo";
import Home from "./pages/home";
import Navbar from "./components/NavBar"; // Import the Navbar component

const App: React.FC = () => {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/demo" element={<Demo />} />
      </Routes>
    </Router>
  );
};

export default App;
