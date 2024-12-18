"use client";

import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import About from "./pages/about";
import Demo from "./pages/demo";
import Home from "./pages/home";
import Navbar from "./components/Navbar"; // Import the Navbar component
import ToggleElement from "./pages/test";

const App: React.FC = () => {
  return (
    // <div className="App">
    //   <ToggleElement />
    // </div>
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
