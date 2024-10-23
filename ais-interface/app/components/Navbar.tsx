// src/components/Navbar.tsx

import React from "react";
import { Link } from "react-router-dom";
import "../styles/Navbar.css"; // Import CSS for styling

const Navbar: React.FC = () => {
  return (
    <nav className="navbar">
      <Link to="/" className="navbar-title">
        Motion AIS
      </Link>
      <ul className="navbar-list">
        <li className="navbar-item">
          <Link to="/about" className="navbar-link">
            About
          </Link>
        </li>
        <li className="navbar-item">
          <Link to="/demo" className="navbar-link">
            Demo
          </Link>
        </li>
      </ul>
    </nav>
  );
};

export default Navbar;
