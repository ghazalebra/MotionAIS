"use client";

import React, { useState, ChangeEvent } from "react";
import axios from "axios";
import "../styles/Demo.css";

interface FileUploadProps {
  onSave?: (files: File[], sequence_name: string) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onSave }) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [sequenceName, setSequenceName] = useState<string>("");
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      const filesArray = Array.from(event.target.files);
      setSelectedFiles((prevFiles) => [...prevFiles, ...filesArray]);
    }
  };

  const handleSequenceNameChange = (event: ChangeEvent<HTMLInputElement>) => {
    // console.log("here");
    setSequenceName(event.target.value);
  };

  const handleSave = async () => {
    const formData = new FormData();
    formData.append("sequence_name", sequenceName);
    selectedFiles.forEach((file) => {
      formData.append("files", file);
    });

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/upload",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          onUploadProgress: function (progressEvent) {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setUploadProgress(percentCompleted);
          },
        }
      );
      console.log("Files successfully uploaded:", response.data);
      if (onSave) onSave(selectedFiles, sequenceName);
      alert("Files have been saved successfully");
      setUploadError(null); // Clear any previous errors
    } catch (error) {
      console.error("Error uploading files:", error);
      setUploadError("Error uploading files");
    }
  };

  return (
    <div className="upload-sequence">
      <div className="inputs-row">
        <input
          type="text"
          value={sequenceName}
          onChange={(e) => handleSequenceSelect(e)}
          placeholder="Enter sequence name"
        />
        <input
          type="file"
          multiple
          onChange={(e) => handleSave(e.target.files, sequenceName)}
        />
        <button onClick={handleSave}>Upload</button>
      </div>
      <progress value={0} max="100"></progress>
      {uploadError && <p style={{ color: "red" }}>{uploadError}</p>}
    </div>
  );
};

export default FileUpload;
