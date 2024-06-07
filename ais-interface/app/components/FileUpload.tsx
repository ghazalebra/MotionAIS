"use client";

import React, { useState, ChangeEvent } from "react";
import axios from "axios";

interface FileUploadProps {
  onSave?: (files: File[]) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onSave }) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      const filesArray = Array.from(event.target.files);
      setSelectedFiles((prevFiles) => [...prevFiles, ...filesArray]);
    }
  };

  const handleSave = async () => {
    const formData = new FormData();
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
      if (onSave) onSave(selectedFiles);
      alert("Files have been saved successfully");
      setUploadError(null); // Clear any previous errors
    } catch (error) {
      console.error("Error uploading files:", error);
      setUploadError("Error uploading files");
    }
  };

  return (
    <div>
      <input type="file" multiple onChange={handleFileChange} />
      <button onClick={handleSave}>Upload</button>
      <progress value={uploadProgress} max="100"></progress>
      {/* <div>
        <h3>Selected Files</h3>
        <ul>
          {selectedFiles.map((file, index) => (
            <li key={index}>{file.name}</li>
          ))}
        </ul>
      </div> */}

      {uploadError && <p style={{ color: "red" }}>{uploadError}</p>}
    </div>
  );
};

export default FileUpload;

// import React, { useState } from "react";
// import axios from "axios";

// function FileUpload() {
//   const [files, setFiles] = useState([]);
//   const [uploadedFiles, setUploadedFiles] = useState([]);
//   const [uploadProgress, setUploadProgress] = useState(0);

//   function handleMultipleChange(event) {
//     console.log("handle Multiple change");
//     setFiles([...event.target.files]);
//   }

//   function handleMultipleSubmit(event) {
//     console.log("uploading");
//     event.preventDefault();
//     const url = "http://localhost:5000/upload";
//     const formData = new FormData();
//     files.forEach((file, index) => {
//       formData.append(`file${index}`, file);
//     });

//     const config = {
//       headers: {
//         "content-type": "multipart/form-data",
//       },
//       onUploadProgress: function (progressEvent) {
//         const percentCompleted = Math.round(
//           (progressEvent.loaded * 100) / progressEvent.total
//         );
//         setUploadProgress(percentCompleted);
//       },
//     };

//     axios
//       .post(url, formData, config)
//       .then((response) => {
//         console.log(response.data);
//         setUploadedFiles(response.data.files);
//       })
//       .catch((error) => {
//         console.error("Error uploading files: ", error);
//       });
//   }

//   return (
//     <div className="App">
//       <form onSubmit={handleMultipleSubmit}>
//         <h1>Upload the sequence</h1>
//         <input type="file" multiple onChange={handleMultipleChange} />
//         <button type="submit">Upload</button>
//         {/* <progress value={uploadProgress} max="100"></progress> */}
//       </form>
//       {uploadedFiles.map((file, index) => (
//         <img key={index} src={file} alt={`Uploaded content ${index}`} />
//       ))}
//     </div>
//   );
// }

// export default FileUpload;
