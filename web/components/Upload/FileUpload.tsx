import React, { useState } from 'react';
import styles from './Fileupload.module.scss';

interface FileUploadProps {
  onFileUpload: (files: File[]) => Promise<any>;
  onSaveOutput: (content: any) => void;
  disabled?: boolean;
  multiple?: boolean;
  onClearScreen: () => void; // Add this prop
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload, onSaveOutput, disabled, multiple, onClearScreen }) => {
  const [files, setFiles] = useState<File[]>([]);
  const [lastUploadedData, setLastUploadedData] = useState<any>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = event.target.files;
    if (selectedFiles) {
      setFiles(Array.from(selectedFiles));
      onClearScreen(); // Call the onClearScreen function when new files are selected
    }
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (files.length === 0) {
      alert('Please select one or more PDF files to upload.');
      return;
    }
    setLastUploadedData(null);
    onClearScreen(); // Call the onClearScreen function before processing new files
    onFileUpload(files)
      .then(data => {
        setLastUploadedData(data);
      })
      .catch(error => {
        console.error('Upload failed:', error);
      });
  };

  return (
    <form onSubmit={handleSubmit} className={styles.uploadForm}>
      <div className={styles.uploadInputContainer}>
        <input
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          className={styles.fileInput}
          id="fileInput"
          multiple={multiple}
          disabled={disabled}
        />
        <label htmlFor="fileInput" className={styles.fileInputLabel}>
          Upload files
        </label>
        <button type="submit" disabled={disabled} className={styles.processButton}>
          {disabled ? 'Processing...' : 'Process'}
        </button>
        <button
          type="button"
          onClick={() => onSaveOutput(lastUploadedData)}
          disabled={disabled}
          className={styles.fileInputLabel}
        >
          Save Response
        </button>
      </div>
    </form>
  );
};

export default FileUpload;