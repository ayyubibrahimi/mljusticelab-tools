import React, { useState } from 'react';
import styles from './UploadInterface.module.scss';

interface FileUploadProps {
  onFileUpload: (files: File[]) => void;
  disabled?: boolean;
  multiple?: boolean;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload, disabled, multiple }) => {
  const [files, setFiles] = useState<File[]>([]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = event.target.files;
    if (selectedFiles) {
      setFiles(Array.from(selectedFiles));
    }
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (files.length === 0) {
      alert('Please select one or more PDF files to upload.');
      return;
    }
    onFileUpload(files);
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
        />
        <label htmlFor="fileInput" className={styles.fileInputLabel}>
          Upload files
        </label>
        <button type="submit" disabled={disabled} className={styles.processButton}>
          {disabled ? 'Processing...' : 'Process'}
        </button>
      </div>
    </form>
  );
};

export default FileUpload;