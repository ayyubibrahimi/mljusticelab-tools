import React, { useState } from 'react';
import styles from './UploadInterface.module.scss';

interface FileUploadProps {
  onFileUpload: (file: File) => void;
  disabled?: boolean;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload, disabled }) => {
  const [file, setFile] = useState<File | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setFile(event.target.files?.[0] || null);
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file) {
      alert('Please select a PDF file to upload.');
      return;
    }
    onFileUpload(file);
  };

  return (
    <form onSubmit={handleSubmit} className={styles.uploadForm}>
      <div className={styles.uploadInputContainer}>
        <input type="file" accept=".pdf" onChange={handleFileChange} className={styles.fileInput} id="fileInput" />
        <label htmlFor="fileInput" className={styles.fileInputLabel}>Choose file</label>
        <button type="submit" disabled={disabled} className={styles.processButton}>
          {disabled ? 'Processing...' : 'Process'}
        </button>
      </div>
    </form>
  );
};

export default FileUpload;