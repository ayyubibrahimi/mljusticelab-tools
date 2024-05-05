import React from 'react';
import styles from './UploadInterface.module.scss';

interface ModelDropdownProps {
  selectedModel: string;
  onModelChange: (model: string) => void;
}

const ModelDropdown: React.FC<ModelDropdownProps> = ({ selectedModel, onModelChange }) => {
  return (
    <div className={styles.modelDropdown}>
      <select
        value={selectedModel}
        onChange={(e) => onModelChange(e.target.value)}
        className={styles.modelSelect}
      >
        <option value="">Select Model</option>
        <option value="gpt-4-0125-preview">GPT-4</option>
        <option value="gpt-3.5-0125">GPT-3.5</option>
        <option value="claude-3-haiku-20240307">Claude (Haiku)</option>
        <option value="claude-3-sonnet-20240229">Claude (Sonnet)</option>
        <option value="claude-3-opus-20240229">Claude (Opus)</option>
      </select>
    </div>
  );
};

export default ModelDropdown;