import React from 'react';
import styles from './UploadInterface.module.scss';

interface ScriptDropdownProps {
  selectedScript: 'process.py' | 'toc.py' | 'entity.py' | 'bulk_summary.py';
  onScriptChange: (script: 'process.py' | 'toc.py' | 'entity.py' | 'bulk_summary.py') => void;
}

const ScriptDropdown: React.FC<ScriptDropdownProps> = ({ selectedScript, onScriptChange }) => {
  return (
    <div className={styles.scriptDropdown}>
      <select
        value={selectedScript}
        onChange={(e) => onScriptChange(e.target.value as 'process.py' | 'toc.py' | 'entity.py' | 'bulk_summary.py')}
        className={styles.scriptSelect}
      >
        <option value="process.py">Generate Summary</option>
        <option value="toc.py">Generate Timeline</option>
        <option value="entity.py">Extract Entities</option>
        <option value="bulk_summary.py">Bulk Summary</option>
      </select>
    </div>
  );
};

export default ScriptDropdown;