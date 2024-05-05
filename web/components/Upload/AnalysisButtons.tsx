import React from 'react';
import classNames from 'classnames';
import styles from './UploadInterface.module.scss';

interface AnalysisButtonsProps {
  selectedAnalysis: 'process.py' | 'toc.py' | 'entity.py' | 'bulk_summary.py' | null;
  onAnalysisClick: (analysis: 'process.py' | 'toc.py' | 'entity.py' | 'bulk_summary.py') => void;
}

const AnalysisButtons: React.FC<AnalysisButtonsProps> = ({ selectedAnalysis, onAnalysisClick }) => {
  return (
    <div className={styles.analysisButtonsContainer}>
      <button
        className={classNames(styles.analysisButton, selectedAnalysis === 'process.py' && styles.selected)}
        onClick={() => onAnalysisClick('process.py')}
      >
        Generate Summary
      </button>
      <button
        className={classNames(styles.analysisButton, selectedAnalysis === 'toc.py' && styles.selected)}
        onClick={() => onAnalysisClick('toc.py')}
      >
        Generate Timeline
      </button>
      <button
        className={classNames(styles.analysisButton, selectedAnalysis === 'entity.py' && styles.selected)}
        onClick={() => onAnalysisClick('entity.py')}
      >
        Extract Entities
      </button>
      <button
        className={classNames(styles.analysisButton, selectedAnalysis === 'bulk_summary.py' && styles.selected)}
        onClick={() => onAnalysisClick('bulk_summary.py')}
      >
        Bulk Summary
      </button>
    </div>
  );
};

export default AnalysisButtons;