import React, { useState, useEffect, useRef } from 'react';
import FileUpload from './FileUpload';
import styles from './ChatInterface.module.scss';

interface ChatInterfaceProps {
  onFileUpload: (file: File) => void;
  processingStatus: 'idle' | 'processing' | 'completed';
  processingOutput: string;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  onFileUpload,
  processingStatus,
  processingOutput,
}) => {
  const outputRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [processingOutput]);

  return (
    <div className={styles.chatContainer}>
      <FileUpload onFileUpload={onFileUpload} />

      {processingStatus !== 'idle' && (
        <div className={styles.outputContainer}>
          <h2 className={styles.outputHeading}>Processing Output:</h2>
          <div ref={outputRef}>{processingOutput}</div>
        </div>
      )}
    </div>
  );
};

export default ChatInterface;