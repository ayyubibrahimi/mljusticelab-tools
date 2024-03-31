import React, { useState, useRef, useEffect } from 'react';
import FileUpload from './FileUpload';
import styles from './UploadInterface.module.scss';
import Sidebar from './Sidebar';
import PopupBox from './PopupBox';

const UploadInterface: React.FC = () => {
  const [processingStatus, setProcessingStatus] = useState<'idle' | 'processing' | 'completed'>('idle');
  const [sentencePagePairs, setSentencePagePairs] = useState<{ sentence: string; page_number?: number }[]>([]);
  const [selectedPage, setSelectedPage] = useState<number | null>(null);
  const outputRef = useRef<HTMLDivElement>(null);
  const [uploadedFilePath, setUploadedFilePath] = useState<string | null>(null);

  const handleFileUpload = async (file: File) => {
    setProcessingStatus('processing');
    setSentencePagePairs([]);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Response data:', data);

        if (Array.isArray(data.sentencePagePairs)) {
          setSentencePagePairs(data.sentencePagePairs);
          setUploadedFilePath(data.filePath); // Extract the file path from the response
          console.log('Updated sentencePagePairs state:', data.sentencePagePairs);
        } else {
          console.error('Invalid response format. Expected an array of sentence-page pairs.');
        }

        setProcessingStatus('completed');
      } else {
        setProcessingStatus('idle');
        console.error('File processing failed');
      }
    } catch (error) {
      console.error('Error:', error);
      setProcessingStatus('idle');
    }
  };

  const handlePageClick = (pageNumber: number) => {
    setSelectedPage(pageNumber);
  };

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [sentencePagePairs]);

  const uniquePageNumbers = [...new Set(sentencePagePairs.map(pair => pair.page_number ?? null).filter(Boolean))];

  return (
    <div className={styles.container}>
      <div className={styles.contentContainer}>
        <div className={styles.sidebar}>
          <Sidebar pages={uniquePageNumbers as number[]} onPageClick={handlePageClick} />
        </div>
        <div className={styles.outputContainer}>
          {processingStatus === 'processing' && (
            <div className={styles.statusMessage}>The PDF is being processed...</div>
          )}
          {processingStatus === 'completed' && (
            <>
              <h3 className={styles.outputSubheading}>Processing Output:</h3>
              <div ref={outputRef}>
                {sentencePagePairs.map((pair, index) => (
                  <p key={index}>{pair.sentence}</p>
                ))}
              </div>
            </>
          )}
          <div className={styles.uploadSection}>
            <FileUpload onFileUpload={handleFileUpload} />
          </div>
        </div>
      </div>
      {selectedPage && (
        <PopupBox
          pageNumber={selectedPage}
          onClose={() => setSelectedPage(null)}
          filePath={uploadedFilePath}
        />
      )}
    </div>
  );
};

export default UploadInterface;