import React, { useState, useRef, useEffect } from 'react';
import FileUpload from './FileUpload';
import styles from './UploadInterface.module.scss';
import Sidebar from './Sidebar';
import PopupBox from './PopupBox';
import Tippy from '@tippyjs/react';
import 'tippy.js/dist/tippy.css';
import { followCursor } from 'tippy.js';

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
          setUploadedFilePath(data.filePath);
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
        <Sidebar pages={uniquePageNumbers as number[]} onPageClick={handlePageClick} sentencePagePairs={sentencePagePairs} />
        <div className={styles.outputContainer}>
          <div className={styles.outputHeader}>Innocence Lab</div>
          <div className={styles.outputContent}>
            {processingStatus === 'processing' && (
              <div className={styles.statusMessage}>The PDF is being processed...</div>
            )}
            {processingStatus === 'completed' && (
              <div ref={outputRef}>
                {sentencePagePairs.map((pair, index) => (
                  <Tippy
                    key={index}
                    content={`Page ${pair.page_number}`}
                    followCursor={true}
                    plugins={[followCursor]}
                  >
                    <p
                      onClick={() => setSelectedPage(pair.page_number)}
                      className={styles.clickableSentence}
                    >
                      {pair.sentence}
                    </p>
                  </Tippy>
                ))}
              </div>
            )}
          </div>
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