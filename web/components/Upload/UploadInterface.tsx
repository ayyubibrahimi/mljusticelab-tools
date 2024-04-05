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
  const [selectedScript, setSelectedScript] = useState<'process.py' | 'toc.py'>('process.py');
  const [pdfPages, setPdfPages] = useState<string[]>([]);
  const [tocData, setTocData] = useState<{ sentence: string; page_number?: number }[]>([]);

  

  const handleFileUpload = async (file: File) => {
    setProcessingStatus('processing');
    setSentencePagePairs([]);
    setPdfPages([]);
    setTocData([]);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('script', selectedScript);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Response data:', data);

        if (selectedScript === 'process.py' && Array.isArray(data.sentencePagePairs)) {
          setSentencePagePairs(data.sentencePagePairs);
          setUploadedFilePath(data.filePath);
          console.log('Updated sentencePagePairs state:', data.sentencePagePairs);
        } if (selectedScript === 'toc.py' && data.tocData) {
          setTocData(data.tocData);
          setUploadedFilePath(data.filePath);
          
          // Generate the pdfPages array based on the total pages in tocData
          const totalPages = data.tocData.reduce((maxPage, item) => Math.max(maxPage, item.page_number || 0), 0);
          const pdfPagePaths = Array.from({ length: totalPages }, (_, index) => `/uploads/${file.filename}_page_${index + 1}.pdf`);
          setPdfPages(pdfPagePaths);
          
          console.log('Updated tocData state:', data.tocData);
        } else {
          console.error('Invalid response format. Expected an array of sentence-page pairs or TOC data.');
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
  }, [sentencePagePairs, pdfPages, tocData]);

  const uniquePageNumbers = [...new Set(sentencePagePairs.map(pair => pair.page_number ?? null).filter(Boolean))];

return (
    <div className={styles.container}>
      <div className={styles.contentContainer}>
        <Sidebar
          pages={uniquePageNumbers as number[]}
          onPageClick={handlePageClick}
          sentencePagePairs={sentencePagePairs}
          tocData={tocData}
        />
        <div className={styles.outputContainer}>
          <div className={styles.outputHeader}>
            <span>Innocence Lab</span>
            <div className={styles.scriptDropdown}>
              <select
                value={selectedScript}
                onChange={(e) => setSelectedScript(e.target.value as 'process.py' | 'toc.py')}
                className={styles.scriptSelect}
              >
                <option value="process.py">Generate Summary</option>
                <option value="toc.py">Generate Timeline</option>
              </select>
            </div>
          </div>
          <div className={styles.outputContent}>
            {processingStatus === 'processing' && (
              <div className={styles.statusMessage}>The PDF is being processed...</div>
            )}

            {processingStatus === 'completed' && selectedScript === 'process.py' && (
              <div ref={outputRef}>
                {sentencePagePairs.map((pair, index) => (
                  <Tippy key={index} content={`Page ${pair.page_number}`} followCursor={true} plugins={[followCursor]}>
                    <p onClick={() => setSelectedPage(pair.page_number)} className={styles.clickableSentence}>
                      {pair.sentence}
                    </p>
                  </Tippy>
                ))}
              </div>
            )}
            {processingStatus === 'completed' && selectedScript === 'toc.py' && (
              <div ref={outputRef}>
                <div className={styles.tocSection}>
                  {tocData.map((pageData, pageIndex) => (
                    <div key={pageIndex}>
                      {pageData.sentence.map((sentenceData, sentenceIndex) => (
                        <p key={sentenceIndex}>{sentenceData.text}</p>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          <div className={styles.uploadSection}>
            <FileUpload onFileUpload={handleFileUpload} disabled={processingStatus === 'processing'} />
          </div>
        </div>
      </div>
      {selectedPage && (
        <PopupBox pageNumber={selectedPage} onClose={() => setSelectedPage(null)} filePath={uploadedFilePath} />
      )}
    </div>
  );
};

export default UploadInterface;