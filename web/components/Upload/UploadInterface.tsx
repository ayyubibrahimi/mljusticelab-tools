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
  const [sentencePagePairs, setSentencePagePairs] = useState<{
    sentence: string;
    page_number?: number;
    page_number_score?: number;
    page_number_candidate_2?: number;
    page_number_candidate_2_score?: number;
    page_number_candidate_3?: number;
    page_number_candidate_3_score?: number;
  }[]>([]);
  const [selectedPage, setSelectedPage] = useState<number | null>(null);
  const outputRef = useRef<HTMLDivElement>(null);
  const [uploadedFilePath, setUploadedFilePath] = useState<string | null>(null);
  const [selectedScript, setSelectedScript] = useState<'process.py' | 'toc.py' | 'entity.py'>('process.py');
  const [pdfPages, setPdfPages] = useState<string[]>([]);
  const [tocData, setTocData] = useState<{ sentence: string; page_number?: number }[]>([]);
  const [csvFilePath, setCsvFilePath] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('');


  const handleFileUpload = async (file: File) => {
    setProcessingStatus('processing');
    setSentencePagePairs([]);
    setPdfPages([]);
    setTocData([]);
    setCsvFilePath(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('script', selectedScript);
    formData.append('model', selectedModel); // Add the selected model to the form data


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
        } else if (selectedScript === 'toc.py' && data.tocData) {
          setTocData(data.tocData);
          setUploadedFilePath(data.filePath);

          // Generate the pdfPages array based on the total pages in tocData
          const totalPages = data.tocData.reduce((maxPage, item) => Math.max(maxPage, item.page_number || 0), 0);
          const pdfPagePaths = Array.from({ length: totalPages }, (_, index) => `/uploads/${file.name}_page_${index + 1}.pdf`);
          setPdfPages(pdfPagePaths);

          console.log('Updated tocData state:', data.tocData);
        } else if (selectedScript === 'entity.py' && data.csvFilePath) {
          setCsvFilePath(data.csvFilePath);
          console.log('CSV file path:', data.csvFilePath);
        } else {
          console.error('Invalid response format.');
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

  const uniquePageNumbers = [
    ...new Set(
      sentencePagePairs
        .flatMap(pair => [
          pair.page_number,
          pair.page_number_candidate_2,
          pair.page_number_candidate_3,
        ])
        .filter(Boolean)
    ),
  ];

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
          <div className={styles.dropdownContainer}>
            <div className={styles.scriptDropdown}>
              <select
                value={selectedScript}
                onChange={(e) => setSelectedScript(e.target.value as 'process.py' | 'toc.py' | 'entity.py')}
                className={styles.scriptSelect}
              >
                <option value="process.py">Generate Summary</option>
                <option value="toc.py">Generate Timeline</option>
                <option value="entity.py">Extract Entities</option>
              </select>
            </div>
            <div className={styles.modelDropdown}>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
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
          </div>
          <span>Innocence Lab</span>
        </div>

          <div className={styles.outputContent}>
            {processingStatus === 'processing' && (
              <div className={styles.statusMessage}>The PDF is being processed...</div>
            )}

            {processingStatus === 'completed' && selectedScript === 'process.py' && (
              <div ref={outputRef}>
                {sentencePagePairs.map((pair, index) => (
                  <Tippy
                    key={index}
                    content={
                      <>
                        <div className={styles.tippyTitle}>Associated Pages</div>
                        <div className={styles.tippyContent}>
                          <p>Page {pair.page_number} (Probability Score: {pair.page_number_score})</p>
                          {pair.page_number_candidate_2 && (
                            <p>Page {pair.page_number_candidate_2} (Probability Score: {pair.page_number_candidate_2_score})</p>
                          )}
                          {pair.page_number_candidate_3 && (
                            <p>Page {pair.page_number_candidate_3} (Probability Score: {pair.page_number_candidate_3_score})</p>
                          )}
                        </div>
                      </>
                    }
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

            {processingStatus === 'completed' && selectedScript === 'entity.py' && (
              <div>
                <button onClick={() => window.open(csvFilePath, '_blank')}>Download CSV</button>
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