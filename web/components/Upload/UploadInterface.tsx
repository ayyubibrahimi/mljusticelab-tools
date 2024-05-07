import React, { useState, useRef, useEffect } from 'react';
import FileUpload from './FileUpload';
import styles from './UploadInterface.module.scss';
import Sidebar from './Sidebar';
import PopupBox from './PopupBox';
import Tippy from '@tippyjs/react';
import 'tippy.js/dist/tippy.css';
import { followCursor } from 'tippy.js';
import ScriptDropdown from './ScriptDropdown';
import ModelDropdown from './ModelDropdown';
import AnalysisButtons from './AnalysisButtons';

const UploadInterface: React.FC = () => {
  const [processingStatus, setProcessingStatus] = useState<'idle' | 'processing' | 'completed'>('idle');
  const [sentencePagePairs, setSentencePagePairs] = useState([]);
  const [selectedPage, setSelectedPage] = useState<number | null>(null);
  const outputRef = useRef<HTMLDivElement>(null);
  const [uploadedFilePath, setUploadedFilePath] = useState<string | null>(null);
  const [pdfPages, setPdfPages] = useState<string[]>([]);
  const [tocData, setTocData] = useState([]);
  const [csvFilePath, setCsvFilePath] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [selectedScript, setSelectedScript] = useState('process.py');
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const [bulkSummary, setBulkSummary] = useState([]);
  const [savedResponses, setSavedResponses] = useState([]);
  const [displayedContent, setDisplayedContent] = useState(null);
  const [displayedSavedResponse, setDisplayedSavedResponse] = useState(null);

  useEffect(() => {
    const saved = localStorage.getItem('savedResponses');
    if (process.env.NODE_ENV === 'development') {
      localStorage.clear();
    } else if (saved) {
      setSavedResponses(JSON.parse(saved));
    }
  }, []);

  const handleDisplaySavedResponse = (content) => {
    setDisplayedSavedResponse(content);
    setDisplayedContent(null); // Clear the displayed content when showing a saved response
  };

  const saveResponseToLocalStorage = (content) => {
    const newResponseId = savedResponses.length + 1;
    const newResponse = {
      id: newResponseId,
      label: `Saved Response ${newResponseId}`,
      content: {
        filePath: uploadedFilePath,
        sentencePagePairs: sentencePagePairs,
      },
    };
    const updatedResponses = [...savedResponses, newResponse];
    setSavedResponses(updatedResponses);
    localStorage.setItem('savedResponses', JSON.stringify(updatedResponses));
  };

  const handleAnalysisClick = (analysis) => {
    setSelectedAnalysis(analysis);
    setSelectedScript(analysis);
  };

  const handleFileUpload = async (files) => {
    setProcessingStatus('processing');
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    formData.append('script', selectedScript);
    formData.append('model', selectedModel);
  
    try {
      const response = await fetch('/api/upload', { method: 'POST', body: formData });
      if (response.ok) {
        const data = await response.json();
        if (selectedScript === 'process.py') {
          setSentencePagePairs(data.sentencePagePairs);
          setDisplayedContent({
            filePath: data.filePath,
            sentencePagePairs: data.sentencePagePairs,
          });
        } else if (selectedScript === 'toc.py') {
          setTocData(data.tocData);
        } else if (selectedScript === 'entity.py') {
          setCsvFilePath(data.csvFilePath);
        } else if (selectedScript === 'bulk_summary.py') {
          setBulkSummary(data.summaries);
        }
        setUploadedFilePath(data.filePath);
        setDisplayedSavedResponse(null); // Clear the displayed saved response
        setProcessingStatus('completed');
      } else {
        console.error('File processing failed');
        setProcessingStatus('idle');
      }
    } catch (error) {
      console.error('Error:', error);
      setProcessingStatus('idle');
    }
  };

  const handlePageClick = (pageNumber) => {
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

  const handleClearScreen = () => {
    setDisplayedContent(null);
    setDisplayedSavedResponse(null);
    setSelectedAnalysis(null);
    setProcessingStatus('idle');
  };

  return (
    <div className={styles.container}>
      <div className={styles.contentContainer}>
        <Sidebar
          pages={uniquePageNumbers}
          onPageClick={handlePageClick}
          sentencePagePairs={sentencePagePairs}
          tocData={tocData}
          onSavedResponseClick={handleDisplaySavedResponse}
          savedResponses={savedResponses}
        />
        <div className={styles.outputContainer}>
          <div className={styles.outputHeader}>
            <div className={styles.dropdownContainer}>
              <ScriptDropdown selectedScript={selectedScript} onScriptChange={setSelectedScript} />
              <ModelDropdown selectedModel={selectedModel} onModelChange={setSelectedModel} />
            </div>
            <span>Innocence Lab</span>
          </div>
          <div className={styles.outputContent}>
            {processingStatus !== 'completed' && !displayedContent && !displayedSavedResponse && (
              <AnalysisButtons selectedAnalysis={selectedAnalysis} onAnalysisClick={handleAnalysisClick} />
            )}
            
            {displayedContent ? (
              <div className={styles.displayedContentArea}>
                <p>File Path: {displayedContent.filePath}</p>
                <div ref={outputRef} className={styles.processOutput}>
                  {displayedContent.sentencePagePairs.map((pair, index) => (
                    <React.Fragment key={index}>
                      <Tippy
                        content={
                          <div className={styles.tippyContent}>
                            <div className={styles.tippyTitle}>Associated Pages</div>
                            <p>Page {pair.page_number} (Score: {pair.page_number_score.toFixed(2)})</p>
                            {pair.page_number_candidate_2 && (
                              <p>Page {pair.page_number_candidate_2} (Score: {pair.page_number_candidate_2_score.toFixed(2)})</p>
                            )}
                            {pair.page_number_candidate_3 && (
                              <p>Page {pair.page_number_candidate_3} (Score: {pair.page_number_candidate_3_score.toFixed(2)})</p>
                            )}
                          </div>
                        }
                        theme="custom"
                        followCursor={true}
                        plugins={[followCursor]}
                        className={styles.tippyTooltip}
                      >
                        <span
                          onClick={() => setSelectedPage(pair.page_number)}
                          className={styles.clickableSentence}
                        >
                          {pair.sentence}
                        </span>
                      </Tippy>
                      {(index + 1) % 4 === 0 && <br />}
                      {(index + 1) % 4 === 0 && <br />}
                    </React.Fragment>
                  ))}
                </div>
              </div>
            ) : displayedSavedResponse ? (
              <div className={styles.displayedSavedResponseArea}>
                <p>File Path: {displayedSavedResponse.filePath}</p>
                <div ref={outputRef} className={styles.processOutput}>
                  {displayedSavedResponse.sentencePagePairs.map((pair, index) => (
                    <React.Fragment key={index}>
                      <Tippy
                        content={
                          <div className={styles.tippyContent}>
                            <div className={styles.tippyTitle}>Associated Pages</div>
                            <p>Page {pair.page_number} (Score: {pair.page_number_score.toFixed(2)})</p>
                            {pair.page_number_candidate_2 && (
                              <p>Page {pair.page_number_candidate_2} (Score: {pair.page_number_candidate_2_score.toFixed(2)})</p>
                            )}
                            {pair.page_number_candidate_3 && (
                              <p>Page {pair.page_number_candidate_3} (Score: {pair.page_number_candidate_3_score.toFixed(2)})</p>
                            )}
                          </div>
                        }
                        theme="custom"
                        followCursor={true}
                        plugins={[followCursor]}
                        className={styles.tippyTooltip}
                      >
                        <span
                          onClick={() => setSelectedPage(pair.page_number)}
                          className={styles.clickableSentence}
                        >
                          {pair.sentence}
                        </span>
                      </Tippy>
                      {(index + 1) % 4 === 0 && <br />}
                      {(index + 1) % 4 === 0 && <br />}
                    </React.Fragment>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
          <div className={styles.uploadSection}>
            <FileUpload
              onFileUpload={handleFileUpload}
              onSaveOutput={saveResponseToLocalStorage}
              disabled={processingStatus === 'processing'}
              multiple
              onClearScreen={handleClearScreen} // Pass the handleClearScreen function
            />
          </div>
        </div>
      </div>
      {selectedPage && (
        <PopupBox pageNumber={selectedPage} onClose={() => setSelectedPage(null)} filePath={uploadedFilePath} />
      )}
    </div>
  );
}
export default UploadInterface;
