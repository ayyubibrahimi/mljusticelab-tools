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
  const [expandedFiles, setExpandedFiles] = useState({});
  const [expandedSavedFiles, setExpandedSavedFiles] = useState({});
  const [renderedOutput, setRenderedOutput] = useState(null);

  const toggleExpanded = (filename) => {
    setExpandedFiles((prevState) => ({
      ...prevState,
      [filename]: !prevState[filename],
    }));
  };

  const toggleExpandedSaved = (filename) => {
    setExpandedSavedFiles((prevState) => ({
      ...prevState,
      [filename]: !prevState[filename],
    }));
  };



  const handleAnalysisClick = (analysis) => {
    setSelectedAnalysis(analysis);
    setSelectedScript(analysis);
  };

  const handleFileUpload = async (files) => {

    handleClearScreen();
    
    setProcessingStatus('processing');
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    formData.append('script', selectedScript);
    formData.append('model', selectedModel);
  
    try {
      const response = await fetch('/api/upload', { method: 'POST', body: formData });
      if (response.ok) {
        const data = await response.json();
        console.log('Raw data received from the backend:', data);
  
        if (selectedScript === 'process.py') {
          if (Array.isArray(data.results)) {
            setDisplayedContent({
              sentencePagePairs: data.results,
            });
          } else {
            console.error('Invalid data format for sentence-page pairs');
            setProcessingStatus('idle');
            return;
          }
  
          // Group sentence-page pairs by filename
          const groupedSentencePagePairs = parsedSentencePagePairs.reduce((acc, pair) => {
            if (!acc[pair.filename]) {
              acc[pair.filename] = [];
            }
            acc[pair.filename].push(pair);
            return acc;
          }, {});
  
          setDisplayedContent({
            filePaths: files.map(file => file.name),
            groupedSentencePagePairs,
          });
        } else if (selectedScript === 'toc.py') {
          setTocData(data.tocData);
        } else if (selectedScript === 'entity.py') {
          setCsvFilePath(data.csvFilePath);
        } else if (selectedScript === 'bulk_summary.py') {
          setBulkSummary(data.summaries);
        }
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

  const uniquePageNumbers = sentencePagePairs && Array.isArray(sentencePagePairs)
  ? [
      ...new Set(
        sentencePagePairs
          .flatMap(pair => [
            pair.page_number,
            pair.page_number_candidate_2,
            pair.page_number_candidate_3,
          ])
          .filter(Boolean)
      ),
    ]
  : [];

  const handleClearScreen = () => {
    setDisplayedContent(null);
    setDisplayedSavedResponse(null);
    setSelectedAnalysis(null);
    setProcessingStatus('idle');
    setRenderedOutput(null); // Clear the renderedOutput state
  };

  useEffect(() => {
    if (displayedContent && selectedScript === 'process.py') {
      const output = (
        <div className={styles.displayedContentArea}>
          {Object.entries(
            displayedContent.sentencePagePairs.reduce((acc, pair) => {
              if (!acc[pair.filename]) {
                acc[pair.filename] = [];
              }
              acc[pair.filename].push(pair);
              return acc;
            }, {})
          ).map(([filename, fileSentences]) => (
            <div key={filename}>
              <button
                className={styles.collapsibleButton}
                onClick={() => {
                  toggleExpanded(filename);
                  const savedContent = JSON.parse(localStorage.getItem('displayedContent'));
                  if (savedContent && savedContent.filename === filename) {
                    setDisplayedContent(savedContent);
                  }
                }}
              >
                {filename}
              </button>
              {expandedFiles[filename] && (
                <div className={styles.collapsibleContent}>
                  {fileSentences.map((pair, index) => (
                    <Tippy
                      key={index}
                      content={
                        <div className={styles.tippyContent}>
                          <div className={styles.tippyTitle}>Associated Pages</div>
                          <p>Page {pair.page_number} (Score: {pair.page_number_score.toFixed(2)})</p>
                          {pair.page_number_candidate_2 !== null && (
                            <p>Page {pair.page_number_candidate_2} (Score: {pair.page_number_candidate_2_score.toFixed(2)})</p>
                          )}
                          {pair.page_number_candidate_3 !== null && (
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
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      );
      setRenderedOutput(output);
      localStorage.setItem('displayedContent', JSON.stringify(displayedContent));
    }
  }, [displayedContent, selectedScript, expandedFiles]);

  useEffect(() => {
    const saved = localStorage.getItem('savedResponses');
    if (process.env.NODE_ENV === 'development') {
      localStorage.clear();
    } else if (saved) {
      setSavedResponses(JSON.parse(saved));
    }
  }, []);

  const saveResponseToLocalStorage = (content) => {
    const newResponseId = savedResponses.length + 1;
    const newResponse = {
      id: newResponseId,
      label: `Saved Response ${newResponseId}`,
      script: selectedScript,
      content: displayedContent, // Save the displayedContent instead of renderedOutput
    };
    const updatedResponses = [...savedResponses, newResponse];
    setSavedResponses(updatedResponses);
    localStorage.setItem('savedResponses', JSON.stringify(updatedResponses));
  };

  const handleDisplaySavedResponse = (response) => {
    setDisplayedSavedResponse(response);
    setDisplayedContent(response.content); // Set the displayedContent state with the saved content
  };


  const handleRenameSavedResponse = (responseId, newLabel) => {
    const updatedResponses = savedResponses.map((response) => {
      if (response.id === responseId) {
        return { ...response, label: newLabel };
      }
      return response;
    });
    setSavedResponses(updatedResponses);
    localStorage.setItem('savedResponses', JSON.stringify(updatedResponses));
  };
  
  const handleDeleteSavedResponse = (responseId) => {
    const updatedResponses = savedResponses.filter((response) => response.id !== responseId);
    setSavedResponses(updatedResponses);
    localStorage.setItem('savedResponses', JSON.stringify(updatedResponses));
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
          onDeleteSavedResponse={handleDeleteSavedResponse}
          onRenameSavedResponse={handleRenameSavedResponse}
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
            
            {renderedOutput}
    
            {displayedSavedResponse && displayedSavedResponse.renderedOutput && (
              <div
                className={styles.displayedSavedResponseArea}
                dangerouslySetInnerHTML={{ __html: displayedSavedResponse.renderedOutput }}
              />
            )}
          </div>
          <div className={styles.uploadSection}>
            <FileUpload
              onFileUpload={handleFileUpload}
              onSaveOutput={saveResponseToLocalStorage}
              disabled={processingStatus === 'processing'}
              multiple
              onClearScreen={handleClearScreen}
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
