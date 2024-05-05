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
  const [pdfPages, setPdfPages] = useState<string[]>([]);

  interface TocItem {
    sentence: { text: string }[];
    page_number: number;
  }
  
  const [tocData, setTocData] = useState<TocItem[]>([]);
  
  const [csvFilePath, setCsvFilePath] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [selectedScript, setSelectedScript] = useState<'process.py' | 'toc.py' | 'entity.py' | 'bulk_summary.py'>('process.py');
  const [selectedAnalysis, setSelectedAnalysis] = useState<'process.py' | 'toc.py' | 'entity.py' | 'bulk_summary.py' | null>(null);

  const handleAnalysisClick = (analysis: 'process.py' | 'toc.py' | 'entity.py' | 'bulk_summary.py') => {
    setSelectedAnalysis(analysis);
    setSelectedScript(analysis);
  };

  interface BulkSummaryItem {
    filename: string;
    summary: string;
    total_pages: number;
  }
  
  const [bulkSummary, setBulkSummary] = useState<BulkSummaryItem[]>([]);
  

  const handleFileUpload = async (files: File[]) => {
    setProcessingStatus('processing');
    setSentencePagePairs([]);
    setPdfPages([]);
    setTocData([]);
    setCsvFilePath(null);
    setBulkSummary([]);

    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });
    formData.append('script', selectedScript);
    formData.append('model', selectedModel);


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
          console.log('Updated tocData state:', data.tocData);

        } else if (selectedScript === 'entity.py' && data.csvFilePath) {
          setCsvFilePath(data.csvFilePath);
          console.log('CSV file path:', data.csvFilePath);
        }

        if (selectedScript === 'bulk_summary.py' && Array.isArray(data.summaries)) {
          setBulkSummary(data.summaries);
          console.log('Updated bulkSummary state:', data.summaries);
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
              <ScriptDropdown
                selectedScript={selectedScript}
                onScriptChange={setSelectedScript}
              />
              <ModelDropdown
                selectedModel={selectedModel}
                onModelChange={setSelectedModel}
              />
            </div>
            <span>Innocence Lab</span>
          </div>

          <div className={styles.outputContent}>
            {processingStatus !== 'completed' && (
              <AnalysisButtons
                selectedAnalysis={selectedAnalysis}
                onAnalysisClick={handleAnalysisClick}
              />
            )}
          </div>
          {processingStatus === 'completed' && selectedScript === 'process.py' && (
                <div ref={outputRef} className={styles.processOutput}>
                  {sentencePagePairs.map((pair, index) => (
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
              )}
          
          
          {processingStatus === 'completed' && selectedScript === 'toc.py' && (
              <div ref={outputRef}>
                <div className={styles.tocSection}>
                  <pre>
                    {tocData.map((pageData, pageIndex) => (
                      pageData.sentence.map((sentenceData, sentenceIndex) => {
                        const text = sentenceData.text;
                        const modifiedText = text.replace(/^(\d+\.)$/gm, '\n$1');
                        return modifiedText.replace(/^\s+/gm, '');
                      }).join('\n')
                    )).join('\n')}
                  </pre>
                </div>
              </div>
            )}
                                                
            {processingStatus === 'completed' && selectedScript === 'entity.py' && (
              <div>
                <button onClick={() => window.open(csvFilePath, '_blank')}>Download CSV</button>
              </div>
            )}
            
            {processingStatus === 'completed' && selectedScript === 'bulk_summary.py' && (
              <div ref={outputRef}>
                {bulkSummary.length > 0 ? (
                  bulkSummary.map((item, index) => (
                    <div key={index}>
                      <h3>File: {item.filename}</h3>
                      <p>Total Pages: {item.total_pages}</p>
                      <p>{item.summary}</p>
                      <hr />
                    </div>
                  ))
                ) : (
                  <p>No summary available</p>
                )}
              </div>
            )}
          <div className={styles.uploadSection}>
            <FileUpload onFileUpload={handleFileUpload} disabled={processingStatus === 'processing'} multiple />
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