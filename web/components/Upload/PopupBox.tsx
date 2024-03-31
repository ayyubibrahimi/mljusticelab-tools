import React, { useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import styles from './PopupBox.module.scss';

pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.js`;

interface PopupBoxProps {
  pageNumber: number;
  onClose: () => void;
  filePath: string | null;
}

const PopupBox: React.FC<PopupBoxProps> = ({ pageNumber, onClose, filePath }) => {
  const [numPages, setNumPages] = useState<number | null>(null);

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
  };

  return (
    <div className={styles.popupOverlay}>
      <div className={styles.popupBox}>
        <div className={styles.header}>
          <button className={styles.closeButton} onClick={onClose}>
            Close
          </button>
        </div>
        <div className={styles.pdfContainer}>
          {filePath ? (
            <Document file={filePath} onLoadSuccess={onDocumentLoadSuccess}>
              <Page pageNumber={pageNumber} className={styles.pdfPage} />
            </Document>
          ) : (
            <div className={styles.noFileMessage}>No PDF file available</div>
          )}
        </div>
        {numPages && (
          <div className={styles.footer}>
            Page {pageNumber} of {numPages}
          </div>
        )}
      </div>
    </div>
  );
};

export default PopupBox;