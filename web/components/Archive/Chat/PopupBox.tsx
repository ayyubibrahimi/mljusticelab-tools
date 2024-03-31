import React from 'react';
import { Document, Page } from 'react-pdf';
import styles from './PopupBox.module.scss';

interface PopupBoxProps {
  pageNumber: number;
  onClose: () => void;
  filePath: string | null;
}

const PopupBox: React.FC<PopupBoxProps> = ({ pageNumber, onClose, filePath }) => {
  return (
    <div className={styles.popupOverlay}>
      <div className={styles.popupBox}>
        <button className={styles.closeButton} onClick={onClose}>
          Close
        </button>
        {filePath && (
          <Document file={filePath} onLoadSuccess={() => {}}>
            <Page pageNumber={pageNumber} />
          </Document>
        )}
      </div>
    </div>
  );
};

export default PopupBox;