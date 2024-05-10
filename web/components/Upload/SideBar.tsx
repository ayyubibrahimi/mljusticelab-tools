import React from 'react';
import styles from './Sidebar.module.scss';

interface SidebarProps {
  sentencePagePairs?: { sentence: string; page_number?: number }[];
  tocData?: { sentence: string; page_number?: number }[];
  onPageClick: (pageNumber: number) => void;
  onSavedResponseClick: (response: any) => void; // Callback function to handle clicking on a saved response
  savedResponses: { id: number; label: string; content: any }[]; // Array of saved responses
}

const Sidebar: React.FC<SidebarProps> = ({
  sentencePagePairs = [],
  tocData = [],
  onPageClick,
  onSavedResponseClick,
  savedResponses,
}) => {
  const getPageNumbers = () => {
    const processPageNumbers = sentencePagePairs.map(pair => pair.page_number).filter(Boolean) as number[];
    const tocPageNumbers = tocData.map(item => item.page_number).filter(Boolean) as number[];
    return [...new Set([...processPageNumbers, ...tocPageNumbers])];
  };

  const pageNumbers = getPageNumbers();

  return (
    <div className={styles.sidebar}>
      <h3>Saved Responses</h3>
      <ul>
        {savedResponses.map(response => (
          <li key={response.id}>
            <button onClick={() => onSavedResponseClick(response)}>
              {response.label}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Sidebar;