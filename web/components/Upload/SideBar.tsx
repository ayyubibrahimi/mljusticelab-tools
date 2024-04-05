import React from 'react';
import styles from './Sidebar.module.scss';

interface SidebarProps {
  sentencePagePairs: { sentence: string; page_number?: number }[];
  tocData: { sentence: string; page_number?: number }[];
  onPageClick: (pageNumber: number) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ sentencePagePairs, tocData, onPageClick }) => {
  const getPageNumbers = () => {
    const processPageNumbers = sentencePagePairs.map(pair => pair.page_number).filter(Boolean) as number[];
    const tocPageNumbers = tocData.map(item => item.page_number).filter(Boolean) as number[];
    return [...new Set([...processPageNumbers, ...tocPageNumbers])];
  };

  const pageNumbers = getPageNumbers();

  return (
    <div className={styles.sidebar}>
      <h3>Citations</h3>
      <ul>
        {pageNumbers.map(pageNumber => (
          <li key={pageNumber}>
            <button onClick={() => onPageClick(pageNumber)}>Page {pageNumber}</button>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Sidebar;