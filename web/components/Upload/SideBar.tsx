import React from 'react';
import styles from './Sidebar.module.scss';

interface SidebarProps {
  pages?: number[];
  onPageClick: (pageNumber: number) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ pages = [], onPageClick }) => {
  return (
    <div className={styles.sidebar}>
      <h3>Citations</h3>
      <ul>
        {pages.map((pageNumber) => (
          <li key={pageNumber}>
            <button onClick={() => onPageClick(pageNumber)}>Page {pageNumber}</button>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Sidebar;