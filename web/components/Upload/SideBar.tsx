import React, { useState } from 'react';
import styles from './Sidebar.module.scss';

const Sidebar = ({
  pages,
  onPageClick,
  sentencePagePairs,
  tocData,
  onSavedResponseClick,
  savedResponses,
  onDeleteSavedResponse,
  onRenameSavedResponse,
}) => {
  const [editingResponseId, setEditingResponseId] = useState(null);
  const [newLabel, setNewLabel] = useState('');

  const handleRenameClick = (responseId) => {
    setEditingResponseId(responseId);
    const response = savedResponses.find((resp) => resp.id === responseId);
    setNewLabel(response.label);
  };

  const handleRenameSubmit = (responseId) => {
    onRenameSavedResponse(responseId, newLabel);
    setEditingResponseId(null);
    setNewLabel('');
  };

  return (
    <div className={styles.sidebar}>
      <div className={styles.savedResponses}>
        <h3>Saved Responses</h3>
        {savedResponses.map((response) => (
          <div key={response.id} className={styles.savedResponse}>
            {editingResponseId === response.id ? (
              <div>
                <input
                  type="text"
                  value={newLabel}
                  onChange={(e) => setNewLabel(e.target.value)}
                  className={styles.renameInput}
                />
                <button
                  onClick={() => handleRenameSubmit(response.id)}
                  className={styles.renameButton}
                >
                  Save
                </button>
              </div>
            ) : (
              <div>
                <button
                  onClick={() => onSavedResponseClick(response)}
                  className={styles.responseLabel}
                >
                  {response.label}
                </button>
                <div className={styles.responseButtons}>
                  <button onClick={() => handleRenameClick(response.id)}>Rename</button>
                  <button onClick={() => onDeleteSavedResponse(response.id)}>Delete</button>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Sidebar;