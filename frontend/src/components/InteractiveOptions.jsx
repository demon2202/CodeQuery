import React, { useState } from 'react';

export default function InteractiveOptions({ spec, onSelect }) {
  const [selectedIdx, setSelectedIdx] = useState(null);

  if (!spec || !spec.options || !Array.isArray(spec.options) || spec.options.length === 0) {
    return null;
  }

  const handleOptionClick = (optText, idx) => {
    if (selectedIdx !== null) return;
    setSelectedIdx(idx);
    onSelect?.(optText);
  };

  return (
    <div className="interactive-options-card">
      <div className="options-header">
        <span className="options-badge">Exploration Paths</span>
        <span className="options-title">{spec.question || 'Select an area to explore in detail:'}</span>
      </div>

      <div className="options-grid">
        {spec.options.map((option, idx) => {
          const isSelected = selectedIdx === idx;
          const isOtherSelected = selectedIdx !== null && !isSelected;
          const formattedNumber = String(idx + 1).padStart(2, '0');

          return (
            <button
              key={idx}
              className={`option-btn ${isSelected ? 'selected' : ''} ${isOtherSelected ? 'dimmed' : ''}`}
              onClick={() => handleOptionClick(option, idx)}
              disabled={selectedIdx !== null}
            >
              <span className="option-badge">{formattedNumber}</span>
              <span className="option-text">{option}</span>
              <span className="option-arrow">→</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
