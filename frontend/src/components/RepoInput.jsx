import React, { useState, useCallback, useRef } from 'react';

export default function RepoInput({ onIndexStart }) {
  const [url, setUrl] = useState('');
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const isValid = /^https?:\/\/(www\.)?github\.com\/[\w.-]+\/[\w.-]+\/?$/.test(url.trim());

  const handleChange = useCallback((e) => {
    setUrl(e.target.value);
    setError(null);
  }, []);

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();
    const trimmed = url.trim().replace(/\/+$/, '');
    if (!trimmed || !isValid) {
      setError('Enter a valid GitHub repo URL');
      return;
    }
    let normalized = trimmed;
    if (!normalized.startsWith('http')) normalized = 'https://' + normalized;
    setSubmitting(true);
    setError(null);
    onIndexStart(normalized.replace(/\/+$/, ''));
  }, [url, isValid, onIndexStart]);

  const examples = [
    { label: 'pallets/click', url: 'https://github.com/pallets/click' },
    { label: 'expressjs/express', url: 'https://github.com/expressjs/express' },
    { label: 'fastapi/fastapi', url: 'https://github.com/fastapi/fastapi' },
  ];

  return (
    <div className="idle-view">
      <div className="idle-hero">
        <h2>Ask your codebase anything</h2>
        <p>Point at a public GitHub repo. Get grounded answers with exact file:line citations.</p>
      </div>

      <form className="url-form" onSubmit={handleSubmit}>
        <div className="url-input-group">
          <svg className="url-icon" width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 0a8 8 0 100 16A8 8 0 008 0zm3.5 8a3.5 3.5 0 11-7 0 3.5 3.5 0 017 0z"/>
          </svg>
          <input
            type="text"
            value={url}
            onChange={handleChange}
            placeholder="https://github.com/owner/repo"
            className="url-input"
            disabled={submitting}
            autoFocus
          />
          <button
            type="submit"
            className={`url-submit ${isValid ? 'url-submit-active' : ''}`}
            disabled={!isValid || submitting}
          >
            {submitting ? 'Starting…' : 'Index →'}
          </button>
        </div>
        {error && <p className="url-error">{error}</p>}
      </form>

      <div className="idle-examples">
        <span className="examples-label">Try:</span>
        {examples.map(ex => (
          <button
            key={ex.url}
            className="example-btn"
            onClick={() => { setUrl(ex.url); setError(null); }}
          >
            {ex.label}
          </button>
        ))}
      </div>
    </div>
  );
}
