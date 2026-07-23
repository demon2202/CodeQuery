import React, { useState, useEffect } from 'react';
import { API_BASE } from '../config';

export default function CodeCitation({ citation, repoUrl, onClose }) {
  const [code, setCode] = useState('');
  const [lang, setLang] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const fetchCode = async () => {
      setLoading(true);
      setError(null);
      try {
        const params = new URLSearchParams({
          repo_url: repoUrl,
          file_path: citation.file_path,
          start_line: String(citation.start_line),
          end_line: String(citation.end_line),
        });
        const res = await fetch(`${API_BASE}/api/repos/file?${params}`);
        if (!res.ok) throw new Error(res.statusText);
        const data = await res.json();
        if (!cancelled) {
          setCode(data.content);
          setLang(data.language);
          setLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      }
    };
    fetchCode();
    return () => { cancelled = true; };
  }, [citation, repoUrl]);

  const loc = `${citation.file_path}:${citation.start_line}-${citation.end_line}`;

  return (
    <div className="cite-panel">
      <div className="cite-bar">
        <span className="cite-loc">{loc}</span>
        {lang && <span className="cite-lang">{lang}</span>}
        <button className="copy-btn" onClick={() => navigator.clipboard?.writeText(code)} title="Copy code">
          <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"/><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"/></svg>
        </button>
        <button className="cite-close" onClick={onClose}>✕</button>
      </div>
      {loading && <div className="cite-loading">Loading…</div>}
      {error && <div className="cite-error">{error}</div>}
      {!loading && !error && (
        <div className="cite-code">
          <div className="cite-lines">
            {code.split('\n').map((_, i) => (
              <span key={i} className="cite-ln">{i + citation.start_line}</span>
            ))}
          </div>
          <pre className="cite-pre"><code>{code}</code></pre>
        </div>
      )}
    </div>
  );
}