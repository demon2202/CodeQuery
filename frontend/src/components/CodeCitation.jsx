import React, { useState, useEffect } from 'react';

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
        const res = await fetch(`/api/repos/file?${params}`);
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
