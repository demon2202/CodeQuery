import React, { useState, useEffect } from 'react';
import MessageContent from './MessageContent';

export default function RepoSummary({ repoUrl, filesInfo, onClose }) {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const fetchSummary = async () => {
      try {
        const res = await fetch(`/api/repos/summary?repo_url=${encodeURIComponent(repoUrl)}`);
        if (!cancelled && res.ok) {
          const data = await res.json();
          setSummary(data);
        } else if (!cancelled) {
          setError('Failed to generate summary');
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
      if (!cancelled) setLoading(false);
    };
    fetchSummary();
    return () => { cancelled = true; };
  }, [repoUrl]);

  // Quick stats from indexing info (always available immediately)
  const stats = summary?.stats || null;

  return (
    <div className="summary-card">
      <div className="summary-header">
        <span className="summary-title">Repo Overview</span>
        <button className="summary-close" onClick={onClose}>✕</button>
      </div>

      {/* Immediate stats */}
      {stats && (
        <div className="summary-stats">
          <div className="stat-item">
            <span className="stat-val">{stats.languages?.length || 0}</span>
            <span className="stat-label">Languages</span>
          </div>
          <div className="stat-item">
            <span className="stat-val">{stats.files || 0}</span>
            <span className="stat-label">Files</span>
          </div>
          <div className="stat-item">
            <span className="stat-val">{stats.chunks || 0}</span>
            <span className="stat-label">Chunks</span>
          </div>
          <div className="stat-item">
            <span className="stat-val">{stats.functions || 0}</span>
            <span className="stat-label">Functions</span>
          </div>
        </div>
      )}

      {/* Language breakdown */}
      {stats?.languages && (
        <div className="summary-langs">
          {stats.languages.map((l, i) => (
            <div key={i} className="lang-bar-row">
              <span className="lang-name">{l.name}</span>
              <div className="lang-bar-track">
                <div className="lang-bar-fill" style={{ width: `${l.pct}%` }}></div>
              </div>
              <span className="lang-pct">{l.pct}%</span>
            </div>
          ))}
        </div>
      )}

      {/* LLM-generated summary */}
      {loading && <div className="summary-loading">Generating overview…</div>}
      {error && <div className="summary-error">{error}</div>}
      {summary?.text && (
        <div className="summary-text">
          <MessageContent content={summary.text} />
        </div>
      )}
    </div>
  );
}
