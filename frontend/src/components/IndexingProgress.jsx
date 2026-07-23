import React, { useState, useEffect, useRef } from 'react';
import { API_BASE } from '../config';

const STAGES = [
  { key: 'cloning', label: 'Clone' },
  { key: 'walking', label: 'Scan' },
  { key: 'parsing', label: 'Parse' },
  { key: 'embedding', label: 'Embed' },
  { key: 'storing', label: 'Store' },
];

/**
 * Parse SSE text from a ReadableStream.
 */
async function* readSSE(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      try {
        yield JSON.parse(line.slice(6));
      } catch {
        // malformed SSE — skip
      }
    }
  }

  if (buffer.startsWith('data: ')) {
    try { yield JSON.parse(buffer.slice(6)); } catch {}
  }
}

export default function IndexingProgress({ repoUrl, onProgress, onCancel }) {
  const [stage, setStage] = useState('cloning');
  const [current, setCurrent] = useState(0);
  const [total, setTotal] = useState(0);
  const [message, setMessage] = useState('');
  const [log, setLog] = useState([]);
  const [cancelling, setCancelling] = useState(false);
  const startedRef = useRef(false);
  const abortRef = useRef(null);

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;

    const controller = new AbortController();
    abortRef.current = controller;

    const startIndexing = async () => {
      let res;
      try {
        res = await fetch(`${API_BASE}/api/repos/index`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ repo_url: repoUrl }),
          signal: controller.signal,
        });
      } catch (err) {
        if (err.name === 'AbortError') return; // user cancelled — handled by handleCancel
        onProgress({
          type: 'error',
          message: 'Cannot reach the backend server. Is it running?\n\nStart it in another terminal:\n  cd backend\n  uvicorn app.main:app --port 8000',
        });
        return;
      }

      // Handle non-200 responses
      if (!res.ok) {
        let detail = `Server error (HTTP ${res.status})`;
        try {
          const body = await res.json();
          detail = body.detail || body.message || detail;
        } catch {
          try {
            const text = await res.text();
            if (text.includes('ECONNREFUSED') || text.includes('proxy error')) {
              detail = 'Backend server is not running. Start it with:\n  cd backend\n  uvicorn app.main:app --port 8000';
            }
          } catch {}
        }
        onProgress({ type: 'error', message: detail });
        return;
      }

      // Stream SSE events
      try {
        for await (const event of readSSE(res)) {
          onProgress(event);
          if (event.stage) {
            setStage(event.stage);
            if (event.current !== undefined) {
              setCurrent(event.current);
              setTotal(event.total || 0);
            }
          }
          if (event.message) {
            setMessage(event.message);
            setLog(prev => [...prev.slice(-6), { stage: event.stage, msg: event.message }]);
          }
        }
      } catch (err) {
        if (err.name === 'AbortError') return; // user cancelled
        onProgress({ type: 'error', message: `Stream error: ${err.message}` });
      }
    };

    startIndexing();

    return () => controller.abort();
  }, [repoUrl, onProgress]);

  const handleCancel = async () => {
    setCancelling(true);
    abortRef.current?.abort();
    // Clean up whatever got partially created (clone/snapshot/chroma data)
    // for this repo — cancelling should leave nothing behind.
    try {
      await fetch(`${API_BASE}/api/repos/index?repo_url=${encodeURIComponent(repoUrl)}&delete_files=true`, {
        method: 'DELETE',
      });
    } catch {}
    onCancel();
  };

  const stageIdx = STAGES.findIndex(s => s.key === stage);
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;
  const repoName = repoUrl.replace('https://github.com/', '');

  return (
    <div className="index-view">
      <div className="index-card">
        <div className="index-title">
          <div className="spinner"></div>
          <span>Indexing <strong>{repoName}</strong></span>
        </div>

        <div className="stage-bar">
          {STAGES.map((s, i) => (
            <div
              key={s.key}
              className={`stage-pip ${i < stageIdx ? 'done' : ''} ${i === stageIdx ? 'active' : ''}`}
            >
              <div className="pip-dot">{i < stageIdx ? '✓' : i + 1}</div>
              <div className="pip-label">{s.label}</div>
            </div>
          ))}
        </div>

        {total > 0 && (
          <div className="progress-row">
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${pct}%` }}></div>
            </div>
            <span className="progress-num">{current}/{total}</span>
          </div>
        )}

        <p className="index-msg">{message}</p>

        {log.length > 0 && (
          <div className="index-log">
            {log.map((entry, i) => (
              <div key={i} className="log-line">
                <span className="log-stage">{entry.stage}</span>
                <span className="log-msg">{entry.msg}</span>
              </div>
            ))}
          </div>
        )}

        <div className="index-footer">
          <button className="btn-cancel" onClick={handleCancel} disabled={cancelling}>
            {cancelling ? 'Cancelling…' : 'Cancel & discard'}
          </button>
        </div>
      </div>
    </div>
  );
}