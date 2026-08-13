import React, { useState, useEffect, useRef } from 'react';
import { API_BASE } from '../config';

const STAGES = [
  { key: 'cloning', label: 'Clone' },
  { key: 'walking', label: 'Scan' },
  { key: 'parsing', label: 'Parse & Index' },
  { key: 'embedding', label: 'Embed' },
  { key: 'storing', label: 'Finalize' },
];

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
      if (line.startsWith(':')) {
        yield { type: 'heartbeat' };
        continue;
      }
      if (!line.startsWith('data: ')) continue;
      try {
        yield JSON.parse(line.slice(6));
      } catch {}
    }
  }

  if (buffer.startsWith('data: ')) {
    try { yield JSON.parse(buffer.slice(6)); } catch {}
  }
}

const STALL_HINT_MS = 25000;

export default function IndexingProgress({ repoUrl, onProgress, onCancel }) {
  const [stage, setStage] = useState('cloning');
  const [current, setCurrent] = useState(0);
  const [total, setTotal] = useState(0);
  const [message, setMessage] = useState('Connecting to the backend…');
  const [cloneDetail, setCloneDetail] = useState('');
  const [log, setLog] = useState([]);
  const [cancelling, setCancelling] = useState(false);
  const [stalled, setStalled] = useState(false);
  const abortRef = useRef(null);
  const lastEventRef = useRef(Date.now());
  const terminalReceivedRef = useRef(false);

  useEffect(() => {
    const controller = new AbortController();
    abortRef.current = controller;

    const stallTimer = setInterval(() => {
      if (Date.now() - lastEventRef.current > STALL_HINT_MS && !terminalReceivedRef.current) {
        setStalled(true);
      }
    }, 2000);

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
        if (err.name === 'AbortError') return;
        terminalReceivedRef.current = true;
        onProgress({
          type: 'error',
          message: 'Cannot reach the backend server. If hosted on Render Free Tier, the server may be waking up from sleep (takes ~45-60s). Please try again.',
        });
        return;
      }

      if (!res.ok) {
        let detail = `Server error (HTTP ${res.status})`;
        try {
          const body = await res.json();
          detail = body.detail || body.message || detail;
        } catch {
          try {
            const text = await res.text();
            if (text.includes('ECONNREFUSED') || text.includes('proxy error')) {
              detail = 'Backend server is waking up or unreachable. Please wait a moment and try again.';
            }
          } catch {}
        }
        terminalReceivedRef.current = true;
        onProgress({ type: 'error', message: detail });
        return;
      }

      try {
        for await (const event of readSSE(res)) {
          lastEventRef.current = Date.now();
          setStalled(false);
          if (event.type === 'heartbeat') {
            continue;
          }
          if (event.type === 'complete' || event.type === 'error') {
            terminalReceivedRef.current = true;
          }
          onProgress(event);
          if (event.stage) {
            setStage(event.stage);
            if (event.current !== undefined) {
              setCurrent(event.current);
              setTotal(event.total || 0);
            }
          }
          if (event.message) {
            if (event.stage === 'cloning' && event.message.match(/(Receiving|Resolving|Counting|Compressing|remote:|Cloning into|Updating files|Filtering)/i)) {
              setCloneDetail(event.message);
            } else {
              setMessage(event.message);
              setCloneDetail('');
              setLog(prev => [...prev.slice(-6), { stage: event.stage, msg: event.message }]);
            }
          }
        }
        if (!terminalReceivedRef.current) {
          onProgress({
            type: 'error',
            message: 'Connection closed unexpectedly. If on Render free tier, verify your configuration or check logs.',
          });
        }
      } catch (err) {
        if (err.name === 'AbortError') return;
        terminalReceivedRef.current = true;
        onProgress({ type: 'error', message: `Stream error: ${err.message}` });
      }
    };

    const startTimer = setTimeout(startIndexing, 0);

    return () => {
      clearTimeout(startTimer);
      controller.abort();
      clearInterval(stallTimer);
    };
  }, [repoUrl, onProgress]);

  const handleCancel = async () => {
    setCancelling(true);
    abortRef.current?.abort();
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

        {cloneDetail && (
          <p className="index-clone-detail">{cloneDetail}</p>
        )}

        {stalled && (
          <p className="index-stall-hint">
            Waiting for backend response... If your Render instance is waking up, this may take up to a minute.
          </p>
        )}

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