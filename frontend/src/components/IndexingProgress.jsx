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

const STALL_HINT_MS = 20000; // show a hint if no progress update for 20s

export default function IndexingProgress({ repoUrl, onProgress, onCancel }) {
  const [stage, setStage] = useState('cloning');
  const [current, setCurrent] = useState(0);
  const [total, setTotal] = useState(0);
  const [message, setMessage] = useState('');
  const [cloneDetail, setCloneDetail] = useState(''); // raw git progress line
  const [log, setLog] = useState([]);
  const [cancelling, setCancelling] = useState(false);
  const [stalled, setStalled] = useState(false);
  const startedRef = useRef(false);
  const abortRef = useRef(null);
  const lastEventRef = useRef(Date.now());
  const terminalReceivedRef = useRef(false);

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;

    const controller = new AbortController();
    abortRef.current = controller;

    // Surface a "this is taking a while" hint if nothing's happened in a
    // while — most commonly a Render free-tier cold start waking up, or a
    // slow GitHub clone. Doesn't fail anything, just stops the spinner from
    // looking broken with zero explanation.
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
        if (err.name === 'AbortError') return; // user cancelled — handled by handleCancel
        terminalReceivedRef.current = true;
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
        terminalReceivedRef.current = true;
        onProgress({ type: 'error', message: detail });
        return;
      }

      // Stream SSE events
      try {
        for await (const event of readSSE(res)) {
          lastEventRef.current = Date.now();
          setStalled(false);
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
            // During cloning, git emits many raw progress lines ("Receiving objects: 42%").
            // Show them as a sub-detail instead of the main message to avoid log spam.
            if (event.stage === 'cloning' && event.message.match(/(Receiving|Resolving|Counting|Compressing|remote:|Cloning into|Updating files|Filtering)/i)) {
              setCloneDetail(event.message);
            } else {
              setMessage(event.message);
              setCloneDetail('');
              setLog(prev => [...prev.slice(-6), { stage: event.stage, msg: event.message }]);
            }
          }
        }
        // Stream ended (server closed the connection) without ever sending
        // a 'complete' or 'error' event — this used to leave the UI frozen
        // on whatever stage it last saw, forever, with no indication
        // anything had gone wrong. Treat it as an explicit failure instead.
        if (!terminalReceivedRef.current) {
          onProgress({
            type: 'error',
            message: 'Connection to the server closed before indexing finished. This can happen on a slow network, a large repo, or if the server restarted mid-request (e.g. a free-tier host going to sleep). Try again — if it keeps happening, try a smaller repo.',
          });
        }
      } catch (err) {
        if (err.name === 'AbortError') return; // user cancelled
        terminalReceivedRef.current = true;
        onProgress({ type: 'error', message: `Stream error: ${err.message}` });
      }
    };

    startIndexing();

    return () => {
      controller.abort();
      clearInterval(stallTimer);
    };
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

        {cloneDetail && (
          <p className="index-clone-detail">{cloneDetail}</p>
        )}

        {stalled && (
          <p className="index-stall-hint">
            ⏳ This is taking a while — large repos or slow networks can take 60–120s to clone. Hang tight!
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