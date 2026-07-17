import React, { useState, useCallback, useEffect, useRef } from 'react';
import RepoInput from './components/RepoInput';
import IndexingProgress from './components/IndexingProgress';
import ChatInterface from './components/ChatInterface';
import PixelBlast from './components/PixelBlast';
import './styles/index.css';

export default function App() {
  const [state, setState] = useState('idle');
  const [repoUrl, setRepoUrl] = useState('');
  const [error, setError] = useState(null);
  const [indexedInfo, setIndexedInfo] = useState(null);
  const [backendOk, setBackendOk] = useState(null);
  const [gitAvailable, setGitAvailable] = useState(true);
  const [keepData, setKeepData] = useState(false);
  const cleanupRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const res = await fetch('/api/chat/health');
        if (!cancelled) {
          if (res.ok) {
            const data = await res.json();
            setBackendOk(true);
            setGitAvailable(data.git_available !== false);
          } else {
            setBackendOk(false);
          }
        }
      } catch {
        if (!cancelled) setBackendOk(false);
      }
    };
    check();
    // Only poll when backend is DOWN. Once it's up, stop hammering it.
    const interval = setInterval(() => {
      if (backendOk !== true) check();
    }, 5000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [backendOk]);

  // Cleanup old repo when starting a new one (if keepData is off)
  const handleIndexStart = useCallback(async (url) => {
    // If there's an existing repo and keepData is off, clean it up
    if (repoUrl && !keepData && !cleanupRef.current) {
      cleanupRef.current = true;
      try {
        await fetch(`/api/repos/index?repo_url=${encodeURIComponent(repoUrl)}&delete_files=true`, {
          method: 'DELETE',
        });
      } catch {}
      cleanupRef.current = false;
    }

    setRepoUrl(url);
    setError(null);
    setIndexedInfo(null);
    setState('indexing');
  }, [repoUrl, keepData]);

  const handleProgress = useCallback((event) => {
    if (event.type === 'complete') {
      setIndexedInfo(event);
      setState('indexed');
    } else if (event.type === 'error') {
      setError(event.message);
      setState('error');
    }
  }, []);

  const handleReset = useCallback(async () => {
    // If keepData is off, delete the repo's index + files
    if (repoUrl && !keepData) {
      try {
        await fetch(`/api/repos/index?repo_url=${encodeURIComponent(repoUrl)}&delete_files=true`, {
          method: 'DELETE',
        });
      } catch {}
    }

    setState('idle');
    setRepoUrl('');
    setError(null);
    setIndexedInfo(null);
  }, [repoUrl, keepData]);

  return (
    <div className="app">
      {state === 'idle' && (
        <div className="bg-container">
          <PixelBlast
            variant="square"
            pixelSize={4}
            color="#B497CF"
            patternScale={2}
            patternDensity={1}
            pixelSizeJitter={0}
            enableRipples
            rippleSpeed={0.4}
            rippleThickness={0.12}
            rippleIntensityScale={1.5}
            liquid={false}
            liquidStrength={0.12}
            liquidRadius={1.2}
            liquidWobbleSpeed={5}
            speed={0.5}
            edgeFade={0.25}
            transparent
          />
        </div>
      )}

      <header className="header">
        <div className="header-left">
          <div className="logo">
            <svg width="40" height="40" viewBox="0 0 20 20" fill="currentColor">
              <rect x="1" y="1" width="7.5" height="7.5" rx="2"/>
              <rect x="11.5" y="1" width="7.5" height="7.5" rx="2"/>
              <rect x="1" y="11.5" width="7.5" height="7.5" rx="2"/>
              <rect x="11.5" y="11.5" width="7.5" height="7.5" rx="2"/>
            </svg>
          </div>
          <h1>CodeQuery</h1>
        </div>
        <div className="header-right">
          {state === 'indexed' && (
            <>
              <label className="keep-toggle" title="Keep repo data on disk after you leave">
                <input
                  type="checkbox"
                  checked={keepData}
                  onChange={(e) => setKeepData(e.target.checked)}
                />
                <span>Keep data</span>
              </label>
              <button className="btn-ghost" onClick={handleReset}>New repo</button>
            </>
          )}
        </div>
      </header>

      {backendOk === false && (
        <div className="backend-banner">
          <span>⚠ Backend not running.</span>
          <code>uvicorn app.main:app --port 8000</code>
          <span className="banner-hint">Retrying…</span>
        </div>
      )}

      {backendOk && !gitAvailable && (
        <div className="backend-banner" style={{ background: 'rgba(251, 191, 36, 0.05)', borderBottomColor: 'rgba(251, 191, 36, 0.15)' }}>
          <span>⚠ Git is not installed or not in PATH.</span>
          <a href="https://git-scm.com/downloads" target="_blank" rel="noopener" style={{ color: '#fbbf24', fontSize: 13 }}>Install Git →</a>
        </div>
      )}

      <main className="main">
        {state === 'idle' && <RepoInput onIndexStart={handleIndexStart} />}

        {state === 'indexing' && (
          <IndexingProgress
            repoUrl={repoUrl}
            onProgress={handleProgress}
          />
        )}

        {state === 'error' && (
          <div className="error-view">
            <div className="error-icon">⚠</div>
            <h2>Indexing Failed</h2>
            <p className="error-detail">{error}</p>
            <button className="btn-primary" onClick={handleReset}>Try Again</button>
          </div>
        )}

        {state === 'indexed' && (
          <div className="chat-view">
            <div className="chat-header">
              <div className="repo-pill">
                <span className="repo-dot"></span>
                <span className="repo-label">{repoUrl.replace('https://github.com/', '')}</span>
              </div>
              <span className="repo-meta">
                {indexedInfo?.files_indexed} files · {indexedInfo?.chunks_created} chunks · {indexedInfo?.time_seconds}s
              </span>
            </div>
            <ChatInterface repoUrl={repoUrl} />
          </div>
        )}
      </main>
    </div>
  );
}
