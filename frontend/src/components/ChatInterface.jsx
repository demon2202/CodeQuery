import React, { useState, useCallback, useRef, useEffect, lazy, Suspense } from 'react';
import { API_BASE } from '../config';

const CodeCitation = lazy(() => import('./CodeCitation'));
const MessageContent = lazy(() => import('./MessageContent'));

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
      try { yield JSON.parse(line.slice(6)); } catch {}
    }
  }
  if (buffer.startsWith('data: ')) {
    try { yield JSON.parse(buffer.slice(6)); } catch {}
  }
}

function backendErrorMessage(status) {
  if (status === 0 || status === 502 || status === 503 || status === 504) {
    return 'Backend server is not responding. Make sure it\'s running:\n\n  cd backend\n  uvicorn app.main:app --port 8000';
  }
  return `Server error (HTTP ${status})`;
}

export default function ChatInterface({ repoUrl, askAboutFile, onAskHandled }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [activeCitation, setActiveCitation] = useState(null);
  const [starters, setStarters] = useState([]);
  const [llmStarters, setLlmStarters] = useState(null);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const messagesRef = useRef([]);

  // Keep messagesRef in sync
  useEffect(() => { messagesRef.current = messages; }, [messages]);

  // Handle "Ask about file" from FileTree
  useEffect(() => {
    if (askAboutFile && askAboutFile.path && !streaming) {
      const q = `Explain what ${askAboutFile.path} does and how it works`;
      setInput(q);
      inputRef.current?.focus();
      // Auto-send after a brief moment
      setTimeout(() => {
        if (inputRef.current) {
          const form = inputRef.current.closest('form');
          if (form) form.requestSubmit();
        }
      }, 100);
      onAskHandled?.();
    }
  }, [askAboutFile]);

  // Fetch dynamic starters
  useEffect(() => {
    let cancelled = false;
    const fetchStarters = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/chat/starters?repo_url=${encodeURIComponent(repoUrl)}`);
        if (!cancelled && res.ok) {
          const data = await res.json();
          setStarters(data.starters || []);
          setLlmStarters(data.llm_starters || null);
        }
      } catch {}
    };
    fetchStarters();
    return () => { cancelled = true; };
  }, [repoUrl]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = useCallback(async (e, overrideQ) => {
    e?.preventDefault();
    const q = (overrideQ || input).trim();
    if (!q || streaming) return;

    setInput('');
    setStreaming(true);
    const uid = Date.now();
    const aid = uid + 1;

    setMessages(prev => [
      ...prev,
      { id: uid, role: 'user', content: q },
      { id: aid, role: 'assistant', content: '', sources: [], streaming: true },
    ]);

    // Build conversation history from previous messages (last 4 Q&A pairs)
    const history = [];
    const completedMsgs = messagesRef.current.filter(m => !m.streaming && m.content);
    for (let i = 0; i < completedMsgs.length; i++) {
      const m = completedMsgs[i];
      if (m.role === 'user' && i + 1 < completedMsgs.length && completedMsgs[i + 1].role === 'assistant') {
        history.push({
          question: m.content,
          answer: completedMsgs[i + 1].content.slice(0, 500),
        });
      }
    }

    let res;
    try {
      res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_url: repoUrl, question: q, history: history.slice(-4) }),
      });
    } catch (err) {
      setMessages(prev => prev.map(m =>
        m.id === aid ? { ...m, content: 'Cannot reach backend. Is the server running?', streaming: false } : m
      ));
      setStreaming(false);
      return;
    }

    if (!res.ok) {
      let detail = backendErrorMessage(res.status);
      try {
        const body = await res.json();
        detail = body.detail || detail;
      } catch {}
      setMessages(prev => prev.map(m =>
        m.id === aid ? { ...m, content: detail, streaming: false } : m
      ));
      setStreaming(false);
      return;
    }

    try {
      for await (const ev of readSSE(res)) {
        if (ev.type === 'sources') {
          setMessages(prev => prev.map(m =>
            m.id === aid ? { ...m, sources: ev.chunks || [] } : m
          ));
        } else if (ev.type === 'token') {
          setMessages(prev => prev.map(m =>
            m.id === aid ? { ...m, content: m.content + ev.text } : m
          ));
        } else if (ev.type === 'done') {
          setMessages(prev => prev.map(m =>
            m.id === aid ? { ...m, content: ev.answer || m.content, streaming: false } : m
          ));
        } else if (ev.type === 'error') {
          setMessages(prev => prev.map(m =>
            m.id === aid ? { ...m, content: `⚠ ${ev.message}`, streaming: false } : m
          ));
        }
      }
    } catch (err) {
      setMessages(prev => prev.map(m =>
        m.id === aid ? { ...m, content: `Stream error: ${err.message}`, streaming: false } : m
      ));
    }
    setStreaming(false);
  }, [input, repoUrl, streaming]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }, [send]);

  // Merge starters: show LLM ones if available, else file-based
  const displayStarters = llmStarters || starters;

  return (
    <div className="chat">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <div className="empty-icon">💬</div>
            <h3>Ask about this codebase</h3>
            <div className="starter-btns">
              {displayStarters.map(s => (
                <button key={s} className="starter" onClick={() => { setInput(s); inputRef.current?.focus(); send(null, s); }}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map(msg => (
          <div key={msg.id} className={`msg msg-${msg.role}`}>
            <div className="msg-avatar">{msg.role === 'user' ? '→' : '✦'}</div>
            <div className="msg-body">
              {msg.role === 'assistant' && msg.sources?.length > 0 && (
                <div className="sources">
                  {msg.sources.map((src, i) => (
                    <button
                      key={i}
                      className={`src-btn ${activeCitation?.file_path === src.file_path && activeCitation?.start_line === src.start_line ? 'src-btn-active' : ''}`}
                      onClick={() => setActiveCitation(prev =>
                        prev?.file_path === src.file_path && prev?.start_line === src.start_line ? null : src
                      )}
                      title={`${src.file_path}:${src.start_line}-${src.end_line}`}
                    >
                      <span className="src-file">{src.file_path.split('/').pop()}</span>
                      <span className="src-loc">:{src.start_line}</span>
                    </button>
                  ))}
                </div>
              )}

              {msg.role === 'assistant' && activeCitation && msg.sources?.some(
                s => s.file_path === activeCitation.file_path && s.start_line === activeCitation.start_line
              ) && (
                <Suspense fallback={<div className="cite-loading">Loading code…</div>}>
                  <CodeCitation citation={activeCitation} repoUrl={repoUrl} onClose={() => setActiveCitation(null)} />
                </Suspense>
              )}

              <div className="msg-text">
                {msg.role === 'user' ? msg.content : (
                  msg.content ? (
                    <Suspense fallback={<span>{msg.content}</span>}>
                      <MessageContent content={msg.content} />
                    </Suspense>
                  ) : (msg.streaming ? <span className="cursor">▍</span> : '')
                )}
              </div>
              {msg.role === 'assistant' && !msg.streaming && msg.content && (
                <button className="copy-answer-btn" onClick={() => navigator.clipboard?.writeText(msg.content)} title="Copy answer">
                  <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"/><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"/></svg>
                </button>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <form className="chat-bar" onSubmit={send}>
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={streaming ? 'Waiting…' : 'Ask about this codebase…'}
          disabled={streaming}
          className="chat-input"
        />
        <button type="submit" className="chat-send" disabled={!input.trim() || streaming}>
          <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
            <path d="M1 1l14 7-14 7V9l9-1-9-1V1z"/>
          </svg>
        </button>
      </form>
    </div>
  );
}