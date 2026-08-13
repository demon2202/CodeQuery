import React, { useState, useCallback, useRef, useEffect, lazy, Suspense } from 'react';
import { API_BASE } from '../config';

const CodeCitation = lazy(() => import('./CodeCitation'));
const MessageContent = lazy(() => import('./MessageContent'));

const SLASH_COMMANDS = [
  {
    cmd: '/grill-me',
    label: '/grill-me',
    desc: 'Interactive guided exploration of architecture & design trade-offs',
    prompt: '/grill-me on key architectural components and design decisions',
  },
  {
    cmd: '/architecture',
    label: '/architecture',
    desc: 'Generate complete multi-tier architecture breakdown and diagram',
    prompt: '/architecture of this repository with full component diagram',
  },
  {
    cmd: '/dataflow',
    label: '/dataflow',
    desc: 'Trace end-to-end request lifecycle and sequence diagram',
    prompt: '/dataflow tracing main request lifecycle and event pipeline',
  },
  {
    cmd: '/summary',
    label: '/summary',
    desc: 'High-level technical overview, entrypoints, and tech stack',
    prompt: '/summary of key modules, purpose, and tech stack',
  },
  {
    cmd: '/components',
    label: '/components',
    desc: 'Breakdown of primary modules, services, and classes',
    prompt: 'Explain the core components, modules, and file responsibilities',
  },
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
    return 'Backend server is not responding. Ensure the backend is active at port 8000.';
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
  const [showSlashMenu, setShowSlashMenu] = useState(false);
  const [slashIdx, setSlashIdx] = useState(0);

  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const messagesRef = useRef([]);

  useEffect(() => { messagesRef.current = messages; }, [messages]);

  useEffect(() => {
    if (askAboutFile && askAboutFile.path && !streaming) {
      const q = `Explain what ${askAboutFile.path} does and how it works`;
      setInput(q);
      inputRef.current?.focus();
      setTimeout(() => {
        if (inputRef.current) {
          const form = inputRef.current.closest('form');
          if (form) form.requestSubmit();
        }
      }, 100);
      onAskHandled?.();
    }
  }, [askAboutFile]);

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

  const filteredCommands = input.startsWith('/')
    ? SLASH_COMMANDS.filter(c => c.cmd.toLowerCase().startsWith(input.trim().toLowerCase()) || input === '/')
    : [];

  useEffect(() => {
    setShowSlashMenu(input.startsWith('/') && filteredCommands.length > 0);
    setSlashIdx(0);
  }, [input, filteredCommands.length]);

  const send = useCallback(async (e, overrideQ) => {
    e?.preventDefault();
    const q = (overrideQ || input).trim();
    if (!q || streaming) return;

    setInput('');
    setShowSlashMenu(false);
    setStreaming(true);
    const uid = Date.now();
    const aid = uid + 1;

    setMessages(prev => [
      ...prev,
      { id: uid, role: 'user', content: q },
      { id: aid, role: 'assistant', content: '', sources: [], streaming: true },
    ]);

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
            m.id === aid ? { ...m, content: `Error: ${ev.message}`, streaming: false } : m
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

  const handleSelectOption = useCallback((optionText) => {
    send(null, optionText);
  }, [send]);

  const handleSelectSlash = (cmdObj) => {
    setShowSlashMenu(false);
    send(null, cmdObj.prompt);
  };

  const handleKeyDown = useCallback((e) => {
    if (showSlashMenu && filteredCommands.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSlashIdx(idx => (idx + 1) % filteredCommands.length);
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSlashIdx(idx => (idx - 1 + filteredCommands.length) % filteredCommands.length);
        return;
      }
      if (e.key === 'Tab' || (e.key === 'Enter' && !e.shiftKey)) {
        e.preventDefault();
        const selected = filteredCommands[slashIdx];
        if (selected) {
          handleSelectSlash(selected);
          return;
        }
      }
      if (e.key === 'Escape') {
        setShowSlashMenu(false);
        return;
      }
    }

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }, [send, showSlashMenu, filteredCommands, slashIdx]);

  const displayStarters = llmStarters || starters;

  return (
    <div className="chat">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <div className="empty-tag">CODEBASE INTELLIGENCE</div>
            <h3>Query architecture, data flows, and code patterns</h3>
            <p className="empty-sub">Type a question or select a command below. Type <code>/</code> for command menu.</p>
            <div className="starter-btns">
              <button
                className="starter starter-cmd"
                onClick={() => send(null, "/grill-me on this codebase architecture and key design decisions")}
              >
                <span className="starter-cmd-name">/grill-me</span>
                <span className="starter-cmd-desc">Guided architectural interview</span>
              </button>

              <button
                className="starter starter-cmd"
                onClick={() => send(null, "/architecture of this repository with full component diagram")}
              >
                <span className="starter-cmd-name">/architecture</span>
                <span className="starter-cmd-desc">Full system component diagram</span>
              </button>

              <button
                className="starter starter-cmd"
                onClick={() => send(null, "/dataflow tracing main request lifecycle and event pipeline")}
              >
                <span className="starter-cmd-name">/dataflow</span>
                <span className="starter-cmd-desc">End-to-end request sequence</span>
              </button>

              {displayStarters.slice(0, 3).map(s => (
                <button key={s} className="starter" onClick={() => { setInput(s); inputRef.current?.focus(); send(null, s); }}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map(msg => (
          <div key={msg.id} className={`msg msg-${msg.role}`}>
            <div className="msg-avatar-badge">{msg.role === 'user' ? 'USER' : 'CQ'}</div>
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
                <Suspense fallback={<div className="cite-loading">Loading code citation…</div>}>
                  <CodeCitation citation={activeCitation} repoUrl={repoUrl} onClose={() => setActiveCitation(null)} />
                </Suspense>
              )}

              <div className="msg-text">
                {msg.role === 'user' ? msg.content : (
                  msg.content ? (
                    <Suspense fallback={<span>{msg.content}</span>}>
                      <MessageContent content={msg.content} onSelectOption={handleSelectOption} />
                    </Suspense>
                  ) : (msg.streaming ? <span className="cursor">▍</span> : '')
                )}
              </div>

              {msg.role === 'assistant' && !msg.streaming && msg.content && (
                <div className="msg-actions">
                  <button className="copy-answer-btn" onClick={() => navigator.clipboard?.writeText(msg.content)} title="Copy raw answer text">
                    Copy Answer
                  </button>
                  <button
                    className="grill-action-btn"
                    onClick={() => send(null, `/grill-me on the key components of ${repoUrl.split('/').pop()}`)}
                    title="Start guided exploration of this topic"
                  >
                    /grill-me
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-wrapper">
        {/* Slash Command Autocomplete Popover */}
        {showSlashMenu && (
          <div className="slash-menu">
            <div className="slash-menu-header">COMMANDS</div>
            {filteredCommands.map((c, i) => (
              <div
                key={c.cmd}
                className={`slash-item ${i === slashIdx ? 'active' : ''}`}
                onClick={() => handleSelectSlash(c)}
              >
                <div className="slash-item-cmd">{c.label}</div>
                <div className="slash-item-desc">{c.desc}</div>
              </div>
            ))}
          </div>
        )}

        <form className="chat-bar" onSubmit={send}>
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={streaming ? 'Analyzing codebase…' : 'Ask anything, or type / for commands (/grill-me, /architecture)…'}
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
    </div>
  );
}