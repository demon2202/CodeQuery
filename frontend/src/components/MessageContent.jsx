/**
 * MessageContent — renders assistant message content with:
 * - Interactive Options / Grill-Me blocks (```options with JSON)
 * - Chart.js charts (```chart blocks with JSON)
 * - Mermaid diagrams (```mermaid blocks)
 * - Code blocks with token syntax highlighting, line numbers, language label, copy button
 * - Bold, inline code, bullet lists, and basic formatting
 * - Clickable file:line citations
 *
 * The syntax highlighter is intentionally dependency-free (regex tokenizer),
 * keeping the bundle small and working offline in embedded previews.
 */

import React, { useMemo, lazy, Suspense } from 'react';

const MermaidDiagram = lazy(() => import('./MermaidDiagram'));
const ChartComponent = lazy(() => import('./ChartComponent'));
const InteractiveOptions = lazy(() => import('./InteractiveOptions'));

// ── Zero-dependency syntax highlighting ──────────────────────────────────────

const KEYWORDS = {
  python: 'def|class|import|from|as|return|if|elif|else|for|while|try|except|finally|with|lambda|pass|raise|yield|async|await|None|True|False|self|is|in|not|and|or|del|global|nonlocal|assert|print',
  javascript: 'const|let|var|function|return|if|else|for|while|do|switch|case|default|break|continue|new|class|extends|super|this|null|undefined|true|false|import|from|export|async|await|try|catch|finally|throw|typeof|instanceof|in|of|delete|void|yield',
  typescript:
    'const|let|var|function|return|if|else|for|while|do|switch|case|default|break|continue|new|class|extends|super|this|null|undefined|true|false|import|from|export|async|await|try|catch|finally|throw|typeof|instanceof|in|of|delete|void|yield|interface|type|implements|enum|namespace|readonly|public|private|protected|abstract|string|number|boolean|any|void|as|satisfies',
  generic: 'if|else|for|while|return|function|class|struct|fn|pub|use|mod|package|import|include|void|int|char|long|static|const|let|var|new|delete|try|catch|throw|func|go|defer|select|chan|match|case|end|do|then|elsif|begin|define|require|local|nil|true|false|null',
};

const _tokenizerCache = new Map();

function _getTokenizer(lang) {
  const l = (lang || '').toLowerCase();
  let key = 'generic';
  if (/^(py|python)$/.test(l)) key = 'python';
  else if (/^(js|jsx|mjs|cjs|javascript|tsx?|typescript)$/.test(l)) key = l.startsWith('ts') || l === 'typescript' ? 'typescript' : 'javascript';

  if (_tokenizerCache.has(key)) return _tokenizerCache.get(key);

  const isPython = key === 'python';
  // Single-line string matcher (not escape-aware on purpose): good enough for
  // highlighting and keeps the JS source free of escape-nesting hazards.
  const commentAlt = isPython
    ? '#[^\\n]*'
    : '\\/\\/[^\\n]*|\\/\\*[\\s\\S]*?\\*\\/';
  const stringAlt = isPython
    ? '"""[\\s\\S]*?"""|\'\'\'[\\s\\S]*?\'\'\'|"[^"\\n]*"|\'[^\'\\n]*\''
    : '"[^"\\n]*"|\'[^\'\\n]*\'|`[^`]*`';
  const kw = KEYWORDS[key];
  const regex = new RegExp(
    '(' + commentAlt + ')|(' + stringAlt + ')|(\\b\\d[\\d_\\.]*\\b)|(\\b(?:' + kw + ')\\b)',
    'g'
  );

  const tok = { regex, isPython };
  _tokenizerCache.set(key, tok);
  return tok;
}

function highlightLine(lineText, lang) {
  if (!lang || lang === 'text' || lang === 'markdown' || lang === 'md') return lineText;
  const { regex } = _getTokenizer(lang);
  const nodes = [];
  let last = 0;
  let match;
  let k = 0;
  regex.lastIndex = 0;
  while ((match = regex.exec(lineText)) !== null) {
    if (match.index > last) nodes.push(lineText.slice(last, match.index));
    const cls = match[1] ? 'tok-com'
      : match[2] ? 'tok-str'
      : match[3] ? 'tok-num'
      : match[4] ? 'tok-kw'
      : null;
    if (cls) {
      nodes.push(<span key={k++} className={cls}>{match[0]}</span>);
    } else {
      nodes.push(match[0]);
    }
    last = match.index + match[0].length;
  }
  if (last < lineText.length) nodes.push(lineText.slice(last));
  return nodes.length ? nodes : lineText;
}

const HIGHLIGHTABLE = /^(py|python|js|jsx|mjs|cjs|javascript|ts|tsx|typescript|c|h|cpp|cc|cxx|cs|java|go|rs|rust|rb|ruby|php|sh|bash|zsh|lua|swift|kt|scala)$/i;

function CodeBlock({ lang, code }) {
  const lines = useMemo(() => code.split('\n'), [code]);
  const canHighlight = HIGHLIGHTABLE.test(lang || '');
  const showGutter = lines.length > 2;

  return (
    <div className="msg-code-block">
      <div className="msg-code-top">
        {lang && <span className="msg-code-lang">{lang}</span>}
        <button className="copy-btn" onClick={() => navigator.clipboard?.writeText(code)} title="Copy code">
          <svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"/><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"/></svg>
        </button>
      </div>
      <pre className={showGutter ? 'with-gutter' : ''}>
        <code>
          {lines.map((line, i) => (
            <span key={i} className="code-line">
              {showGutter && <span className="line-no" aria-hidden="true">{i + 1}</span>}
              <span className="line-src">{canHighlight ? highlightLine(line, lang) : line}</span>
              {'\n'}
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}

export default function MessageContent({ content, onSelectOption }) {
  const parts = useMemo(() => parseContent(content), [content]);

  return (
    <>
      {parts.map((part, i) => {
        if (part.type === 'options') {
          return (
            <Suspense key={i} fallback={<div className="options-loading">Loading options…</div>}>
              <InteractiveOptions spec={part.spec} onSelect={onSelectOption} />
            </Suspense>
          );
        }
        if (part.type === 'chart') {
          return (
            <Suspense key={i} fallback={<div className="chart-loading">Loading chart…</div>}>
              <ChartComponent spec={part.spec} />
            </Suspense>
          );
        }
        if (part.type === 'mermaid') {
          return (
            <Suspense key={i} fallback={<div className="mermaid-loading">Loading diagram…</div>}>
              <MermaidDiagram code={part.code} />
            </Suspense>
          );
        }
        if (part.type === 'code') {
          return <CodeBlock key={i} lang={part.lang} code={part.code} />;
        }
        // Rich markdown rendering: headings, tables, lists, quotes, hr, inline styles
        return renderMarkdownBlock(part.text, i);
      })}
    </>
  );
}


function parseContent(text) {
  if (!text) return [];

  const parts = [];
  const regex = /```(\w*)\n([\s\S]*?)```/g;
  let lastIdx = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIdx) {
      parts.push({ type: 'text', text: text.slice(lastIdx, match.index) });
    }

    const lang = match[1].toLowerCase();
    const code = match[2].trim();

    if (lang === 'options' || lang === 'interactive') {
      const spec = tryParseOptionsJSON(code);
      if (spec) {
        parts.push({ type: 'options', spec });
      } else {
        parts.push({ type: 'code', lang: 'options', code });
      }
    } else if (lang === 'chart') {
      const spec = tryParseChartJSON(code);
      if (spec) {
        parts.push({ type: 'chart', spec });
        } else {
        parts.push({ type: 'code', lang: 'chart (invalid JSON)', code });
      }
    } else if (lang === 'mermaid') {
      parts.push({ type: 'mermaid', code });
    } else {
      parts.push({ type: 'code', lang: lang || 'text', code });
    }

    lastIdx = match.index + match[0].length;
  }

  if (lastIdx < text.length) {
    parts.push({ type: 'text', text: text.slice(lastIdx) });
  }

  return parts;
}


function tryParseOptionsJSON(code) {
  try {
    const spec = JSON.parse(code);
    if (spec.options && Array.isArray(spec.options)) return spec;
  } catch {}

  try {
    const cleaned = code.replace(/,\s*([}\]])/g, '$1').replace(/'/g, '"');
    const spec = JSON.parse(cleaned);
    if (spec.options && Array.isArray(spec.options)) return spec;
  } catch {}

  try {
    const match = code.match(/\{[\s\S]*"options"\s*:[\s\S]*\}/);
    if (match) {
      const cleaned = match[0].replace(/,\s*([}\]])/g, '$1');
      const spec = JSON.parse(cleaned);
      if (spec.options && Array.isArray(spec.options)) return spec;
    }
  } catch {}

  return null;
}


/**
 * Try to parse chart JSON — handles common LLM mistakes
 */
function tryParseChartJSON(code) {
  try {
    const spec = JSON.parse(code);
    if (spec.type && spec.labels && spec.datasets) return spec;
  } catch {}

  try {
    const cleaned = code.replace(/,\s*([}\]])/g, '$1');
    const spec = JSON.parse(cleaned);
    if (spec.type && spec.labels && spec.datasets) return spec;
  } catch {}

  try {
    const cleaned = code.replace(/'/g, '"').replace(/,\s*([}\]])/g, '$1');
    const spec = JSON.parse(cleaned);
    if (spec.type && spec.labels && spec.datasets) return spec;
  } catch {}

  try {
    const jsonMatch = code.match(/\{[\s\S]*"type"\s*:[\s\S]*\}/);
    if (jsonMatch) {
      const cleaned = jsonMatch[0].replace(/,\s*([}\]])/g, '$1');
      const spec = JSON.parse(cleaned);
      if (spec.type && spec.labels && spec.datasets) return spec;
    }
  } catch {}

  return null;
}


// ── Lightweight markdown engine (no dependencies) ─────────────────────────
// Handles what LLM answers actually contain: h1–h6 headings, pipe tables,
// bullet + numbered lists (incl. • and - and *), blockquotes, horizontal
// rules, and inline **bold** / *italic* / ~~strike~~ / `code` / [links](url).

const INLINE_RE = /`([^`]+)`|\*\*([^*]+?)\*\*|\*([^*\n]+?)\*|~~([^~]+?)~~|\[([^\]]+)\]\((https?:[^)\s]+)\)/g;

const HEADING_RE = /^(#{1,6})\s+(.+?)\s*$/;
const HR_RE = /^\s*(-{3,}|\*{3,}|_{3,})\s*$/;
const QUOTE_RE = /^\s*>\s?(.*)$/;
const BULLET_RE = /^\s*[-*+•]\s+(.*)$/;
const NUMBER_RE = /^\s*(\d+)[.)]\s+(.*)$/;

function _isCitationBold(text) {
  return /^[\w./\\:-]+$/.test(text) && text.includes(':');
}

function renderBold(text, key) {
  if (_isCitationBold(text)) {
    return <strong key={key} className="msg-bold msg-cite">{text}</strong>;
  }
  return <strong key={key} className="msg-bold">{text}</strong>;
}

function renderInlineText(text, keyPrefix = '') {
  if (!text) return null;
  const nodes = [];
  let last = 0;
  let k = 0;
  INLINE_RE.lastIndex = 0;
  let m;
  while ((m = INLINE_RE.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    const key = keyPrefix + (k++);
    if (m[1] !== undefined) {
      nodes.push(<code key={key} className="msg-inline-code">{m[1]}</code>);
    } else if (m[2] !== undefined) {
      nodes.push(renderBold(m[2], key));
    } else if (m[3] !== undefined) {
      nodes.push(<em key={key} className="msg-em">{m[3]}</em>);
    } else if (m[4] !== undefined) {
      nodes.push(<del key={key} className="msg-del">{m[4]}</del>);
    } else if (m[5] !== undefined) {
      nodes.push(<a key={key} className="msg-link" href={m[6]} target="_blank" rel="noreferrer">{m[5]}</a>);
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes.length ? nodes : text;
}

function isTableSeparator(line) {
  const stripped = line.trim().replace(/^\|/, '').replace(/\|$/, '');
  if (!stripped.includes('-')) return false;
  const cells = stripped.split('|');
  let hasSep = false;
  for (const c of cells) {
    const t = c.trim();
    if (t === '') continue;
    if (!/^:?-{2,}:?$/.test(t)) return false;
    hasSep = true;
  }
  return hasSep;
}

function splitRow(line) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(c => c.trim());
}

// Pure, testable block parser — returns descriptors, no JSX.
function parseTextBlocks(text) {
  const lines = text.split('\n');
  const blocks = [];
  const isBlank = l => !l.trim();
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    if (isBlank(line)) { i++; continue; }

    // Pipe table: header row + ---|--- separator
    if (line.includes('|') && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      const header = splitRow(line);
      const align = splitRow(lines[i + 1]).map(c => {
        const t = c.trim();
        if (t.startsWith(':') && t.endsWith(':')) return 'center';
        if (t.endsWith(':')) return 'right';
        return 'left';
      });
      i += 2;
      const rows = [];
      while (i < lines.length && !isBlank(lines[i]) && lines[i].includes('|')) {
        rows.push(splitRow(lines[i]));
        i++;
      }
      blocks.push({ type: 'table', header, align, rows });
      continue;
    }

    if (HR_RE.test(line)) { blocks.push({ type: 'hr' }); i++; continue; }

    const h = line.match(HEADING_RE);
    if (h) { blocks.push({ type: 'heading', level: h[1].length, text: h[2] }); i++; continue; }

    if (QUOTE_RE.test(line)) {
      const buf = [];
      while (i < lines.length && QUOTE_RE.test(lines[i])) { buf.push(lines[i].match(QUOTE_RE)[1]); i++; }
      blocks.push({ type: 'quote', text: buf.join(' ') });
      continue;
    }

    if (BULLET_RE.test(line)) {
      const items = [];
      while (i < lines.length && BULLET_RE.test(lines[i])) { items.push(lines[i].match(BULLET_RE)[1]); i++; }
      blocks.push({ type: 'ul', items });
      continue;
    }

    if (NUMBER_RE.test(line)) {
      const items = [];
      while (i < lines.length && NUMBER_RE.test(lines[i])) {
        const mm = lines[i].match(NUMBER_RE);
        items.push({ n: parseInt(mm[1], 10), text: mm[2] });
        i++;
      }
      blocks.push({ type: 'ol', items });
      continue;
    }

    // Paragraph: consecutive lines that don't start a new block
    const buf = [line];
    i++;
    while (i < lines.length && !isBlank(lines[i])) {
      const l = lines[i];
      if (HEADING_RE.test(l) || HR_RE.test(l) || QUOTE_RE.test(l) || BULLET_RE.test(l) || NUMBER_RE.test(l)) break;
      if (i + 1 < lines.length && isTableSeparator(lines[i + 1])) break;
      buf.push(l);
      i++;
    }
    blocks.push({ type: 'p', text: buf.join('\n') });
  }

  return blocks;
}

function renderMarkdownBlock(text, blockKey) {
  const blocks = parseTextBlocks(text);
  return (
    <React.Fragment key={blockKey}>
      {blocks.map((b, i) => {
        switch (b.type) {
          case 'heading': {
            const level = Math.min(Math.max(b.level, 1), 4);
            return <div key={i} className={`msg-h msg-h${level}`}>{renderInlineText(b.text, `h${i}-`)}</div>;
          }
          case 'hr':
            return <div key={i} className="msg-hr" />;
          case 'quote':
            return <div key={i} className="msg-quote">{renderInlineText(b.text, `q${i}-`)}</div>;
          case 'ul':
            return (
              <div key={i} className="msg-list">
                {b.items.map((it, j) => (
                  <div key={j} className="msg-li">
                    <span className="msg-li-mark">•</span>
                    <span className="msg-li-body">{renderInlineText(it, `ul${i}-${j}-`)}</span>
                  </div>
                ))}
              </div>
            );
          case 'ol':
            return (
              <div key={i} className="msg-list">
                {b.items.map((it, j) => (
                  <div key={j} className="msg-li">
                    <span className="msg-li-num">{it.n}.</span>
                    <span className="msg-li-body">{renderInlineText(it.text, `ol${i}-${j}-`)}</span>
                  </div>
                ))}
              </div>
            );
          case 'table':
            return (
              <div key={i} className="msg-table-wrap">
                <table className="msg-table">
                  <thead>
                    <tr>
                      {b.header.map((c, j) => (
                        <th key={j} style={{ textAlign: b.align[j] || 'left' }}>{renderInlineText(c, `th${i}-${j}-`)}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {b.rows.map((r, j) => (
                      <tr key={j}>
                        {r.map((c, k) => (
                          <td key={k} style={{ textAlign: b.align[k] || 'left' }}>{renderInlineText(c, `td${i}-${j}-${k}-`)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          default:
            return (
              <p key={i} className="msg-p">
                {b.text.split('\n').map((l, j) => (
                  <React.Fragment key={j}>
                    {j > 0 && <br />}
                    {renderInlineText(l, `p${i}-${j}-`)}
                  </React.Fragment>
                ))}
              </p>
            );
        }
      })}
    </React.Fragment>
  );
}
