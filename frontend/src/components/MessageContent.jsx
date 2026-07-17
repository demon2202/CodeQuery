/**
 * MessageContent — renders assistant message content with:
 * - Chart.js charts (```chart blocks with JSON)
 * - Mermaid diagrams (```mermaid blocks)
 * - Code blocks with language labels
 * - Bold, inline code, and basic formatting
 */

import React, { useMemo, lazy, Suspense } from 'react';

const MermaidDiagram = lazy(() => import('./MermaidDiagram'));
const ChartComponent = lazy(() => import('./ChartComponent'));

export default function MessageContent({ content }) {
  const parts = useMemo(() => parseContent(content), [content]);

  return (
    <>
      {parts.map((part, i) => {
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
          return (
            <div key={i} className="msg-code-block">
              {part.lang && <span className="msg-code-lang">{part.lang}</span>}
              <pre><code>{part.code}</code></pre>
            </div>
          );
        }
        // Inline content with basic formatting
        return <span key={i}>{renderInline(part.text)}</span>;
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

    if (lang === 'chart') {
      // Parse chart JSON — if invalid, fall back to showing it as code
      try {
        const spec = JSON.parse(code);
        parts.push({ type: 'chart', spec });
      } catch {
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


function renderInline(text) {
  if (!text) return null;

  const segments = [];
  const regex = /(\*\*(.+?)\*\*|`([^`]+)`)/g;
  let lastIdx = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIdx) {
      segments.push(renderTextWithNewlines(text.slice(lastIdx, match.index)));
    }

    if (match[2]) {
      segments.push(<strong key={match.index} className="msg-bold">{match[2]}</strong>);
    } else if (match[3]) {
      segments.push(<code key={match.index} className="msg-inline-code">{match[3]}</code>);
    }

    lastIdx = match.index + match[0].length;
  }

  if (lastIdx < text.length) {
    segments.push(renderTextWithNewlines(text.slice(lastIdx)));
  }

  return segments.length > 0 ? segments : text;
}


function renderTextWithNewlines(text) {
  const lines = text.split('\n');
  return lines.map((line, i) => (
    <React.Fragment key={i}>
      {i > 0 && <br />}
      {line}
    </React.Fragment>
  ));
}
