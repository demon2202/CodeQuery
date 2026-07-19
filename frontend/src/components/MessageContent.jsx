/**
 * MessageContent — renders assistant message content with:
 * - Chart.js charts (```chart blocks with JSON)
 * - Mermaid diagrams (```mermaid blocks)
 * - Code blocks with language labels and copy button
 * - Bold, inline code, bullet lists, and basic formatting
 * - Clickable file:line citations
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
              <div className="msg-code-top">
                {part.lang && <span className="msg-code-lang">{part.lang}</span>}
                <button className="copy-btn" onClick={() => navigator.clipboard?.writeText(part.code)} title="Copy code">
                  <svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"/><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"/></svg>
                </button>
              </div>
              <pre><code>{part.code}</code></pre>
            </div>
          );
        }
        // Inline content with formatting
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
      // Try multiple JSON parsing strategies
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


/**
 * Try to parse chart JSON — handles common LLM mistakes:
 * 1. Valid JSON (ideal)
 * 2. JSON with trailing commas
 * 3. JSON with single quotes
 * 4. YAML-ish format (attempt to reconstruct)
 */
function tryParseChartJSON(code) {
  // Strategy 1: Direct JSON parse
  try {
    const spec = JSON.parse(code);
    if (spec.type && spec.labels && spec.datasets) return spec;
  } catch {}

  // Strategy 2: Remove trailing commas and try again
  try {
    const cleaned = code.replace(/,\s*([}\]])/g, '$1');
    const spec = JSON.parse(cleaned);
    if (spec.type && spec.labels && spec.datasets) return spec;
  } catch {}

  // Strategy 3: Replace single quotes with double quotes
  try {
    const cleaned = code.replace(/'/g, '"').replace(/,\s*([}\]])/g, '$1');
    const spec = JSON.parse(cleaned);
    if (spec.type && spec.labels && spec.datasets) return spec;
  } catch {}

  // Strategy 4: Try to find a JSON object anywhere in the text
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


function renderInline(text) {
  if (!text) return null;

  const segments = [];
  // Match: **bold**, `code`, - bullet items at line start, ### headings
  const regex = /(\*\*(.+?)\*\*|`([^`]+)`|^### (.+)$|^- (.+))/gm;

  let lastIdx = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIdx) {
      segments.push(renderTextWithNewlines(text.slice(lastIdx, match.index)));
    }

    if (match[2]) {
      // Bold — check if it looks like a file:line citation
      const boldText = match[2];
      if (/^[\w./\\:-]+$/.test(boldText) && boldText.includes(':')) {
        // File:line citation — render with special styling
        segments.push(<strong key={match.index} className="msg-bold msg-cite">{boldText}</strong>);
      } else {
        segments.push(<strong key={match.index} className="msg-bold">{boldText}</strong>);
      }
    } else if (match[3]) {
      segments.push(<code key={match.index} className="msg-inline-code">{match[3]}</code>);
    } else if (match[4]) {
      // Heading
      segments.push(<div key={match.index} className="msg-heading">{match[4]}</div>);
    } else if (match[5]) {
      // Bullet point
      segments.push(<div key={match.index} className="msg-bullet">• {match[5]}</div>);
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
