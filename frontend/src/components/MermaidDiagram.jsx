/**
 * MermaidDiagram — renders refined, scalable Mermaid architecture and sequence diagrams.
 * Features:
 * - Anti-shrunk scalable SVG rendering with proper node padding
 * - Inline zoom & pan controls directly in the toolbar
 * - Fullscreen modal inspection with high-contrast theme
 * - Syntax auto-sanitization for parentheses and unquoted labels
 * - One-click SVG and code export
 */

import React, { useEffect, useRef, useState } from 'react';

function sanitizeMermaid(raw) {
  if (!raw) return '';
  let code = raw.trim();

  // Strip markdown fences if present
  code = code.replace(/^```mermaid\s*/i, '').replace(/```\s*$/, '').trim();

  // Auto-fix unquoted parentheses inside node bracket definitions like A[main(args)] -> A["main(args)"]
  code = code.replace(/(\b[A-Za-z0-9_]+)\[([^"\]\n]*\([^"\]\n]*\)[^"\]\n]*)\]/g, '$1["$2"]');

  // Fix unquoted parentheses inside rounded node definitions like A(main(args)) -> A["main(args)"]
  code = code.replace(/(\b[A-Za-z0-9_]+)\(([^"\)\n]*\([^"\)\n]*\)[^"\)\n]*)\)/g, '$1["$2"]');

  // Fix broken arrows
  code = code.replace(/---\s+>/g, '-->').replace(/--\s+>/g, '-->');

  return code;
}

function cleanSvgOutput(svgStr) {
  if (!svgStr) return '';
  // Remove max-width constraint added by Mermaid so SVG scales generously
  return svgStr.replace(/style="[^"]*max-width:[^;"]*;?[^"]*"/gi, 'style=""');
}

export default function MermaidDiagram({ code }) {
  const containerRef = useRef(null);
  const modalContainerRef = useRef(null);
  const [svgContent, setSvgContent] = useState('');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [zoom, setZoom] = useState(1);

  const cleanCode = sanitizeMermaid(code);

  useEffect(() => {
    let cancelled = false;

    const renderDiagram = async () => {
      try {
        const mermaid = (await import('mermaid')).default;
        mermaid.initialize({
          startOnLoad: false,
          theme: 'base',
          themeVariables: {
            darkMode: true,
            background: '#0d1117',
            mainBkg: '#161b22',
            primaryColor: '#161b22',
            primaryTextColor: '#f0f6fc',
            primaryBorderColor: '#30363d',
            lineColor: '#58a6ff',
            secondaryColor: '#21262d',
            tertiaryColor: '#161b22',
            nodeBorder: '#30363d',
            clusterBkg: '#131822',
            clusterBorder: '#30363d',
            defaultLinkColor: '#58a6ff',
            titleColor: '#f0f6fc',
            edgeLabelBackground: '#0d1117',
            actorBkg: '#161b22',
            actorBorder: '#30363d',
            actorTextColor: '#f0f6fc',
            actorLineColor: '#58a6ff',
            signalColor: '#58a6ff',
            signalTextColor: '#f0f6fc',
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
            fontSize: '14px',
          },
          flowchart: {
            useMaxWidth: false,
            htmlLabels: true,
            curve: 'basis',
            nodeSpacing: 45,
            rankSpacing: 50,
            padding: 20,
          },
          sequence: {
            useMaxWidth: false,
            showSequenceNumbers: true,
            actorFontSize: 14,
            messageFontSize: 13,
          },
        });

        const id = `mermaid-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        const { svg } = await mermaid.render(id, cleanCode);

        if (!cancelled) {
          setSvgContent(cleanSvgOutput(svg));
          setLoading(false);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message || 'Diagram rendering failed');
          setLoading(false);
        }
      }
    };

    renderDiagram();
    return () => { cancelled = true; };
  }, [cleanCode]);

  useEffect(() => {
    if (containerRef.current && svgContent) {
      containerRef.current.innerHTML = svgContent;
    }
    if (modalContainerRef.current && svgContent) {
      modalContainerRef.current.innerHTML = svgContent;
    }
  }, [svgContent, isFullscreen, zoom]);

  const handleCopy = () => {
    navigator.clipboard?.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadSvg = () => {
    if (!svgContent) return;
    const blob = new Blob([svgContent], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `architecture-diagram-${Date.now()}.svg`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (error) {
    return (
      <div className="mermaid-error">
        <div className="mermaid-error-header">
          <span className="error-title">Diagram Syntax Notice</span>
          <button className="copy-btn" onClick={handleCopy}>
            {copied ? 'Copied' : 'Copy Source'}
          </button>
        </div>
        <pre>{cleanCode || code}</pre>
      </div>
    );
  }

  return (
    <>
      <div className="mermaid-wrapper">
        <div className="mermaid-toolbar">
          <div className="toolbar-left">
            <span className="diagram-badge">Architecture Model</span>
          </div>

          <div className="toolbar-right">
            <div className="zoom-controls">
              <button className="tb-btn zoom-btn-inline" onClick={() => setZoom(z => Math.max(0.6, z - 0.15))} title="Zoom Out">-</button>
              <span className="zoom-text">{Math.round(zoom * 100)}%</span>
              <button className="tb-btn zoom-btn-inline" onClick={() => setZoom(z => Math.min(2.5, z + 0.15))} title="Zoom In">+</button>
              <button className="tb-btn" onClick={() => setZoom(1)} title="Reset Scale">Reset</button>
            </div>

            <button className="tb-btn" onClick={handleCopy} title="Copy Mermaid code">
              {copied ? 'Copied' : 'Copy'}
            </button>
            <button className="tb-btn" onClick={handleDownloadSvg} title="Export SVG">
              SVG
            </button>
            <button className="tb-btn tb-btn-expand" onClick={() => { setIsFullscreen(true); setZoom(1.1); }} title="Fullscreen View">
              Expand ↗
            </button>
          </div>
        </div>

        <div className="mermaid-container">
          {loading && <div className="mermaid-loading">Rendering technical diagram…</div>}
          <div
            ref={containerRef}
            className="mermaid-output"
            style={{ transform: `scale(${zoom})`, transformOrigin: 'top center', transition: 'transform 0.15s ease-out' }}
          />
        </div>
      </div>

      {/* Fullscreen Interactive Modal */}
      {isFullscreen && (
        <div className="mermaid-modal-overlay" onClick={() => setIsFullscreen(false)}>
          <div className="mermaid-modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title">
                <span className="diagram-badge">Expanded Architecture Model</span>
              </div>
              <div className="modal-controls">
                <button className="zoom-btn" onClick={() => setZoom(z => Math.max(0.5, z - 0.2))}>-</button>
                <span className="zoom-label">{Math.round(zoom * 100)}%</span>
                <button className="zoom-btn" onClick={() => setZoom(z => Math.min(3.5, z + 0.2))}>+</button>
                <button className="zoom-btn reset" onClick={() => setZoom(1)}>Reset</button>
                <button className="modal-close-btn" onClick={() => setIsFullscreen(false)}>Close ✕</button>
              </div>
            </div>
            <div className="modal-body">
              <div
                ref={modalContainerRef}
                className="mermaid-modal-output"
                style={{ transform: `scale(${zoom})`, transformOrigin: 'top center' }}
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
