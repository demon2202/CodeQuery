/**
 * MermaidDiagram — renders Mermaid diagram code as SVG.
 * Lazy-loaded. Catches errors gracefully.
 */

import React, { useEffect, useRef, useState } from 'react';

export default function MermaidDiagram({ code }) {
  const containerRef = useRef(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    const renderDiagram = async () => {
      try {
        const mermaid = (await import('mermaid')).default;
        mermaid.initialize({
          startOnLoad: false,
          theme: 'dark',
          themeVariables: {
            background: '#0a0a0a',
            primaryColor: '#1c1c1c',
            primaryTextColor: '#e5e5e5',
            primaryBorderColor: '#2a2a2a',
            lineColor: '#555555',
            secondaryColor: '#151515',
            tertiaryColor: '#111111',
            fontFamily: '-apple-system, BlinkMacSystemFont, Segoe UI, system-ui, sans-serif',
            fontSize: '13px',
          },
        });

        const id = `mermaid-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        const { svg } = await mermaid.render(id, code.trim());

        if (!cancelled && containerRef.current) {
          containerRef.current.innerHTML = svg;
          setLoading(false);
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
  }, [code]);

  if (error) {
    return (
      <div className="mermaid-error">
        <span>⚠ Diagram error</span>
        <pre>{code}</pre>
      </div>
    );
  }

  return (
    <div className="mermaid-container">
      {loading && <div className="mermaid-loading">Rendering diagram…</div>}
      <div ref={containerRef} className="mermaid-output" />
    </div>
  );
}
