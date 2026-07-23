import React, { useState, useEffect, useCallback } from 'react';
import { API_BASE } from '../config';

const LANG_ICONS = {
  python: '🐍', javascript: 'JS', typescript: 'TS', go: 'Go',
  rust: 'Rs', java: '☕', ruby: '💎', html: '◁', css: '◇',
  json: '{}', yaml: '⚙', markdown: '📝', bash: '⚙', sql: '⚙', text: '📄',
};

export default function FileTree({ repoUrl, onFileClick, onClose }) {
  const [tree, setTree] = useState(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(new Set());
  const [selectedPath, setSelectedPath] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const fetchTree = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/repos/tree?repo_url=${encodeURIComponent(repoUrl)}`);
        if (!cancelled && res.ok) {
          const data = await res.json();
          setTree(data);
          // Auto-expand first level using path for uniqueness
          const firstLevel = new Set();
          (data.children || []).forEach(c => {
            if (c.type === 'dir') firstLevel.add(c.path || c.name);
          });
          setExpanded(firstLevel);
        }
      } catch {}
      if (!cancelled) setLoading(false);
    };
    fetchTree();
    return () => { cancelled = true; };
  }, [repoUrl]);

  const toggleDir = useCallback((path) => {
    setExpanded(prev => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }, []);

  const handleClick = useCallback((node) => {
    if (node.type === 'dir') {
      // Use node.path (or node.name as fallback) to avoid same-name dir collision
      toggleDir(node.path || node.name);
    } else {
      setSelectedPath(node.path);
      onFileClick?.(node);
    }
  }, [toggleDir, onFileClick]);

  return (
    <div className="file-tree">
      <div className="ft-header">
        <span className="ft-title">Files</span>
        <button className="ft-close" onClick={onClose} title="Close file tree">✕</button>
      </div>
      <div className="ft-body">
        {loading && <div className="ft-loading">Loading…</div>}
        {tree && <TreeNode node={tree} expanded={expanded} onClick={handleClick} selectedPath={selectedPath} depth={0} />}
      </div>
    </div>
  );
}

function TreeNode({ node, expanded, onClick, selectedPath, depth }) {
  const isDir = node.type === 'dir';
  const isOpen = expanded.has(node.path || node.name);
  const isSelected = node.path === selectedPath;

  return (
    <div className="ft-node">
      <div
        className={`ft-row ${isSelected ? 'ft-selected' : ''}`}
        style={{ paddingLeft: `${depth * 14 + 10}px` }}
        onClick={() => onClick(node)}
      >
        {isDir ? (
          <span className="ft-arrow">{isOpen ? '▾' : '▸'}</span>
        ) : (
          <span className="ft-arrow ft-arrow-file">·</span>
        )}
        <span className="ft-icon">
          {isDir ? (isOpen ? '📂' : '📁') : (LANG_ICONS[node.language] || '📄')}
        </span>
        <span className="ft-name" title={node.path || node.name}>{node.name}</span>
      </div>
      {isDir && isOpen && (node.children || []).map((child, i) => (
        <TreeNode
          key={child.path || `${child.name}-${i}`}
          node={child}
          expanded={expanded}
          onClick={onClick}
          selectedPath={selectedPath}
          depth={depth + 1}
        />
      ))}
    </div>
  );
}