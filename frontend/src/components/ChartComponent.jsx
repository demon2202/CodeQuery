/**
 * ChartComponent — renders Chart.js charts from JSON spec.
 * Lazy-loaded so Chart.js (~60KB) is only fetched when a chart appears.
 *
 * The LLM outputs ```chart blocks with JSON like:
 * {
 *   "type": "pie",
 *   "title": "File types",
 *   "labels": ["Python", "JavaScript", "HTML"],
 *   "datasets": [{"data": [45, 30, 25]}]
 * }
 *
 * Supported types: pie, doughnut, bar, line, radar, polarArea
 */

import React, { useEffect, useRef, useState } from 'react';

// Chart.js colors that work on dark backgrounds
const PALETTE = [
  '#34d399', '#60a5fa', '#fbbf24', '#f87171', '#a78bfa',
  '#fb923c', '#38bdf8', '#4ade80', '#e879f9', '#f472b6',
];

export default function ChartComponent({ spec }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    const renderChart = async () => {
      try {
        // Dynamic import — Chart.js is only loaded when a chart actually appears
        const { Chart, registerables } = await import('chart.js');
        Chart.register(...registerables);

        if (cancelled || !canvasRef.current) return;

        // Destroy previous chart if it exists
        if (chartRef.current) {
          chartRef.current.destroy();
          chartRef.current = null;
        }

        const ctx = canvasRef.current.getContext('2d');
        const chartType = spec.type || 'bar';
        const isPie = ['pie', 'doughnut', 'polarArea'].includes(chartType);

        // Build datasets with colors
        const datasets = (spec.datasets || []).map((ds, i) => {
          const base = {
            label: ds.label || '',
            data: ds.data || [],
          };

          if (isPie) {
            base.backgroundColor = PALETTE.slice(0, (ds.data || []).length);
            base.borderColor = '#0d0d0d';
            base.borderWidth = 2;
          } else if (chartType === 'line') {
            base.borderColor = PALETTE[i % PALETTE.length];
            base.backgroundColor = PALETTE[i % PALETTE.length] + '20';
            base.fill = true;
            base.tension = 0.3;
            base.pointBackgroundColor = PALETTE[i % PALETTE.length];
            base.pointRadius = 4;
          } else if (chartType === 'radar') {
            base.borderColor = PALETTE[i % PALETTE.length];
            base.backgroundColor = PALETTE[i % PALETTE.length] + '30';
            base.pointBackgroundColor = PALETTE[i % PALETTE.length];
          } else {
            // bar
            base.backgroundColor = ds.data?.map((_, j) => PALETTE[j % PALETTE.length] + 'cc') || [];
            base.borderColor = ds.data?.map((_, j) => PALETTE[j % PALETTE.length]) || [];
            base.borderWidth = 1;
            base.borderRadius = 4;
          }

          return base;
        });

        chartRef.current = new Chart(ctx, {
          type: chartType,
          data: {
            labels: spec.labels || [],
            datasets,
          },
          options: {
            responsive: true,
            maintainAspectRatio: true,
            animation: { duration: 600 },
            plugins: {
              title: {
                display: !!spec.title,
                text: spec.title || '',
                color: '#e5e5e5',
                font: { size: 14, weight: '600', family: '-apple-system, BlinkMacSystemFont, Segoe UI, system-ui, sans-serif' },
                padding: { bottom: 16 },
              },
              legend: {
                display: datasets.length > 1 || isPie,
                position: isPie ? 'right' : 'top',
                labels: {
                  color: '#999999',
                  font: { size: 11 },
                  padding: 12,
                  usePointStyle: true,
                  pointStyleWidth: 8,
                },
              },
              tooltip: {
                backgroundColor: '#1c1c1c',
                titleColor: '#e5e5e5',
                bodyColor: '#999999',
                borderColor: '#2a2a2a',
                borderWidth: 1,
                padding: 10,
                cornerRadius: 6,
              },
            },
            scales: isPie ? {} : {
              x: {
                ticks: { color: '#666666', font: { size: 11 } },
                grid: { color: '#1f1f1f' },
                border: { color: '#2a2a2a' },
              },
              y: {
                ticks: { color: '#666666', font: { size: 11 } },
                grid: { color: '#1f1f1f' },
                border: { color: '#2a2a2a' },
                beginAtZero: true,
              },
            },
          },
        });
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    };

    renderChart();

    return () => {
      cancelled = true;
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, [spec]);

  if (error) {
    return (
      <div className="chart-error">
        <span>⚠ Chart error: {error}</span>
        <pre>{JSON.stringify(spec, null, 2)}</pre>
      </div>
    );
  }

  return (
    <div className="chart-container">
      <canvas ref={canvasRef} />
    </div>
  );
}
