import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy API calls to the FastAPI backend during development
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    // Code splitting — keep main bundle small, lazy-load heavy deps
    rollupOptions: {
      output: {
        manualChunks: {
          'mermaid': ['mermaid'],
          'three': ['three', 'postprocessing'],
          'chartjs': ['chart.js'],
        },
      },
    },
  },
});
