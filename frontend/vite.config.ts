import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/SarkerlabLLM/ner_llm_tool/',
  server: {
    port: 5173,
    allowedHosts: ['sesame.bmi.emory.edu'],
    proxy: {
      '/SarkerlabLLM/ner_llm_tool/api': {
        target: 'http://localhost:5002',
        rewrite: (path) => path.replace(/^\/SarkerlabLLM\/ner_llm_tool/, ''),
      },
      '/api': 'http://localhost:5002',
    },
  },
});
