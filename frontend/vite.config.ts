import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      '/api': {
        // 127.0.0.1, not localhost: on Windows 'localhost' resolves to ::1 first,
        // and a WSL relay squats on [::1]:8000, so requests hang there instead of
        // reaching uvicorn (which only listens on 127.0.0.1).
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        // Match the client's long timeout so the proxy doesn't drop slow
        // LLM/embedding responses first.
        timeout: 300_000,
        proxyTimeout: 300_000,
      },
    },
  },
});
