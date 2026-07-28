import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
      // Lets the browser import the same indicator math the functions use.
      '@shared': path.resolve(__dirname, 'shared'),
    },
  },
  server: {
    // 5173 sits inside a Windows/Hyper-V dynamic port-exclusion range on
    // some machines, which makes Vite fail with EACCES instead of a normal
    // "port in use" error. 5273 is outside all reserved ranges.
    port: 5273,
    // When running plain `vite` (not `netlify dev`), proxy /api to the
    // Netlify dev server so the frontend works either way.
    proxy: {
      '/api': {
        target: 'http://localhost:8888',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          charts: ['recharts'],
          supabase: ['@supabase/supabase-js'],
        },
      },
    },
  },
});
