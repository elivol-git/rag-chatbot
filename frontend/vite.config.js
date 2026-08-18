import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server proxies /api to Flask so the frontend uses one path in both
// dev and production (where Flask serves the built bundle itself).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
    },
  },
})
