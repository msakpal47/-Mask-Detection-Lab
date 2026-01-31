import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/health': 'http://localhost:8000',
      '/predict': 'http://localhost:8000',
      '/predict-image': 'http://localhost:8000',
      '/images': 'http://localhost:8000',
      '/file': 'http://localhost:8000'
    }
  }
})
