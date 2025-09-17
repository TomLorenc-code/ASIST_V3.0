// vite.config.js
import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  server: {
    proxy: {
      // Proxy API requests
      '/api': {
        target: 'http://localhost:3000', // Your Flask backend
        changeOrigin: true,
      },
      // Proxy Auth0 related Flask routes
      '/login': {
        target: 'http://localhost:3000', // Your Flask backend
        changeOrigin: true,
      },
      '/callback': {
        target: 'http://localhost:3000', // Your Flask backend
        changeOrigin: true,
      },
      '/logout': {
        target: 'http://localhost:3000', // Your Flask backend
        changeOrigin: true,
      }
    }
  }
})
