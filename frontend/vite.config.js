import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const DEV_PROXY_TARGET =
  process.env.VITE_API_PROXY_TARGET ||
  process.env.API_PROXY_TARGET ||
  "http://localhost:5000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    allowedHosts: ["localhost", ".ngrok-free.dev", ".ngrok.io", ".railway.app"],
    proxy: {
      "/api": {
        target: DEV_PROXY_TARGET,
        changeOrigin: true,
      },
    },
  },
});
