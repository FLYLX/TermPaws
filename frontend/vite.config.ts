import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import { tanstackRouter } from "@tanstack/router-plugin/vite"
import react from "@vitejs/plugin-react-swc"
import { defineConfig } from "vite"

// https://vitejs.dev/config/
export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  plugins: [
    tanstackRouter({
      target: "react",
      autoCodeSplitting: true,
    }),
    react(),
    tailwindcss(),
  ],
  server: {
    // 开发服务器代理：前端同源调 /api，dev 时转发到本地后端容器。
    // 仅 dev 生效，生产构建不受影响（生产由后端同源服务静态文件）。
    proxy: {
      "/api": {
        target: process.env.VITE_DEV_API_TARGET || "http://localhost:28888",
        changeOrigin: true,
      },
    },
  },
})
