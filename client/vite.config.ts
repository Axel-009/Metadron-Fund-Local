import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  build: {
    outDir: "../deploy/terminal",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: undefined,
      },
    },
  },
  css: {
    postcss: {
      plugins: [],
    },
  },
});
