import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `make ui-dev`: hot reload here, API calls proxied to the FastAPI server (DESIGN §9).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
  // mermaid loads its diagram types on demand; the large chunks are never sent unused
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 2000 },
});
