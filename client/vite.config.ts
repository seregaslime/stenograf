import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  // относительные пути — чтобы собранный dist открывался в Electron через file://
  base: "./",
  server: { port: 5173, strictPort: true },
});
