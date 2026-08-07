import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  // относительные пути — чтобы собранный dist открывался в Electron через file://
  base: "./",
  // host задан явно: без него vite слушает только IPv6, и открытый в браузере
  // http://localhost:5173 отваливается по ECONNREFUSED на машинах, где localhost
  // резолвится в 127.0.0.1. Раньше флаг приходилось дописывать в каждую команду.
  server: { host: "127.0.0.1", port: 5173, strictPort: true },
});
