const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("stenograf", {
  platform: process.platform, // "darwin" | "win32" | "linux"
});
