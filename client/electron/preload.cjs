const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("stenograf", {
  platform: process.platform, // "darwin" | "win32" | "linux"
  // Автозахват системного звука (см. electron-audio-loopback в main.cjs)
  enableLoopbackAudio: () => ipcRenderer.invoke("enable-loopback-audio"),
  disableLoopbackAudio: () => ipcRenderer.invoke("disable-loopback-audio"),
  getScreenPermission: () => ipcRenderer.invoke("get-screen-permission"),
  openScreenSettings: () => ipcRenderer.invoke("open-screen-settings"),
});
