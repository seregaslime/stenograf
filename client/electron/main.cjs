const { app, BrowserWindow, ipcMain, shell, systemPreferences } = require("electron");
const { initMain } = require("electron-audio-loopback");
const path = require("path");

// Захват системного звука без виртуальных драйверов (BlackHole и т.п.):
// библиотека включает Chromium-флаги (ScreenCaptureKit на macOS 13+, WASAPI на
// Windows, PulseAudio на Linux) и по IPC-событию enable-loopback-audio ставит
// обработчик getDisplayMedia с audio: 'loopback'.
// Вызывается ДО app.whenReady() — флаги командной строки позже не применяются.
initMain();

// Статус разрешения «Запись экрана и звука системы» (macOS). Программно его
// запросить нельзя — можно только открыть нужный раздел Системных настроек.
ipcMain.handle("get-screen-permission", () =>
  process.platform === "darwin" ? systemPreferences.getMediaAccessStatus("screen") : "granted",
);
ipcMain.handle("open-screen-settings", () =>
  shell.openExternal(
    "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
  ),
);

function createWindow() {
  const win = new BrowserWindow({
    width: 1320,
    height: 860,
    minWidth: 980,
    minHeight: 640,
    title: "Стенограф",
    backgroundColor: "#0e1116",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
    },
  });

  const devUrl = process.env.ELECTRON_START_URL;
  if (devUrl) {
    win.loadURL(devUrl);
    win.webContents.openDevTools({ mode: "detach" });
  } else {
    win.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  }
}

app.whenReady().then(async () => {
  if (process.platform === "darwin") {
    // macOS спрашивает доступ к микрофону один раз
    await systemPreferences.askForMediaAccess("microphone").catch(() => {});
  }

  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
