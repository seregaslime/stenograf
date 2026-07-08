const { app, BrowserWindow, session, systemPreferences, desktopCapturer } = require("electron");
const path = require("path");

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

  if (process.platform === "win32") {
    // Windows: getDisplayMedia({audio:true}) отдаёт системный звук (WASAPI loopback).
    // На macOS loopback в Chromium нет — там системный звук берётся через BlackHole.
    session.defaultSession.setDisplayMediaRequestHandler((_request, callback) => {
      desktopCapturer.getSources({ types: ["screen"] }).then((sources) => {
        callback({ video: sources[0], audio: "loopback" });
      });
    });
  }

  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
