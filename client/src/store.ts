// Настройки клиента в localStorage: адрес сервера, токен, устройства, режим отладки.
const PREFIX = "stenograf.";

export const DEFAULT_SERVER_URL = "http://127.0.0.1:8765";

export function getSetting(key: string, fallback = ""): string {
  return localStorage.getItem(PREFIX + key) ?? fallback;
}

export function setSetting(key: string, value: string): void {
  localStorage.setItem(PREFIX + key, value);
}

export function getServerUrl(): string {
  return getSetting("serverUrl", DEFAULT_SERVER_URL).replace(/\/+$/, "");
}

/**
 * Токен доступа к серверу. Пустая строка — обычный случай: сервер без
 * заведённых людей ничего не спрашивает, и заставлять вводить токен там,
 * где он не нужен, значит ломать локальный запуск ради удалённого.
 */
export function getToken(): string {
  return getSetting("serverToken").trim();
}

export function isDebugMode(): boolean {
  return getSetting("debug") === "1";
}

export function platform(): string {
  return window.stenograf?.platform ?? "web";
}
