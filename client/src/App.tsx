import { useEffect, useState } from "react";
import { api } from "./api/rest";
import HistoryPage from "./pages/HistoryPage";
import LivePage from "./pages/LivePage";
import MeetingPage from "./pages/MeetingPage";
import SettingsPage from "./pages/SettingsPage";
import SpeakersPage from "./pages/SpeakersPage";
import { getServerUrl } from "./store";
import type { HealthDto } from "./types";

export type Page =
  | { name: "live" }
  | { name: "history" }
  | { name: "meeting"; id: number }
  | { name: "speakers" }
  | { name: "settings" };

const NAV: { key: Page["name"]; icon: string; label: string }[] = [
  { key: "live", icon: "🎙️", label: "Встреча" },
  { key: "history", icon: "🗂️", label: "История" },
  { key: "speakers", icon: "👥", label: "Спикеры" },
  { key: "settings", icon: "⚙️", label: "Настройки" },
];

export default function App() {
  const [page, setPage] = useState<Page>({ name: "live" });
  const [health, setHealth] = useState<HealthDto | null>(null);

  useEffect(() => {
    let alive = true;
    const poll = () =>
      api
        .health()
        .then((h) => alive && setHealth(h))
        .catch(() => alive && setHealth(null));
    poll();
    const timer = setInterval(poll, 7000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  const active = page.name === "meeting" ? "history" : page.name;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="logo">
          <div className="logo-mark">🎙️</div>
          <div>
            <div className="logo-name">Стенограф</div>
            <div className="logo-sub">протокол встреч</div>
          </div>
        </div>
        <nav className="nav">
          {NAV.map((item) => (
            <button
              key={item.key}
              className={`nav-item ${active === item.key ? "active" : ""}`}
              onClick={() => setPage({ name: item.key } as Page)}
            >
              <span className="icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="server-status" title={getServerUrl()}>
          <span className={`dot ${health ? "ok" : "err"}`} />
          <span className="status-text">
            {health ? `сервер на связи · whisper ${health.asr.model}` : "сервер недоступен"}
          </span>
        </div>
      </aside>
      <div className="page-host">
        {page.name === "live" && <LivePage navigate={setPage} health={health} />}
        {page.name === "history" && <HistoryPage navigate={setPage} />}
        {page.name === "meeting" && <MeetingPage id={page.id} navigate={setPage} />}
        {page.name === "speakers" && <SpeakersPage />}
        {page.name === "settings" && <SettingsPage onServerChange={() => setHealth(null)} />}
      </div>
    </div>
  );
}
