import { useEffect, useState } from "react";
import { api } from "./api/rest";
import IndexingToast from "./components/IndexingToast";
import { useAutoIndexing } from "./llm/indexing";
import HistoryPage from "./pages/HistoryPage";
import KnowledgePage from "./pages/KnowledgePage";
import LivePage from "./pages/LivePage";
import MeetingPage from "./pages/MeetingPage";
import SettingsPage from "./pages/SettingsPage";
import SpeakersPage from "./pages/SpeakersPage";
import { getServerUrl } from "./store";
import type { HealthDto } from "./types";

export type Page =
  | { name: "live" }
  | { name: "history" }
  // autosummarize — встречу только что закончили с галочкой «составить
  // протокол по завершении»: считает его приложение, и повод — этот переход
  | { name: "meeting"; id: number; autosummarize?: boolean }
  | { name: "speakers" }
  | { name: "knowledge" }
  | { name: "settings" };

const NAV: { key: Page["name"]; icon: string; label: string }[] = [
  { key: "live", icon: "🎙️", label: "Встреча" },
  { key: "history", icon: "🗂️", label: "История" },
  { key: "speakers", icon: "👥", label: "Спикеры" },
  { key: "knowledge", icon: "📚", label: "База знаний" },
  { key: "settings", icon: "⚙️", label: "Настройки" },
];

export default function App() {
  const [page, setPage] = useState<Page>({ name: "live" });
  const [health, setHealth] = useState<HealthDto | null>(null);
  const [livePhase, setLivePhase] = useState<string | null>(null);

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

  // Индексация сама досчитывает новое: после встречи и при запуске приложения.
  // Во время встречи — нет: Ollama считает подсказки, и очередь к ней одна
  useAutoIndexing(health !== null, livePhase !== null);

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
              onClick={() => {
                if (item.key !== "live" && livePhase === "live") {
                  if (!window.confirm("Встреча ещё идёт. Завершить её и перейти?")) return;
                }
                setPage({ name: item.key } as Page);
              }}
            >
              <span className="icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="server-status" title={getServerUrl()}>
          <span className={`dot ${health ? "ok" : "err"}`} />
          <span className="status-text">
            {!health
              ? "сервер недоступен"
              : health.asr
                ? `сервер на связи · ${health.asr.model}`
                : "сервер требует токен"}
          </span>
        </div>
      </aside>
      <div className="page-host">
        {page.name === "live" && <LivePage navigate={setPage} health={health} onPhaseChange={setLivePhase} />}
        {page.name === "history" && <HistoryPage navigate={setPage} />}
        {page.name === "meeting" && (
          <MeetingPage id={page.id} autosummarize={page.autosummarize} navigate={setPage} />
        )}
        {page.name === "speakers" && <SpeakersPage />}
        {page.name === "knowledge" && <KnowledgePage />}
        {page.name === "settings" && <SettingsPage onServerChange={() => setHealth(null)} />}
      </div>
      {/* Рядом со страницами, а не внутри: окошко про индексацию — общее для
          всего приложения, и страница ему не хозяин */}
      <IndexingToast наЭкранеБазы={page.name === "knowledge"} />
    </div>
  );
}
