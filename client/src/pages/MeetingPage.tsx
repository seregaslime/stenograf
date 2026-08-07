import { marked } from "marked";
import { useEffect, useMemo, useState } from "react";
import type { Page } from "../App";
import { api } from "../api/rest";
import Transcript from "../components/Transcript";
import { isDebugMode } from "../store";
import type { MeetingDetail } from "../types";

export default function MeetingPage({
  id,
  navigate,
}: {
  id: number;
  navigate: (page: Page) => void;
}) {
  const [meeting, setMeeting] = useState<MeetingDetail | null>(null);
  const [error, setError] = useState("");

  const load = () =>
    api
      .meeting(id)
      .then((detail) => {
        setMeeting(detail);
        setError("");
      })
      .catch((exc: Error) => setError(exc.message));

  useEffect(() => {
    void load();
  }, [id]);

  // Пока сервер составляет резюме — опрашиваем
  useEffect(() => {
    if (meeting?.status !== "summarizing") return;
    const timer = setInterval(load, 4000);
    return () => clearInterval(timer);
  }, [meeting?.status]);

  const summaryHtml = useMemo(
    () => (meeting?.summary ? (marked.parse(meeting.summary) as string) : ""),
    [meeting?.summary],
  );

  async function resummarize() {
    try {
      await api.summarize(id);
      await load();
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  const date = meeting?.started_at
    ? new Date(meeting.started_at).toLocaleString("ru-RU", {
        day: "2-digit",
        month: "long",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "";

  return (
    <div className="content">
      <button className="back-link" onClick={() => navigate({ name: "history" })}>
        ← К истории встреч
      </button>
      {error && <div className="banner error">{error}</div>}
      {!meeting ? (
        <div className="empty">
          <span className="spinner" /> Загрузка…
        </div>
      ) : (
        <>
          <h1>{meeting.title}</h1>
          <p className="page-sub">{date}</p>
          <div className="toolbar">
            <a className="btn small" href={api.exportUrl(id, "md")}>
              ⬇ Экспорт .md
            </a>
            <a className="btn small" href={api.exportUrl(id, "txt")}>
              ⬇ Экспорт .txt
            </a>
            {meeting.status !== "summarizing" && (
              <button className="btn small" onClick={resummarize}>
                ↻ {meeting.summary ? "Пересоздать резюме" : "Создать резюме"}
              </button>
            )}
          </div>
          <div className="meeting-layout">
            <div>
              <h2 style={{ marginBottom: 12 }}>Транскрипт</h2>
              {meeting.segments.length === 0 ? (
                <div className="empty">Распознанной речи нет</div>
              ) : (
                <Transcript segments={meeting.segments} debug={isDebugMode()} />
              )}
            </div>
            <div className="card summary-panel">
              <h2 style={{ marginBottom: 10 }}>Итоги встречи</h2>
              {meeting.status === "summarizing" && (
                <div className="banner info">
                  <span className="spinner" />{" "}
                  {meeting.summary_progress ? (
                    <>
                      Встреча длинная — модель разбирает её по фрагментам, шаг{" "}
                      {meeting.summary_progress[0]} из {meeting.summary_progress[1]}. Между
                      запросами выдерживается минута, чтобы уложиться в лимит API.
                    </>
                  ) : (
                    <>Модель составляет протокол — обычно это занимает до пары минут…</>
                  )}
                </div>
              )}
              {meeting.summary_error && (
                <div className="banner warn">{meeting.summary_error}</div>
              )}
              {meeting.summary ? (
                <>
                  <div
                    className="summary-md"
                    dangerouslySetInnerHTML={{ __html: summaryHtml }}
                  />
                  {meeting.summary_model && (
                    <p style={{ color: "var(--muted)", fontSize: 11.5, marginTop: 12 }}>
                      Модель: {meeting.summary_model} (локально)
                    </p>
                  )}
                </>
              ) : (
                meeting.status === "done" &&
                !meeting.summary_error && (
                  <p style={{ color: "var(--muted)" }}>Резюме ещё не создано.</p>
                )
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
