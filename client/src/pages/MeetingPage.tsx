import { marked } from "marked";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Page } from "../App";
import { api } from "../api/rest";
import Transcript from "../components/Transcript";
import { LlmRouter } from "../llm/router";
import { loadLlmSettings, llmReady } from "../llm/settings";
import { generateSummary } from "../llm/summary";
import { isDebugMode } from "../store";
import type { MeetingDetail } from "../types";

export default function MeetingPage({
  id,
  navigate,
  autosummarize = false,
}: {
  id: number;
  navigate: (page: Page) => void;
  /** Встречу только что закончили с галочкой «составить протокол». */
  autosummarize?: boolean;
}) {
  const [meeting, setMeeting] = useState<MeetingDetail | null>(null);
  const [error, setError] = useState("");
  // Протокол теперь считает приложение, а не сервер: прогресс по фрагментам
  // приходит прямо отсюда, а не опросом состояния встречи.
  const [progress, setProgress] = useState<[number, number] | null>(null);

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

  // Протокол сразу после встречи. Ровно один раз: перезаход на страницу или
  // обновление данных не должны запускать модель заново — она стоит минут.
  const автозапуск = useRef(false);
  useEffect(() => {
    if (!autosummarize || автозапуск.current || !meeting) return;
    if (meeting.summary || meeting.status !== "done") return;
    автозапуск.current = true;
    void resummarize();
  }, [autosummarize, meeting]);

  // Встречу мог оставить в «составляется» прошлый запуск: сервер больше ничего
  // не считает, и сама она из этого состояния не выйдет — опрос бы висел вечно.
  useEffect(() => {
    if (meeting?.status !== "summarizing") return;
    const timer = setInterval(load, 4000);
    return () => clearInterval(timer);
  }, [meeting?.status]);

  const summaryHtml = useMemo(
    () => (meeting?.summary ? (marked.parse(meeting.summary) as string) : ""),
    [meeting?.summary],
  );

  /**
   * Составляет протокол здесь, в приложении, и отдаёт серверу готовый текст.
   *
   * Причину неудачи отправляем туда же: иначе встреча осталась бы без объяснения,
   * а раньше его писал сервер — потому что считал он же.
   */
  async function resummarize() {
    const settings = loadLlmSettings();
    if (!llmReady(settings)) {
      setError("Модель не настроена: укажите её в настройках приложения.");
      return;
    }
    if (!meeting) return;

    setError("");
    setProgress([0, 1]);
    try {
      const текст = await generateSummary(
        new LlmRouter(settings),
        {
          segments: meeting.segments,
          title: meeting.title,
          date,
          mode: meeting.meeting_mode,
        },
        (шаг, всего) => setProgress([шаг, всего]),
      );
      await api.saveSummary(id, {
        text: текст,
        model: new LlmRouter(settings).modelFor("summary"),
      });
    } catch (exc) {
      const причина = (exc as Error).message;
      setError(причина);
      // Молча проглотить нельзя: встреча должна показывать, почему протокола
      // нет, а не выглядеть так, будто его и не просили.
      await api.saveSummary(id, { error: причина }).catch(() => {});
    } finally {
      setProgress(null);
      await load();
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
              <button className="btn small" onClick={resummarize} disabled={progress !== null}>
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
              {progress && (
                <div className="banner info">
                  <span className="spinner" />{" "}
                  {progress[1] > 1 ? (
                    <>
                      Встреча длинная — модель разбирает её по фрагментам, шаг {progress[0]} из{" "}
                      {progress[1]}. Не закрывайте приложение: протокол считается здесь.
                    </>
                  ) : (
                    <>Модель составляет протокол — обычно это занимает до пары минут…</>
                  )}
                </div>
              )}
              {!progress && meeting.status === "summarizing" && (
                <div className="banner warn">
                  Встреча осталась в состоянии «составляется» с прошлого запуска. Протокол
                  теперь считает приложение, поэтому нажмите «Создать резюме» ещё раз.
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
                      Модель: {meeting.summary_model}
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
