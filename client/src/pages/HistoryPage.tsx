import { useEffect, useRef, useState, type MouseEvent } from "react";
import type { Page } from "../App";
import { api } from "../api/rest";
import { formatTime } from "../components/Transcript";
import type { SearchHit, MeetingListItem } from "../types";

const STATUS_LABEL: Record<MeetingListItem["status"], string> = {
  live: "идёт",
  summarizing: "резюме…",
  done: "готово",
};

function formatDate(iso: string | null): string {
  if (!iso) return "";
  return new Date(iso).toLocaleString("ru-RU", {
    day: "2-digit",
    month: "long",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function duration(meeting: MeetingListItem): string {
  if (!meeting.started_at || !meeting.ended_at) return "";
  const seconds = (new Date(meeting.ended_at).getTime() - new Date(meeting.started_at).getTime()) / 1000;
  return formatTime(seconds);
}

function mmss(seconds: number): string {
  const м = Math.floor(seconds / 60);
  return `${м}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

export default function HistoryPage({ navigate }: { navigate: (page: Page) => void }) {
  const [meetings, setMeetings] = useState<MeetingListItem[] | null>(null);
  const [error, setError] = useState("");
  // Поиск по смыслу: вопрос своими словами, в ответ — цитаты из прошлых встреч
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState("");
  const [answer, setAnswer] = useState("");
  const [answering, setAnswering] = useState(false);
  // Вопрос, ответ на который сейчас ждём. Модель думает секунды (локально —
  // десятки), и без этой сверки ответ на прошлый вопрос успевает вернуться
  // последним и лечь под цитаты к новому: ровно то «уверенно не про то»,
  // ради защиты от которого цитаты и показываются.
  const ждём = useRef("");

  const load = () =>
    api
      .meetings()
      .then((list) => {
        setMeetings(list);
        setError("");
      })
      .catch((exc: Error) => setError(exc.message));

  useEffect(() => {
    void load();
  }, []);

  async function find() {
    const текст = query.trim();
    if (!текст) {
      setHits(null);
      return;
    }
    setSearching(true);
    setSearchError("");
    setAnswer("");
    try {
      // Первый запрос после новой встречи заодно её индексирует — он дольше
      const { results } = await api.search(текст);
      setHits(results);
      if (results.length) void ask(текст);
    } catch (exc) {
      setSearchError((exc as Error).message);
      setHits(null);
    } finally {
      setSearching(false);
    }
  }

  /** Ответ модели вторым шагом: цитаты уже на экране, их читают, пока она думает. */
  async function ask(текст: string) {
    ждём.current = текст;
    setAnswering(true);
    try {
      const итог = await api.searchAnswer(текст);
      if (ждём.current !== текст) return;  // пока думали, спросили другое
      setAnswer(итог.answer);
      setHits(итог.results);
    } catch (exc) {
      if (ждём.current !== текст) return;
      // Цитаты уже показаны и сами по себе полезны — ошибку ответа показываем
      // рядом с ними, а не вместо них.
      setSearchError((exc as Error).message);
    } finally {
      if (ждём.current === текст) setAnswering(false);
    }
  }

  async function remove(meeting: MeetingListItem, event: MouseEvent) {
    event.stopPropagation();
    if (!confirm(`Удалить встречу «${meeting.title}» вместе с транскриптом?`)) return;
    try {
      await api.deleteMeeting(meeting.id);
      await load();
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  return (
    <div className="content">
      <h1>История встреч</h1>
      <p className="page-sub">Транскрипты и протоколы сохраняются локально на сервере.</p>
      {error && <div className="banner error">{error}</div>}

      <div className="card settings-block">
        <div style={{ display: "flex", gap: 10 }}>
          <input
            className="input grow"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && void find()}
            placeholder="О чём говорили? Например: что решили по срокам"
          />
          <button className="btn primary" onClick={() => void find()} disabled={searching}>
            {searching ? <span className="spinner" /> : "Найти"}
          </button>
        </div>
        <span className="hint">
          Ищет по смыслу, а не по словам: «что решили по срокам» найдёт разговор,
          где говорили «двигаем сдачу на следующий месяц»
        </span>
        {searchError && (
          <div className="banner error" style={{ marginTop: 10 }}>
            {searchError}
          </div>
        )}
        {(answering || answer) && (
          <div className="banner" style={{ marginTop: 10 }}>
            {answering ? (
              <span style={{ color: "var(--muted)" }}>
                <span className="spinner" /> модель читает найденное…
              </span>
            ) : (
              <div style={{ whiteSpace: "pre-wrap" }}>{answer}</div>
            )}
          </div>
        )}
        {hits?.length === 0 && (
          <div className="hint" style={{ marginTop: 10 }}>
            Ничего похожего не нашлось
          </div>
        )}
        {hits?.map((hit, i) => (
          <div
            key={`${hit.meeting_id}-${i}`}
            className="list-item"
            style={{ marginTop: 10 }}
            onClick={() => navigate({ name: "meeting", id: hit.meeting_id })}
          >
            <div className="grow">
              <div className="meta">
                {hit.meeting_title} · {formatDate(hit.started_at)} · {mmss(hit.start_s)} ·
                близость {hit.similarity.toFixed(2)}
              </div>
              <div>{hit.text}</div>
            </div>
          </div>
        ))}
      </div>
      {meetings && meetings.length === 0 && (
        <div className="empty">
          <div className="big-icon">🗂️</div>
          Пока пусто — проведите первую встречу
        </div>
      )}
      <div className="list">
        {(meetings ?? []).map((meeting) => (
          <div
            key={meeting.id}
            className="list-item"
            onClick={() => navigate({ name: "meeting", id: meeting.id })}
          >
            <div className="grow">
              <div className="title">
                {meeting.title} {meeting.has_summary && "📝"}
              </div>
              <div className="meta">
                {formatDate(meeting.started_at)}
                {duration(meeting) && ` · ${duration(meeting)}`}
                {` · реплик: ${meeting.segments_count}`}
              </div>
            </div>
            <span className={`chip ${meeting.status}`}>{STATUS_LABEL[meeting.status]}</span>
            <button className="btn small danger" onClick={(event) => remove(meeting, event)}>
              Удалить
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
