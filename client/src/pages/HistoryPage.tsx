import { useEffect, useState, type MouseEvent } from "react";
import type { Page } from "../App";
import { api } from "../api/rest";
import { formatTime } from "../components/Transcript";
import type { MeetingListItem } from "../types";

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

export default function HistoryPage({ navigate }: { navigate: (page: Page) => void }) {
  const [meetings, setMeetings] = useState<MeetingListItem[] | null>(null);
  const [error, setError] = useState("");

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
