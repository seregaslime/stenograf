import type { SegmentDto } from "../types";
import Avatar, { speakerColor } from "./Avatar";

export function formatTime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mmss = `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return h > 0 ? `${h}:${mmss}` : mmss;
}

function Message({
  segment,
  debug,
  selected,
  onToggle,
}: {
  segment: SegmentDto;
  debug?: boolean;
  selected?: boolean;
  onToggle?: (id: number) => void;
}) {
  const speaker = segment.speaker;
  const isSelf = speaker?.is_self ?? false;
  const name = speaker?.name ?? "Неизвестный";
  const id = speaker?.id ?? 0;
  return (
    <div className={`msg ${isSelf ? "self" : ""} ${selected ? "picked" : ""}`}>
      <Avatar id={id} name={name} isSelf={isSelf} />
      <div className="msg-body">
        <div className="msg-meta">
          <span className="msg-name" style={{ color: speakerColor(id, isSelf) }}>
            {name}
          </span>
          <span className="msg-time">{formatTime(segment.start_s)}</span>
          {debug && (
            <span className="msg-debug">
              {segment.channel}
              {segment.similarity != null ? ` · sim ${segment.similarity.toFixed(3)}` : ""}
            </span>
          )}
          {/* Отдельная кнопка, а не клик по пузырю: иначе нельзя было бы
              выделить текст реплики мышью, чтобы его скопировать. */}
          {onToggle && (
            <button
              type="button"
              className={`msg-pick ${selected ? "on" : ""}`}
              title={selected ? "Убрать из вопроса" : "Спросить про эту реплику"}
              aria-pressed={selected}
              onClick={() => onToggle(segment.id)}
            >
              {selected ? "✓" : "?"}
            </button>
          )}
        </div>
        <div className="bubble">{segment.text}</div>
      </div>
    </div>
  );
}

export default function Transcript({
  segments,
  debug,
  selectedIds,
  onToggle,
}: {
  segments: SegmentDto[];
  debug?: boolean;
  /** Выделенные реплики — про них будет задан вопрос модели. */
  selectedIds?: Set<number>;
  /** Не передан — режим выделения выключен (история встречи). */
  onToggle?: (id: number) => void;
}) {
  return (
    <div className="transcript">
      {segments.map((segment) => (
        <Message
          key={segment.id}
          segment={segment}
          debug={debug}
          selected={selectedIds?.has(segment.id)}
          onToggle={onToggle}
        />
      ))}
    </div>
  );
}
