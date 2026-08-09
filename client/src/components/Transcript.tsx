import type { SegmentDto } from "../types";
import Avatar, { speakerColor } from "./Avatar";

/** Пауза, после которой реплики одного спикера уже не одно высказывание.
 *  Тридцать секунд — это перерыв в разговоре, а не заминка внутри мысли: склеив
 *  их, мы приписали бы к сказанному до паузы время начала, которое отстоит на
 *  полминуты, и лента врала бы про то, когда что прозвучало. */
const GROUP_GAP_S = 30;

export function formatTime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mmss = `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return h > 0 ? `${h}:${mmss}` : mmss;
}

/** Подряд идущие реплики одного спикера — в одну группу.
 *
 *  Группируем только в отображении: сегменты в БД привязаны ко времени, каналу
 *  и схожести голоса, на них держится диаризация, и склеивать их там нельзя.
 *  Группа рвётся на смене спикера и на паузе длиннее GROUP_GAP_S.
 */
export function groupSegments(segments: SegmentDto[]): SegmentDto[][] {
  const groups: SegmentDto[][] = [];
  for (const segment of segments) {
    const current = groups[groups.length - 1];
    const previous = current?.[current.length - 1];
    const sameSpeaker = previous && (previous.speaker?.id ?? 0) === (segment.speaker?.id ?? 0);
    if (current && sameSpeaker && segment.start_s - previous.end_s <= GROUP_GAP_S) {
      current.push(segment);
    } else {
      groups.push([segment]);
    }
  }
  return groups;
}

function Replica({
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
  return (
    <div className={`replica ${selected ? "picked" : ""}`}>
      <div className="bubble">{segment.text}</div>
      {debug && (
        <span className="msg-debug">
          {segment.channel}
          {segment.similarity != null ? ` · sim ${segment.similarity.toFixed(3)}` : ""}
        </span>
      )}
      {/* Отдельная кнопка, а не клик по пузырю: иначе нельзя было бы
          выделить текст реплики мышью, чтобы его скопировать.
          Кнопка у каждой реплики, а не у группы: спрашивают про конкретную
          фразу, и склейка в отображении не должна этого отнимать. */}
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
  );
}

function Group({
  segments,
  debug,
  selectedIds,
  onToggle,
}: {
  segments: SegmentDto[];
  debug?: boolean;
  selectedIds?: Set<number>;
  onToggle?: (id: number) => void;
}) {
  const first = segments[0];
  const speaker = first.speaker;
  const isSelf = speaker?.is_self ?? false;
  const name = speaker?.name ?? "Неизвестный";
  const id = speaker?.id ?? 0;
  return (
    <div className={`msg ${isSelf ? "self" : ""}`}>
      <Avatar id={id} name={name} isSelf={isSelf} />
      <div className="msg-body">
        {/* Имя и время — по одному разу на группу: время первой реплики
            отвечает на вопрос «когда он это начал говорить». */}
        <div className="msg-meta">
          <span className="msg-name" style={{ color: speakerColor(id, isSelf) }}>
            {name}
          </span>
          <span className="msg-time">{formatTime(first.start_s)}</span>
        </div>
        {segments.map((segment) => (
          <Replica
            key={segment.id}
            segment={segment}
            debug={debug}
            selected={selectedIds?.has(segment.id)}
            onToggle={onToggle}
          />
        ))}
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
      {groupSegments(segments).map((group) => (
        <Group
          key={group[0].id}
          segments={group}
          debug={debug}
          selectedIds={selectedIds}
          onToggle={onToggle}
        />
      ))}
    </div>
  );
}
