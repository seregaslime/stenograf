import { Fragment, useEffect, useRef, useState } from "react";

import type { SegmentDto, SpeakerRef } from "../types";
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

/** Кому отдать выделенные слова: номера первого и последнего слова реплики. */
export type Reassign = (
  segmentId: number, firstWord: number, lastWord: number, speakerId: number,
) => Promise<void>;

/** Какие слова пузыря выделены мышью: [первое, последнее] или null.
 *
 *  Слово считается выделенным, если в выделение попала хотя бы одна его буква.
 *  Касания мало: протянув выделение от самого конца предыдущего слова, человек
 *  его не выбирал, а у выделения граница стоит ровно на его последней букве.
 *  Выделение, вылезшее за пузырь, не в счёт — это копирование текста из ленты,
 *  а не просьба кого-то переназначить.
 */
export function selectedWords(bubble: Element, selection: Selection | null): [number, number] | null {
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) return null;
  const range = selection.getRangeAt(0);
  if (!bubble.contains(range.commonAncestorContainer)) return null;
  const номера: number[] = [];
  bubble.querySelectorAll<HTMLElement>("[data-word]").forEach((span) => {
    // Границы — по тексту слова, а не по элементу: «после последней буквы» и
    // «после элемента» для Range разные точки, и первая оказалась бы внутри слова.
    const слово = document.createRange();
    слово.selectNodeContents(span.firstChild ?? span);
    const конецПозжеНачалаСлова = range.compareBoundaryPoints(Range.START_TO_END, слово) > 0;
    const началоРаньшеКонцаСлова = range.compareBoundaryPoints(Range.END_TO_START, слово) < 0;
    if (конецПозжеНачалаСлова && началоРаньшеКонцаСлова) номера.push(Number(span.dataset.word));
  });
  return номера.length ? [Math.min(...номера), Math.max(...номера)] : null;
}

/** Реплика поделилась на куски — они встают на её место, порядок ленты сохраняется. */
export function replaceSegment(
  segments: SegmentDto[], segmentId: number, parts: SegmentDto[],
): SegmentDto[] {
  return segments.flatMap((segment) => (segment.id === segmentId ? parts : [segment]));
}

function Replica({
  segment,
  debug,
  selected,
  onToggle,
  speakers,
  onReassign,
}: {
  segment: SegmentDto;
  debug?: boolean;
  selected?: boolean;
  onToggle?: (id: number) => void;
  speakers?: SpeakerRef[];
  onReassign?: Reassign;
}) {
  const bubble = useRef<HTMLDivElement>(null);
  const [range, setRange] = useState<[number, number] | null>(null);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState("");
  // Без времени слов делить нечем: старые встречи и живая лента показывают текст как есть.
  const words = onReassign ? segment.words : null;
  const candidates = (speakers ?? []).filter((s) => s.id !== segment.speaker?.id);

  // Слушаем документ, а не пузырь: выделение тянут до последнего слова и
  // отпускают кнопку уже за краем пузыря — на самом пузыре события не будет.
  // Отпускание внутри меню не в счёт: нажатие на кнопку спикера сбрасывает
  // выделение, и меню пропало бы раньше, чем до кнопки дойдёт щелчок.
  useEffect(() => {
    if (!words) return;
    const pick = (event: Event) => {
      if (!bubble.current || (event.target as Element | null)?.closest?.(".reassign")) return;
      setRange(selectedWords(bubble.current, window.getSelection()));
      setFailure("");
    };
    document.addEventListener("mouseup", pick);
    document.addEventListener("keyup", pick);
    return () => {
      document.removeEventListener("mouseup", pick);
      document.removeEventListener("keyup", pick);
    };
  }, [words]);

  const give = async (speakerId: number) => {
    if (!range || !onReassign) return;
    setBusy(true);
    try {
      await onReassign(segment.id, range[0], range[1], speakerId);
      window.getSelection()?.removeAllRanges();
      setRange(null);
    } catch (exc) {
      setFailure((exc as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className={`replica ${selected ? "picked" : ""}`}>
      <div className="replica-main">
        <div className="bubble" ref={bubble}>
          {/* Слова отдельными элементами — чтобы по выделению понять, какие
              именно; пробелы между ними обычным текстом, и копирование даёт
              ту же строку, что и раньше. */}
          {words
            ? words.map(([, , слово], номер) => (
              <Fragment key={номер}>
                {номер > 0 && " "}
                <span data-word={номер}>{слово}</span>
              </Fragment>
            ))
            : segment.text}
        </div>
        {range && words && (
          <div className="reassign" role="group" aria-label="Отдать выделенные слова">
            <span className="reassign-words">
              «{words.slice(range[0], range[1] + 1).map(([, , слово]) => слово).join(" ")}» →
            </span>
            {candidates.map((speaker) => (
              <button
                key={speaker.id}
                type="button"
                className="btn small"
                disabled={busy}
                onClick={() => void give(speaker.id)}
              >
                {speaker.name}
              </button>
            ))}
            {candidates.length === 0 && <span className="reassign-words">других спикеров нет</span>}
            <button type="button" className="btn small" disabled={busy} onClick={() => setRange(null)}>
              Отмена
            </button>
            {failure && <span className="reassign-error">{failure}</span>}
          </div>
        )}
      </div>
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

/** Новое имя спикера — во ВСЕХ его репликах.
 *
 *  Имя лежит в каждом сегменте копией (так его присылает сервер), поэтому
 *  переименование обязано пройти по всей ленте: иначе один и тот же человек
 *  остался бы «Спикером 3» выше по разговору и «Иваном» ниже.
 */
export function renameInSegments(
  segments: SegmentDto[], speakerId: number, name: string,
): SegmentDto[] {
  return segments.map((segment) =>
    segment.speaker && segment.speaker.id === speakerId
      ? { ...segment, speaker: { ...segment.speaker, name } }
      : segment,
  );
}

/** Двух спикеров объединили — переписываем ленту.
 *
 *  Реплики исчезнувшего профиля переезжают на целевой: без этого один человек
 *  оставался бы в ленте двумя — со старым id, старым цветом аватарки и старым
 *  именем, — пока страницу не перезагрузят.
 */
export function applyMergeToSegments(
  segments: SegmentDto[], sourceId: number, targetId: number, name: string,
): SegmentDto[] {
  return segments.map((segment) => {
    if (!segment.speaker) return segment;
    const mine = segment.speaker.id === sourceId || segment.speaker.id === targetId;
    if (!mine) return segment;
    return { ...segment, speaker: { ...segment.speaker, id: targetId, name } };
  });
}

function SpeakerName({
  id,
  name,
  isSelf,
  onRename,
}: {
  id: number;
  name: string;
  isSelf: boolean;
  onRename?: (id: number, name: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const color = speakerColor(id, isSelf);
  // Переименовывать некого, пока спикер не опознан: id нулевой, и правка ушла
  // бы в никуда.
  if (!onRename || id === 0) {
    return <span className="msg-name" style={{ color }}>{name}</span>;
  }
  if (editing) {
    return (
      <input
        className="msg-name-input"
        defaultValue={name}
        autoFocus
        style={{ color }}
        onBlur={(event) => {
          setEditing(false);
          const value = event.target.value.trim();
          if (value && value !== name) onRename(id, value);
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") event.currentTarget.blur();
          if (event.key === "Escape") {
            event.currentTarget.value = name;  // отмена: blur сравнит и не тронет
            event.currentTarget.blur();
          }
        }}
      />
    );
  }
  return (
    <button
      type="button"
      className="msg-name as-button"
      style={{ color }}
      title="Переименовать участника"
      onClick={() => setEditing(true)}
    >
      {name}
    </button>
  );
}

function Group({
  segments,
  debug,
  selectedIds,
  onToggle,
  onRename,
  speakers,
  onReassign,
}: {
  segments: SegmentDto[];
  debug?: boolean;
  selectedIds?: Set<number>;
  onToggle?: (id: number) => void;
  onRename?: (id: number, name: string) => void;
  speakers?: SpeakerRef[];
  onReassign?: Reassign;
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
          <SpeakerName id={id} name={name} isSelf={isSelf} onRename={onRename} />
          <span className="msg-time">{formatTime(first.start_s)}</span>
        </div>
        {segments.map((segment) => (
          <Replica
            key={segment.id}
            segment={segment}
            debug={debug}
            selected={selectedIds?.has(segment.id)}
            onToggle={onToggle}
            speakers={speakers}
            onReassign={onReassign}
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
  onRename,
  speakers,
  onReassign,
}: {
  segments: SegmentDto[];
  debug?: boolean;
  /** Выделенные реплики — про них будет задан вопрос модели. */
  selectedIds?: Set<number>;
  /** Не передан — режим выделения выключен (история встречи). */
  onToggle?: (id: number) => void;
  /** Не передан — имена не редактируются (история встречи). */
  onRename?: (id: number, name: string) => void;
  /** Кому можно отдать слова — только существующие спикеры, нового отсюда не завести. */
  speakers?: SpeakerRef[];
  /** Не передан — слова не переназначаются (живая лента: времени слов в ней нет). */
  onReassign?: Reassign;
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
          onRename={onRename}
          speakers={speakers}
          onReassign={onReassign}
        />
      ))}
    </div>
  );
}
