/**
 * Транскрипт для модели: сборка текста и нарезка длинной встречи.
 * Перенесено с сервера (llm/summary.py) без изменения чисел — они подбирались
 * на живых встречах.
 */
import type { SegmentDto } from "../types";

/** qwen3:4b с контекстом 8k токенов: ~12000 символов русского текста влезает с запасом. */
export const MAX_TRANSCRIPT_CHARS = 12_000;

/**
 * Меньше этого фрагменты не режем. На каждый уходит минута паузы, поэтому куски
 * по паре тысяч символов превратили бы встречу на 40 минут в получасовое
 * ожидание — молча. Лучше честно сказать, что тариф не тянет.
 */
export const MIN_CHUNK_CHARS = 4_000;

/**
 * Сколько раз подряд делить фрагмент пополам, если ответ не умещается. Три —
 * это восьмушка исходного куска; если не влезло и в неё, дело не в размере, а
 * каждое деление стоит минуты паузы.
 */
export const MAX_RETRY_DEPTH = 3;

/** «02:05» — таймкод реплики, как его писал сервер: с ведущим нулём.
 *  У истории встреч свой формат (без нуля), он к промптам отношения не имеет. */
export function mmss(seconds: number): string {
  const целых = Math.trunc(seconds);
  return `${String(Math.trunc(целых / 60)).padStart(2, "0")}:${String(целых % 60).padStart(2, "0")}`;
}

export interface Transcript {
  text: string;
  /** «Иван (12 реплик), Пётр (4 реплики)» — уходит в шапку промпта. */
  participants: string;
}

/**
 * Текст транскрипта и строка со статистикой участников.
 *
 * maxChars = 0 — не усекать: для API с большим контекстом и для выгрузки, где
 * выбрасывать середину нельзя, человек скачивает полную расшифровку.
 */
export function buildTranscript(
  segments: SegmentDto[],
  maxChars: number = MAX_TRANSCRIPT_CHARS,
): Transcript {
  const lines: string[] = [];
  const counter = new Map<string, number>();
  for (const segment of segments) {
    const name = segment.speaker?.name ?? "Неизвестный";
    counter.set(name, (counter.get(name) ?? 0) + 1);
    lines.push(`[${mmss(segment.start_s)}] ${name}: ${segment.text}`);
  }
  let text = lines.join("\n");
  if (maxChars && text.length > maxChars) {
    // Голова и хвост: начало задаёт тему, конец несёт договорённости, а
    // середина длинной встречи — самое безопасное, что можно выбросить.
    const head = text.slice(0, Math.trunc(maxChars / 4));
    const tail = text.slice(-(maxChars - head.length));
    text = head + "\n[... часть транскрипта опущена из-за длины ...]\n" + tail;
  }
  const participants = [...counter.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([name, n]) => `${name} (${n} реплик)`)
    .join(", ");
  return { text, participants };
}

/**
 * Режет транскрипт на куски не длиннее maxChars — ПО ГРАНИЦАМ РЕПЛИК.
 *
 * Резать по символам нельзя: фраза разорвётся пополам, и обе половины станут
 * бессмыслицей — а именно по ним модель и будет составлять заметки. Реплика
 * длиннее лимита целиком уходит в свой кусок: рвать её всё равно некуда.
 */
export function splitByLines(transcript: string, maxChars: number): string[] {
  if (maxChars <= 0) return [transcript];
  const chunks: string[] = [];
  let current: string[] = [];
  let size = 0;
  for (const line of transcript.split("\n")) {
    if (current.length > 0 && size + line.length + 1 > maxChars) {
      chunks.push(current.join("\n"));
      current = [];
      size = 0;
    }
    current.push(line);
    size += line.length + 1;
  }
  if (current.length > 0) chunks.push(current.join("\n"));
  return chunks;
}
