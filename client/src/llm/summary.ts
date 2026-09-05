/**
 * Протокол встречи считает клиент. Перенесено с сервера (llm/summary.py):
 * влезаем одним запросом — считаем сразу, не влезаем — режем по фрагментам,
 * собираем заметки и сводим их тем же промптом.
 *
 * Числа не тронуты, они подбирались на живых встречах: 4000 символов —
 * минимальный фрагмент (на каждый уходит минута паузы), три деления пополам
 * при неудаче (дальше дело не в размере).
 */
import { LlmError } from "./ollama";
import { buildProtocolPrompt } from "./prompts/protocol";
import type { LlmRouter } from "./router";
import { CHARS_PER_TOKEN } from "./router";
import { MAX_RETRY_DEPTH, MIN_CHUNK_CHARS, buildTranscript, splitByLines } from "./transcript";
import type { SegmentDto } from "../types";

export interface SummaryInput {
  segments: SegmentDto[];
  title: string;
  /** «04.09.2026 19:30» — как показывается человеку. */
  date: string;
  mode: string | null;
}

/** Шаг и всего: длинная встреча идёт фрагментами, и молчать про это нельзя. */
export type OnProgress = (step: number, total: number) => void;

const ПРИЗНАК_ОБРЫВА = "не уместила ответ";

/**
 * Заметки по фрагменту. Не уместился ответ — делим пополам и пробуем снова.
 *
 * Сколько модель потратит на ответ, заранее не известно: у рассуждающих часть
 * лимита уходит на мысли, и доля эта гуляет от фрагмента к фрагменту. Гадать
 * бессмысленно — подстраиваемся по факту обрыва.
 */
async function notesFor(
  llm: LlmRouter,
  chunk: string,
  head: { mode: string | null; title: string; date: string; participants: string },
  part: number,
  total: number,
  depth = 0,
): Promise<string> {
  const { system, prompt } = buildProtocolPrompt({ ...head, text: chunk, part, total });
  try {
    return await llm.generate("summary", prompt, { system, temperature: 0.3 });
  } catch (exc) {
    const обрыв = exc instanceof Error && exc.message.includes(ПРИЗНАК_ОБРЫВА);
    // Рекурсия, а не одна пересдача: половина тоже может не влезть. Глубина
    // ограничена — если не уместилось и на восьмушке, дело не в размере.
    if (!обрыв || depth >= MAX_RETRY_DEPTH) throw exc;
  }

  const halves = splitByLines(chunk, Math.max(Math.trunc(chunk.length / 2), 1));
  if (halves.length < 2) {
    throw new LlmError("Модель не уместила ответ даже на одной реплике.");
  }
  const pieces: string[] = [];
  for (const half of halves) {
    pieces.push(await notesFor(llm, half, head, part, total, depth + 1));
  }
  return pieces.join("\n");
}

/**
 * Длинная встреча: протокол по каждому фрагменту → склейка в общий.
 *
 * Паузы между запросами нет намеренно: ждать столько, сколько нужно, умеет сам
 * клиент API — он читает из заголовков ответа, сколько лимита осталось и когда
 * восстановится. Фиксированная минута была хуже вдвойне: на скользящем окне её
 * не хватало, а при свободном лимите она тратилась впустую.
 *
 * Промежуточные заметки живут в памяти: закрыли приложение — протокол
 * пересоздаётся целиком.
 */
async function summarizeInParts(
  llm: LlmRouter,
  chunks: string[],
  head: { mode: string | null; title: string; date: string; participants: string },
  onProgress: OnProgress,
  notesBudgetChars: number,
): Promise<string> {
  let notes: string[] = [];
  for (const [index, chunk] of chunks.entries()) {
    onProgress(index + 1, chunks.length);
    notes.push(
      `— Фрагмент ${index + 1} —\n` +
        (await notesFor(llm, chunk, head, index + 1, chunks.length)),
    );
  }

  // Заметки тоже обязаны влезть в запрос: у встречи на пять фрагментов они
  // сами набирают тысячи символов, и сведение упирается в тот же лимит, что и
  // транскрипт. Не влезли — сжимаем тем же проходом, который делал заметки.
  let level = 0;
  while (notes.join("\n\n").length > notesBudgetChars && level < MAX_RETRY_DEPTH) {
    level += 1;
    const pieces = splitByLines(notes.join("\n\n"), notesBudgetChars);
    const сжатые: string[] = [];
    for (const [index, piece] of pieces.entries()) {
      сжатые.push(await notesFor(llm, piece, head, index + 1, pieces.length));
    }
    notes = сжатые;
  }

  onProgress(chunks.length + 1, chunks.length + 1);
  // Склейка протоколов фрагментов — тот же промпт: для него это просто текст,
  // по которому надо составить протокол.
  const { system, prompt } = buildProtocolPrompt({ ...head, text: notes.join("\n\n") });
  return llm.generate("summary", prompt, { system, temperature: 0.3 });
}

/**
 * Протокол встречи. Бросает LlmError с человеческим текстом, если составить
 * нельзя — вызывающий отдаёт эту причину серверу, чтобы встреча не висела в
 * «составляется» молча.
 */
export async function generateSummary(
  llm: LlmRouter,
  input: SummaryInput,
  onProgress: OnProgress = () => {},
): Promise<string> {
  if (input.segments.length === 0) {
    throw new LlmError("Встреча не содержит распознанной речи.");
  }
  const budget = llm.budget;
  const { text: transcript, participants } = buildTranscript(input.segments, budget.summaryChars);
  const head = { mode: input.mode, title: input.title, date: input.date, participants };
  const { system, prompt } = buildProtocolPrompt({ ...head, text: transcript });

  // Влезаем ли одним запросом. Считаем вместе с промптом: транскрипт,
  // обрезанный ровно по бюджету, уедет к провайдеру с инструкциями и всё равно
  // получит отказ — на подсказках эта ошибка уже была.
  let chunks = [transcript];
  let limitChars = 0;
  if (budget.summaryTokens) {
    limitChars =
      Math.trunc(budget.summaryTokens * CHARS_PER_TOKEN) -
      system.length - prompt.length + transcript.length;
    if (transcript.length > limitChars) {
      if (limitChars < MIN_CHUNK_CHARS) {
        // Бюджет меньше самого промпта. Резать по крохам нельзя: на каждый
        // кусок уходит минута паузы, и встреча на 40 минут превратилась бы в
        // получасовое молчаливое ожидание.
        throw new LlmError(
          "Лимит тарифа слишком мал для протокола этой встречи. " +
            "Выберите модель с большим лимитом токенов в минуту.",
        );
      }
      chunks = splitByLines(transcript, limitChars);
    }
  }

  if (chunks.length > 1) {
    return summarizeInParts(llm, chunks, head, onProgress, limitChars);
  }
  return llm.generate("summary", prompt, { system, temperature: 0.3 });
}
