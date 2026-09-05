/**
 * Перенос промптов с сервера — сверка с эталоном, снятым с самой серверной
 * реализации (server-golden.json).
 *
 * Обещать «перенёс дословно» дёшево, а формулировки здесь подбирались на живых
 * встречах: лишний пробел в разделе или потерянная строка правил меняют
 * поведение модели молча. Поэтому сравнение посимвольное.
 *
 * Эталон снят 05.09.2026 скриптом с сервера, пока серверная реализация ещё
 * существовала. После её удаления этот файл остаётся единственной записью того,
 * что именно было перенесено.
 */
import { describe, expect, it } from "vitest";

import { buildAnswerPrompt, ASK_ABOUT_SELECTED } from "./answer";
import { buildHintPrompt, parseHint } from "./hint";
import { buildProtocolPrompt } from "./protocol";
import { buildSearchAnswerPrompt } from "./searchAnswer";
import { buildTranscript, splitByLines } from "../transcript";
import type { SegmentDto } from "../../types";
import golden from "./server-golden.json";

/** Те же реплики, что подавались серверной реализации при снятии эталона. */
const сегмент = (start: number, name: string | null, text: string): SegmentDto => ({
  id: 0,
  meeting_id: 1,
  channel: "mic",
  start_s: start,
  end_s: start + 1,
  text,
  similarity: null,
  speaker: name ? { id: 1, name, is_self: false } : null,
});

const СЕГМЕНТЫ: SegmentDto[] = [
  сегмент(0.0, "Сергей", "Давайте по срокам."),
  сегмент(65.4, "Куратор", "Демо переносим на вторник."),
  сегмент(130.9, "Сергей", "Хорошо, к пятнице пришлю разбор."),
  сегмент(190.2, null, "Угу."),
  сегмент(250.0, "Сергей", "И ещё нужен доступ к серверу."),
];

describe("промпт протокола совпадает с серверным", () => {
  it.each(golden.protocol.map((с, i) => [i, с] as const))(
    "случай %i (%o)",
    (_i, случай) => {
      const результат = buildProtocolPrompt({
        mode: случай.input.mode,
        title: случай.input.title,
        date: случай.input.date,
        participants: случай.input.participants,
        text: случай.input.text,
        part: (случай.input as { part?: number }).part,
        total: (случай.input as { total?: number }).total,
      });
      expect(результат.system).toBe(случай.system);
      expect(результат.prompt).toBe(случай.prompt);
    },
  );
});

describe("сборка транскрипта совпадает с серверной", () => {
  it.each(golden.transcript.map((с) => [с.max_chars, с] as const))(
    "предел %i символов",
    (предел, эталон) => {
      const результат = buildTranscript(СЕГМЕНТЫ, предел);
      expect(результат.text).toBe(эталон.text);
      expect(результат.participants).toBe(эталон.participants);
    },
  );
});

describe("нарезка по репликам совпадает с серверной", () => {
  it.each(golden.split.map((с) => [с.max_chars, с] as const))(
    "предел %i символов",
    (предел, эталон) => {
      expect(splitByLines(golden.split_source, предел)).toEqual(эталон.chunks);
    },
  );
});

describe("нарезка не рвёт реплики", () => {
  it("реплика длиннее предела уходит в свой кусок целиком", () => {
    const длинная = "[00:00] Имя: " + "а".repeat(500);
    const куски = splitByLines(`${длинная}\n[00:01] Имя: коротко`, 100);
    expect(куски[0]).toBe(длинная);
    expect(куски).toHaveLength(2);
  });

  it("предел 0 — не режем вовсе (выгрузка полного транскрипта)", () => {
    expect(splitByLines("а\nб\nв", 0)).toEqual(["а\nб\nв"]);
  });
});

describe("промпт подсказки совпадает с серверным", () => {
  it.each(golden.hint.map((с, i) => [i, с] as const))("случай %i", (_i, случай) => {
    const вход = случай.input as Record<string, unknown>;
    const результат = buildHintPrompt({
      mode: вход.mode as string,
      transcript: вход.transcript as string,
      previous: вход.previous as string,
      earlier: вход.earlier as string | undefined,
      title: вход.title as string | undefined,
      participants: вход.participants as string | undefined,
      detailed: вход.detailed as boolean | undefined,
      allowSkip: вход.allow_skip as boolean | undefined,
    });
    expect(результат.system).toBe(случай.system);
    expect(результат.prompt).toBe(случай.prompt);
  });
});

describe("промпт ответа участнику совпадает с серверным", () => {
  it.each(golden.answer.map((с, i) => [i, с] as const))("случай %i", (_i, случай) => {
    const вход = случай.input as Record<string, unknown>;
    const результат = buildAnswerPrompt({
      mode: вход.mode as string,
      question: вход.question as string,
      quoted: вход.quoted as string | undefined,
      earlier: вход.earlier as string | undefined,
      title: вход.title as string | undefined,
      participants: вход.participants as string | undefined,
    });
    expect(результат.system).toBe(случай.system);
    expect(результат.prompt).toBe(случай.prompt);
  });

  it("вопрос по выделенным репликам сформулирован так же", () => {
    expect(ASK_ABOUT_SELECTED).toBe(golden.ask_about_selected);
  });
});

describe("разбор ответа модели совпадает с серверным", () => {
  it.each(golden.parse_hint.map((с) => [с.raw, с.result] as const))(
    "«%s» → %s",
    (сырое, ожидание) => {
      expect(parseHint(сырое)).toBe(ожидание);
    },
  );
});

describe("промпт ответа по прошлым встречам совпадает с серверным", () => {
  it.each(golden.search_answer.map((с, i) => [i, с] as const))("случай %i", (_i, случай) => {
    const результат = buildSearchAnswerPrompt(случай.question, случай.found);
    expect(результат.system).toBe(случай.system);
    expect(результат.prompt).toBe(случай.prompt);
  });
});
