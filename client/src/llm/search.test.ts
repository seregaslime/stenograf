/**
 * Поиск по встречам на стороне приложения. Сервер и модель подменены —
 * проверяется разделение работы: что считает клиент, что спрашивает у сервера.
 */
import { describe, expect, it, vi } from "vitest";

import { OllamaClient } from "./ollama";
import { LlmRouter, type LlmSettings } from "./router";
import {
  answerByFragments,
  indexPending,
  searchMeetings,
  type PendingMeeting,
  type SearchApi,
} from "./search";
import type { SearchHit } from "../types";

const НАСТРОЙКИ: LlmSettings = {
  provider: "local",
  ollamaUrl: "http://192.168.3.10:11434",
  localSummaryModel: "qwen3:4b",
  localHintsModel: "qwen3:1.7b",
  apiBaseUrl: "",
  apiKey: "",
  apiSummaryModel: "",
  apiHintsModel: "",
  embedModel: "bge-m3",
};

const НАЙДЕНО: SearchHit[] = [
  {
    meeting_id: 7,
    meeting_title: "Планёрка",
    started_at: "2026-08-14T10:00:00",
    start_s: 125.6,
    text: "перенесли демо на вторник",
    similarity: 0.71,
  },
];

/** Сервер-заглушка: помнит, что у него просили и что ему прислали. */
function сервер(ждут: PendingMeeting["chunks"] | null = null) {
  const вызовы = { pending: 0, index: [] as unknown[], query: [] as unknown[] };
  const api: SearchApi = {
    pending: async () => {
      вызовы.pending += 1;
      return ждут
        ? { meetings: [{ meeting_id: 7, title: "Планёрка", chunks: ждут }] }
        : { meetings: [] };
    },
    index: async (body) => {
      вызовы.index.push(body);
      return { chunks: body.chunks.length };
    },
    query: async (body) => {
      вызовы.query.push(body);
      return { results: НАЙДЕНО };
    },
  };
  return { api, вызовы };
}

function модельЭмбеддингов(вектор = [1, 0, 0]) {
  return vi
    .spyOn(OllamaClient.prototype, "embed")
    .mockImplementation(async (_m, texts) => texts.map(() => вектор));
}

const кусок = (i: number) => ({
  first_segment_id: i,
  last_segment_id: i + 1,
  start_s: i * 10,
  text: `кусок разговора ${i}`,
});

describe("индексация", () => {
  it("считает векторы своей моделью и возвращает их вместе с кусками", async () => {
    const embed = модельЭмбеддингов([0.5, 0.5, 0]);
    const { api, вызовы } = сервер([кусок(1), кусок(2)]);

    const посчитано = await indexPending(api, НАСТРОЙКИ, "bge-m3");

    expect(посчитано).toBe(2);
    expect(embed).toHaveBeenCalledWith("bge-m3", ["кусок разговора 1", "кусок разговора 2"]);
    const отправлено = вызовы.index[0] as { chunks: { text: string; vector: number[] }[] };
    // Кусок уходит со своим текстом: пересчитывать нарезку на сервере нельзя
    expect(отправлено.chunks[0].text).toBe("кусок разговора 1");
    expect(отправлено.chunks[0].vector).toEqual([0.5, 0.5, 0]);
    embed.mockRestore();
  });

  it("нечего считать — к модели не ходим вовсе", async () => {
    const embed = модельЭмбеддингов();
    const { api } = сервер(null);
    await expect(indexPending(api, НАСТРОЙКИ, "bge-m3")).resolves.toBe(0);
    expect(embed).not.toHaveBeenCalled();
    embed.mockRestore();
  });
});

describe("поиск", () => {
  it("вектор вопроса считает клиент, сравнение просит у сервера", async () => {
    const embed = модельЭмбеддингов([1, 0, 0]);
    const { api, вызовы } = сервер();

    const результат = await searchMeetings(api, НАСТРОЙКИ, "bge-m3", "что решили по срокам", 5);

    expect(результат).toEqual(НАЙДЕНО);
    expect(вызовы.query[0]).toEqual({ model: "bge-m3", vector: [1, 0, 0], limit: 5 });
    embed.mockRestore();
  });

  it("пустой вопрос не будит ни модель, ни сервер", async () => {
    const embed = модельЭмбеддингов();
    const { api, вызовы } = сервер();
    await expect(searchMeetings(api, НАСТРОЙКИ, "bge-m3", "   ")).resolves.toEqual([]);
    expect(embed).not.toHaveBeenCalled();
    expect(вызовы.query).toHaveLength(0);
    embed.mockRestore();
  });
});

describe("ответ по найденному", () => {
  it("отвечает моделью протокола по показанным фрагментам", async () => {
    const llm = new LlmRouter(НАСТРОЙКИ);
    const роли: string[] = [];
    const промпты: string[] = [];
    vi.spyOn(llm, "generate").mockImplementation(async (role, prompt) => {
      роли.push(role);
      промпты.push(prompt);
      return "  На встрече 14 августа перенесли демо на вторник.  ";
    });

    const ответ = await answerByFragments(llm, "что решили по срокам", НАЙДЕНО);

    expect(ответ).toBe("На встрече 14 августа перенесли демо на вторник.");
    expect(роли).toEqual(["summary"]); // выбор роли живёт в одном месте
    expect(промпты[0]).toContain("Найденные фрагменты записей:");
  });

  it("ничего не нашлось — к модели не ходим и не выдумываем ответ", async () => {
    const llm = new LlmRouter(НАСТРОЙКИ);
    const generate = vi.spyOn(llm, "generate");

    await expect(answerByFragments(llm, "вопрос", [])).resolves.toBe("");
    expect(generate).not.toHaveBeenCalled();
  });
});
