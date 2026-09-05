/**
 * Поиск по встречам на стороне приложения. Сервер и модель подменены —
 * проверяется разделение работы: что считает клиент, что спрашивает у сервера.
 */
import { describe, expect, it, vi } from "vitest";

import { OllamaClient } from "./ollama";
import { LlmRouter, type LlmSettings } from "./router";
import {
  answerFromMeetings,
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

describe("ответ по архиву", () => {
  it("отвечает по найденному и возвращает цитаты рядом с ответом", async () => {
    const embed = модельЭмбеддингов();
    const { api } = сервер();
    const llm = new LlmRouter(НАСТРОЙКИ);
    const промпты: string[] = [];
    vi.spyOn(llm, "generate").mockImplementation(async (_role, prompt) => {
      промпты.push(prompt);
      return "  На встрече 14 августа перенесли демо на вторник.  ";
    });

    const { answer, results } = await answerFromMeetings(
      api, llm, НАСТРОЙКИ, "bge-m3", "что решили по срокам",
    );

    expect(answer).toBe("На встрече 14 августа перенесли демо на вторник.");
    expect(results).toEqual(НАЙДЕНО); // цитаты обязаны прийти вместе с ответом
    expect(промпты[0]).toContain("Найденные фрагменты записей:");
    embed.mockRestore();
  });

  it("ничего не нашлось — к модели не ходим и не выдумываем ответ", async () => {
    const embed = модельЭмбеддингов();
    const api: SearchApi = {
      pending: async () => ({ meetings: [] }),
      index: async () => ({ chunks: 0 }),
      query: async () => ({ results: [] }),
    };
    const llm = new LlmRouter(НАСТРОЙКИ);
    const generate = vi.spyOn(llm, "generate");

    const { answer, results } = await answerFromMeetings(api, llm, НАСТРОЙКИ, "bge-m3", "вопрос");

    expect(answer).toBe("");
    expect(results).toEqual([]);
    expect(generate).not.toHaveBeenCalled();
    embed.mockRestore();
  });
});
