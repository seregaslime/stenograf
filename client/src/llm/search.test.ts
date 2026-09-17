/**
 * Поиск по встречам на стороне приложения. Сервер и модель подменены —
 * проверяется разделение работы: что считает клиент, что спрашивает у сервера.
 */
import { describe, expect, it, vi } from "vitest";

import { OllamaClient } from "./ollama";
import { buildSearchAnswerPrompt } from "./prompts/searchAnswer";
import { LlmRouter, type LlmSettings } from "./router";
import {
  answerByFragments,
  EMBED_BATCH,
  indexPending,
  searchMeetings,
  type IndexProgress,
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
    document_id: null,
    document_title: null,
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

describe("документы базы знаний", () => {
  it("индексируются после встреч, со своим document_id и без реплик", async () => {
    const embed = модельЭмбеддингов([0, 1, 0]);
    const отправлено: unknown[] = [];
    const шаги: IndexProgress[] = [];
    const api: SearchApi = {
      pending: async () => ({
        meetings: [{ meeting_id: 7, title: "Планёрка", chunks: [кусок(1)] }],
        documents: [{ document_id: 3, title: "Регламент", chunks: [{ text: "по вторникам" }] }],
      }),
      index: async (body) => {
        отправлено.push(body);
        return { chunks: body.chunks.length };
      },
      query: async () => ({ results: [] }),
    };

    const посчитано = await indexPending(api, НАСТРОЙКИ, "bge-m3", (шаг) => шаги.push(шаг));

    expect(посчитано).toBe(2);
    expect(отправлено[0]).toMatchObject({ meeting_id: 7 });
    // Встреча первой: новая встреча в поиске нужнее вчерашнего регламента
    expect(отправлено[1]).toEqual({
      model: "bge-m3",
      document_id: 3,
      chunks: [{ text: "по вторникам", vector: [0, 1, 0] }],
    });
    // Прогресс — по кускам, с названием источника, который считается сейчас
    expect(шаги).toEqual([
      { chunksDone: 0, chunksTotal: 2, source: "Планёрка" },
      { chunksDone: 1, chunksTotal: 2, source: "Регламент" },
      { chunksDone: 2, chunksTotal: 2, source: "" },
    ]);
    embed.mockRestore();
  });

  it("в ответе модели кусок документа подписан документом, а не встречей", () => {
    const документ: SearchHit = {
      meeting_id: null, meeting_title: null, started_at: null, start_s: null,
      document_id: 3, document_title: "Регламент созвонов",
      text: "по вторникам в 11", similarity: 0.8,
    };
    const { prompt } = buildSearchAnswerPrompt("когда созвоны", [документ, ...НАЙДЕНО]);
    expect(prompt).toContain("[Документ «Регламент созвонов»]\nпо вторникам в 11");
    expect(prompt).toContain("[Встреча «Планёрка», 2026-08-14, 2:05]");
  });
});

describe("индексация пачками", () => {
  it("большой источник уходит модели пачками, прогресс растёт по кускам", async () => {
    const пачки: number[] = [];
    const embed = vi.spyOn(OllamaClient.prototype, "embed").mockImplementation(async (_m, texts) => {
      пачки.push(texts.length);
      return texts.map(() => [1, 0, 0]);
    });
    const отправлено: { chunks: unknown[] }[] = [];
    const api: SearchApi = {
      pending: async () => ({
        meetings: [],
        documents: [{ document_id: 3, title: "Большой", chunks: Array.from({ length: 70 }, (_, i) => ({ text: `кусок ${i}` })) }],
      }),
      index: async (body) => {
        отправлено.push(body);
        return { chunks: body.chunks.length };
      },
      query: async () => ({ results: [] }),
    };
    const готово: number[] = [];

    await indexPending(api, НАСТРОЙКИ, "bge-m3", (шаг) => готово.push(шаг.chunksDone));

    expect(пачки).toEqual([EMBED_BATCH, EMBED_BATCH, 70 - 2 * EMBED_BATCH]);
    expect(готово).toEqual([0, 32, 64, 70]);
    // Сервер получает источник целиком одним запросом: половина векторов
    // сделала бы документ «проиндексированным с дырой»
    expect(отправлено).toHaveLength(1);
    expect(отправлено[0].chunks).toHaveLength(70);
    embed.mockRestore();
  });

  it("модель упала на втором источнике — первый сохранён, повтор продолжает со второго", async () => {
    let вызов = 0;
    const embed = vi.spyOn(OllamaClient.prototype, "embed").mockImplementation(async (_m, texts) => {
      вызов += 1;
      if (вызов === 2) throw new Error("Модель недоступна");
      return texts.map(() => [1, 0, 0]);
    });
    const сохранено = new Set<number>();
    const api: SearchApi = {
      // Как сервер: отданные на индексацию источники больше не ждут
      pending: async () => ({
        meetings: [7, 8].filter((id) => !сохранено.has(id))
          .map((id) => ({ meeting_id: id, title: `Встреча ${id}`, chunks: [кусок(id)] })),
      }),
      index: async (body) => {
        if ("meeting_id" in body) сохранено.add(body.meeting_id);
        return { chunks: body.chunks.length };
      },
      query: async () => ({ results: [] }),
    };

    await expect(indexPending(api, НАСТРОЙКИ, "bge-m3")).rejects.toThrow("Модель недоступна");
    expect([...сохранено]).toEqual([7]);

    await expect(indexPending(api, НАСТРОЙКИ, "bge-m3")).resolves.toBe(1);
    expect([...сохранено]).toEqual([7, 8]);
    embed.mockRestore();
  });
});
