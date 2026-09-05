/**
 * Протокол на клиенте: что происходит с длинной встречей, с обрывом ответа и с
 * тарифом, которого не хватает. Модель подменена — проверяется наша логика, а
 * не качество формулировок (за формулировки отвечает сверка с эталоном в
 * prompts/port.test.ts).
 */
import { describe, expect, it, vi } from "vitest";

import { LlmError } from "./ollama";
import { LlmRouter, type LlmSettings } from "./router";
import { generateSummary } from "./summary";
import type { SegmentDto } from "../types";

const НАСТРОЙКИ: LlmSettings = {
  provider: "local",
  ollamaUrl: "http://192.168.3.10:11434",
  localSummaryModel: "qwen3:4b",
  localHintsModel: "qwen3:1.7b",
  apiBaseUrl: "https://api.groq.com/openai/v1",
  apiKey: "ключ",
  apiSummaryModel: "gpt-oss-120b",
  apiHintsModel: "gpt-oss-20b",
};

function реплики(сколько: number, длина = 40): SegmentDto[] {
  return Array.from({ length: сколько }, (_, i) => ({
    id: i,
    meeting_id: 1,
    channel: "mic" as const,
    start_s: i * 10,
    end_s: i * 10 + 5,
    text: `реплика ${i} `.padEnd(длина, "и"),
    similarity: null,
    speaker: { id: 1, name: "Сергей", is_self: true },
  }));
}

const встреча = { title: "Планёрка", date: "05.09.2026 10:00", mode: "work" };

/** Роутер с подменённой моделью: возвращает, что скажут, и считает вызовы. */
function роутер(ответ: (prompt: string, номер: number) => string, settings = НАСТРОЙКИ) {
  const llm = new LlmRouter(settings);
  const промпты: string[] = [];
  vi.spyOn(llm, "generate").mockImplementation(async (_role, prompt) => {
    промпты.push(prompt);
    return ответ(prompt, промпты.length);
  });
  return { llm, промпты };
}

describe("протокол одним запросом", () => {
  it("короткая встреча — один вызов модели", async () => {
    const { llm, промпты } = роутер(() => "## Краткий итог\nПеренесли демо.");
    const текст = await generateSummary(llm, { ...встреча, segments: реплики(5) });

    expect(текст).toBe("## Краткий итог\nПеренесли демо.");
    expect(промпты).toHaveLength(1);
    expect(промпты[0]).toContain("Расшифровка:");
  });

  it("встреча без речи — понятная причина, а не пустой протокол", async () => {
    const { llm } = роутер(() => "неважно");
    await expect(generateSummary(llm, { ...встреча, segments: [] })).rejects.toThrow(
      /не содержит распознанной речи/,
    );
  });
});

describe("длинная встреча по фрагментам", () => {
  /** Тариф, при котором транскрипт заведомо не влезает одним запросом. */
  const тесныйТариф: LlmSettings = {
    ...НАСТРОЙКИ,
    provider: "api",
    tpmLimits: { "gpt-oss-120b": 8000 },
  };

  it("режет, считает заметки и сводит их отдельным запросом", async () => {
    const { llm, промпты } = роутер((_p, номер) => `заметки ${номер}`, тесныйТариф);
    const шаги: [number, number][] = [];

    const текст = await generateSummary(
      llm,
      { ...встреча, segments: реплики(400, 60) },
      (шаг, всего) => шаги.push([шаг, всего]),
    );

    expect(промпты.length).toBeGreaterThan(2); // фрагменты + сведение
    expect(текст).toBe(`заметки ${промпты.length}`); // последний вызов — сведение
    // Прогресс доходит до конца: последний шаг равен общему числу
    expect(шаги.at(-1)![0]).toBe(шаги.at(-1)![1]);
    // Фрагменты пронумерованы для модели
    expect(промпты[0]).toContain("Это фрагмент 1 из");
  });

  it("тариф меньше самого промпта — честный отказ вместо крошева", async () => {
    // На каждый кусок уходит минута паузы: встреча на 40 минут превратилась бы
    // в получасовое молчаливое ожидание
    const { llm } = роутер(() => "не дойдёт", {
      ...тесныйТариф,
      tpmLimits: { "gpt-oss-120b": 1500 },
    });
    await expect(
      generateSummary(llm, { ...встреча, segments: реплики(400, 60) }),
    ).rejects.toThrow(/Лимит тарифа слишком мал/);
  });
});

describe("модель не уместила ответ", () => {
  /** Пересдача пополам живёт в ветке фрагментов — как и на сервере: одиночный
   *  запрос повторов не делает, и порт этого не менял. */
  const тесныйТариф: LlmSettings = {
    ...НАСТРОЙКИ,
    provider: "api",
    tpmLimits: { "gpt-oss-120b": 8000 },
  };

  it("делит фрагмент пополам и пробует снова", async () => {
    let первый = true;
    const { llm, промпты } = роутер(() => {
      if (первый) {
        первый = false;
        throw new LlmError("Модель не уместила ответ в лимит");
      }
      return "заметки";
    }, тесныйТариф);

    const текст = await generateSummary(llm, { ...встреча, segments: реплики(400, 60) });
    expect(текст).toBe("заметки");
    // Первый фрагмент ушёл дважды: целиком и половинами
    expect(промпты.length).toBeGreaterThan(3);
  });

  it("чужая ошибка пробрасывается, а не дробится бесконечно", async () => {
    const { llm, промпты } = роутер(() => {
      throw new LlmError("Модель недоступна по адресу http://…");
    });
    await expect(generateSummary(llm, { ...встреча, segments: реплики(5) })).rejects.toThrow(
      /недоступна/,
    );
    expect(промпты).toHaveLength(1); // делить пополам смысла нет
  });
});
