/** Экран «База знаний»: сводка, статус источников, индексация с прогрессом.
 *
 *  Сервер и индексация подменены: проверяется то, что видит человек, а сама
 *  индексация пачками проверена в llm/search.test.ts.
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { IndexProgress } from "../llm/search";
import type { KnowledgeStatusDto } from "../types";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const строка = (id: number, title: string, status: "indexed" | "waiting" | "not_ready" | "empty", chunks = 0, chunks_waiting = 0) =>
  ({ id, title, status, chunks, chunks_waiting, chunks_other_models: 0 });

let состояние: KnowledgeStatusDto;
const спрошено: string[] = [];
let индексация: (onProgress: (p: IndexProgress) => void) => Promise<number> = async () => 0;

vi.mock("../api/rest", () => ({
  api: {
    knowledgeStatus: vi.fn(async (model: string) => {
      спрошено.push(model);
      return состояние;
    }),
    searchPending: vi.fn(), searchIndex: vi.fn(), searchQuery: vi.fn(),
  },
}));
vi.mock("../llm/settings", () => ({ loadLlmSettings: () => ({ embedModel: "bge-m3" }) }));
vi.mock("../llm/search", () => ({
  indexPending: vi.fn((_api: unknown, _s: unknown, _m: string, onProgress: (p: IndexProgress) => void) =>
    индексация(onProgress)),
}));

const { default: KnowledgePage, formatDuration, remainingSeconds } = await import("./KnowledgePage");

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  спрошено.length = 0;
  состояние = {
    model: "bge-m3",
    meetings: [
      { ...строка(1, "Планёрка", "indexed", 12), started_at: null },
      { ...строка(2, "Идёт сейчас", "not_ready"), started_at: null },
    ],
    documents: [{ ...строка(3, "Регламент", "waiting", 0, 480), created_at: "2026-09-17T10:00:00Z", chars: 250000 }],
  };
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

const открыть = () => act(async () => root.render(<KnowledgePage />));
const кнопка = () => Array.from(container.querySelectorAll("button")).find((b) => b.classList.contains("primary"));

describe("экран базы знаний", () => {
  it("сводка говорит, что найдётся и сколько ждёт — с оценкой времени", async () => {
    await открыть();
    expect(спрошено).toEqual(["bge-m3"]);  // статус — для выбранной модели
    const текст = container.textContent ?? "";
    expect(текст).toContain("Найдётся поиском: 1 из 3");
    expect(текст).toContain("ждут индексации: 1 (480 кусков, около 2 мин)");
    expect(текст).toContain("Встреча · встреча не закончена");
    expect(текст).toContain("Документ · ждёт индексации · 480 кусков");
  });

  it("когда ждать нечего, кнопка честно говорит об этом", async () => {
    состояние.documents = [];
    await открыть();
    expect(кнопка()?.textContent).toBe("Всё проиндексировано");
    expect(кнопка()?.disabled).toBe(true);
  });

  it("индексация показывает прогресс с источником, потом перечитывает состояние", async () => {
    let отпустить: () => void = () => {};
    индексация = (onProgress) => {
      onProgress({ chunksDone: 32, chunksTotal: 480, source: "Регламент" });
      return new Promise((resolve) => { отпустить = () => resolve(480); });
    };
    await открыть();
    await act(async () => кнопка()!.click());
    expect(container.querySelector(".banner.info")?.textContent).toContain("Кусок 32 из 480 · «Регламент»");

    состояние.documents[0] = { ...состояние.documents[0], status: "indexed", chunks: 480, chunks_waiting: 0 };
    await act(async () => отпустить());
    expect(container.textContent).toContain("Найдётся поиском: 2 из 3");
    expect(спрошено).toHaveLength(2);
  });

  it("ошибка говорит, что посчитанное сохранено, и даёт продолжить", async () => {
    индексация = async () => {
      throw new Error("Модель недоступна.");
    };
    await открыть();
    await act(async () => кнопка()!.click());
    expect(container.querySelector(".banner.error")?.textContent).toBe(
      "Модель недоступна. Уже посчитанное сохранено — можно продолжить.",
    );
    expect(кнопка()?.textContent).toBe("Проиндексировать");
  });
});

describe("оценка времени", () => {
  it("до начала — по замеру, дальше — по тому, как идёт на самом деле", () => {
    expect(remainingSeconds(0, 480, 0)).toBe(120);            // 4 куска/с по замеру
    expect(remainingSeconds(100, 480, 50_000)).toBe(190);     // идёт 2 куска/с — медленнее замера
  });

  it("минуты и часы читаются человеком", () => {
    expect(formatDuration(30)).toBe("меньше минуты");
    expect(formatDuration(400)).toBe("около 7 мин");
    expect(formatDuration(4000)).toBe("около 1 ч 7 мин");
  });
});
