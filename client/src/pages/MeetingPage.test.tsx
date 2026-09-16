/** Страница встречи: плашка ошибки прошлого резюме во время новой попытки.
 *
 *  Сервер и модель подменены: проверяется только то, что видит человек, пока
 *  протокол составляется, — модель тут ни при чём.
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { MeetingDetail } from "../types";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const ПРОШЛАЯ_ОШИБКА = "Не удалось подключиться к модели по адресу http://127.0.0.1:11434";

const встреча: MeetingDetail = {
  id: 1, title: "Планёрка", status: "done",
  started_at: "2026-09-16T10:00:00+00:00", ended_at: "2026-09-16T10:30:00+00:00",
  record_audio: false, meeting_mode: "work",
  summary: null, summary_model: null, summary_error: ПРОШЛАЯ_ОШИБКА, summary_progress: null,
  segments: [{
    id: 1, meeting_id: 1, channel: "mic", start_s: 0, end_s: 2,
    text: "Давайте начнём", similarity: null, speaker: null,
  }],
};

// Составление протокола висит, пока тест сам его не отпустит
let отпустить: (текст: string) => void = () => {};

vi.mock("../api/rest", () => ({
  api: {
    meeting: vi.fn(async () => встреча),
    speakers: vi.fn(async () => []),
    saveSummary: vi.fn(async () => ({})),
  },
}));
vi.mock("../llm/settings", () => ({ loadLlmSettings: () => ({}), llmReady: () => true }));
vi.mock("../llm/router", () => ({
  LlmRouter: class { modelFor() { return "qwen3:4b"; } },
}));
vi.mock("../llm/summary", () => ({
  generateSummary: () => new Promise<string>((resolve) => { отпустить = resolve; }),
}));

const { default: MeetingPage } = await import("./MeetingPage");

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

const плашки = () => Array.from(container.querySelectorAll(".banner.warn")).map((b) => b.textContent);

describe("ошибка прошлого резюме", () => {
  it("видна, пока новой попытки нет", async () => {
    await act(async () => root.render(<MeetingPage id={1} navigate={() => {}} />));
    expect(плашки()).toContain(ПРОШЛАЯ_ОШИБКА);
  });

  it("прячется, пока составляется новое резюме, — иначе новая попытка выглядит упавшей", async () => {
    await act(async () => root.render(<MeetingPage id={1} navigate={() => {}} />));
    const кнопка = Array.from(container.querySelectorAll("button"))
      .find((b) => b.textContent?.includes("Создать резюме"))!;
    await act(async () => кнопка.click());

    expect(container.querySelector(".banner.info")).not.toBeNull();  // составляется
    expect(плашки()).not.toContain(ПРОШЛАЯ_ОШИБКА);

    await act(async () => отпустить("## Итоги"));
  });
});
