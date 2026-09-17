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

// В jsdom у File нет arrayBuffer(), в браузере и Electron он есть — досказываем
if (!File.prototype.arrayBuffer) {
  File.prototype.arrayBuffer = function (this: File) {
    return new Promise<ArrayBuffer>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as ArrayBuffer);
      reader.onerror = () => reject(reader.error);
      reader.readAsArrayBuffer(this);
    });
  };
}

const строка = (id: number, title: string, status: "indexed" | "waiting" | "not_ready" | "empty", chunks = 0, chunks_waiting = 0) =>
  ({ id, title, status, chunks, chunks_waiting, chunks_other_models: 0 });

let состояние: KnowledgeStatusDto;
const спрошено: string[] = [];
const загружено: { filename: string; bytes: number[] }[] = [];
const удалено: number[] = [];
let отказЗагрузки: Error | null = null;
let индексация: (onProgress: (p: IndexProgress) => void) => Promise<number> = async () => 0;

vi.mock("../api/rest", () => ({
  api: {
    knowledgeStatus: vi.fn(async (model: string) => {
      спрошено.push(model);
      return состояние;
    }),
    searchPending: vi.fn(), searchIndex: vi.fn(), searchQuery: vi.fn(),
    uploadDocument: vi.fn(async (filename: string, bytes: Uint8Array) => {
      if (отказЗагрузки) throw отказЗагрузки;
      загружено.push({ filename, bytes: Array.from(bytes) });
      return {};
    }),
    deleteDocument: vi.fn(async (id: number) => {
      удалено.push(id);
      return { deleted: id };
    }),
  },
}));
vi.mock("../llm/settings", () => ({ loadLlmSettings: () => ({ embedModel: "bge-m3" }) }));
vi.mock("../llm/search", () => ({
  indexPending: vi.fn((_api: unknown, _s: unknown, _m: string, onProgress: (p: IndexProgress) => void) =>
    индексация(onProgress)),
}));

const { default: KnowledgePage, formatDuration, remainingSeconds, plural } = await import("./KnowledgePage");

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  спрошено.length = 0;
  загружено.length = 0;
  удалено.length = 0;
  отказЗагрузки = null;
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

  it("сообщение модели без точки не склеивается с припиской", async () => {
    // Так ответила Ollama вживую: команда в конце, точки нет
    индексация = async () => {
      throw new Error("Модель «nomic-embed-text» не найдена. Скачайте её: ollama pull nomic-embed-text");
    };
    await открыть();
    await act(async () => кнопка()!.click());
    expect(container.querySelector(".banner.error")?.textContent).toBe(
      "Модель «nomic-embed-text» не найдена. Скачайте её: ollama pull nomic-embed-text. Уже посчитанное сохранено — можно продолжить.",
    );
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

describe("смена модели эмбеддингов", () => {
  it("предупреждает, если источники посчитаны другой моделью", async () => {
    состояние.documents[0] = { ...состояние.documents[0], chunks_other_models: 480 };
    await открыть();
    const предупреждение = container.querySelector(".banner.warn")?.textContent ?? "";
    expect(предупреждение).toContain("другой моделью эмбеддингов, а выбрана «bge-m3»");
    expect(предупреждение).toContain("вернётесь к старой модели — считать придётся заново");
    expect(container.textContent).toContain("Документ · ждёт индексации · 480 кусков · посчитан другой моделью");
  });

  it("без чужих векторов не пугает", async () => {
    await открыть();
    expect(container.querySelector(".banner.warn")).toBeNull();
  });
});

describe("документы на экране", () => {
  async function выбрать(файл: File) {
    const поле = container.querySelector<HTMLInputElement>("input[type=file]")!;
    Object.defineProperty(поле, "files", { value: [файл], configurable: true });
    await act(async () => {
      поле.dispatchEvent(new Event("change", { bubbles: true }));
      // FileReader отдаёт байты отдельной задачей — ждём её
      await new Promise((resolve) => setTimeout(resolve, 20));
    });
  }

  it("загрузка отправляет байты файла и перечитывает состояние", async () => {
    await открыть();
    await выбрать(new File([new Uint8Array([0xd0, 0xe5, 0xe3])], "Регламент.txt"));
    expect(загружено).toEqual([{ filename: "Регламент.txt", bytes: [0xd0, 0xe5, 0xe3] }]);
    expect(спрошено).toHaveLength(2);
  });

  it("отказ сервера при загрузке показывается", async () => {
    отказЗагрузки = new Error("Пока принимаются только файлы .txt и .md.");
    await открыть();
    await выбрать(new File(["%PDF"], "договор.pdf"));
    expect(container.querySelector(".banner.error")?.textContent).toBe("Пока принимаются только файлы .txt и .md.");
  });

  it("удаление — только у документов и только после подтверждения", async () => {
    await открыть();
    const удалить = Array.from(container.querySelectorAll("button")).filter((b) => b.textContent === "Удалить");
    expect(удалить).toHaveLength(1);  // встречи удаляются в истории, здесь — только документ
    const подтверждение = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true);

    await act(async () => удалить[0].click());
    expect(удалено).toEqual([]);
    await act(async () => удалить[0].click());
    expect(удалено).toEqual([3]);
    подтверждение.mockRestore();
  });
});

describe("оценка времени", () => {
  it("до начала — по замеру, дальше — по тому, как идёт на самом деле", () => {
    expect(remainingSeconds(0, 480, 0)).toBe(120);            // 4 куска/с по замеру
    expect(remainingSeconds(100, 480, 50_000)).toBe(190);     // идёт 2 куска/с — медленнее замера
  });

  it("число со словом склоняется — «2 кусков» на экране выглядело неряшливо", () => {
    expect([1, 2, 5, 11, 21, 22].map((n) => plural(n, ["кусок", "куска", "кусков"]))).toEqual([
      "1 кусок", "2 куска", "5 кусков", "11 кусков", "21 кусок", "22 куска",
    ]);
  });

  it("минуты и часы читаются человеком", () => {
    expect(formatDuration(30)).toBe("меньше минуты");
    expect(formatDuration(400)).toBe("около 7 мин");
    expect(formatDuration(4000)).toBe("около 1 ч 7 мин");
  });
});
