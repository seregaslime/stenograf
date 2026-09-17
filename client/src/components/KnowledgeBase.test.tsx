/** База знаний на странице истории: загрузка файла, список, удаление.
 *
 *  Сервер подменён: проверяется, что приложение отправляет файл байтами (а не
 *  текстом, перечитанным в UTF-8) и показывает то, что вернул сервер.
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { DocumentDto } from "../types";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// В jsdom у File нет arrayBuffer(), в браузере и Electron он есть. Досказываем
// его тестовой среде через FileReader, а не меняем код приложения под jsdom.
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

let список: DocumentDto[] = [];
const загружено: { filename: string; content: string }[] = [];
let отказ: Error | null = null;

vi.mock("../api/rest", () => ({
  api: {
    documents: vi.fn(async () => список),
    uploadDocument: vi.fn(async (filename: string, content: string) => {
      if (отказ) throw отказ;
      загружено.push({ filename, content });
      const документ = { id: список.length + 1, title: filename.replace(/\.\w+$/, ""), created_at: "2026-09-17T10:00:00Z", chars: 42 };
      список = [документ, ...список];
      return документ;
    }),
    deleteDocument: vi.fn(async (id: number) => {
      список = список.filter((д) => д.id !== id);
      return { deleted: id };
    }),
  },
}));

const { default: KnowledgeBase, toBase64 } = await import("./KnowledgeBase");

let container: HTMLDivElement;
let root: Root;

beforeEach(async () => {
  список = [];
  загружено.length = 0;
  отказ = null;
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => root.render(<KnowledgeBase />));
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.restoreAllMocks();
});

async function выбрать(файл: File) {
  const поле = container.querySelector<HTMLInputElement>("input[type=file]")!;
  Object.defineProperty(поле, "files", { value: [файл], configurable: true });
  await act(async () => {
    поле.dispatchEvent(new Event("change", { bubbles: true }));
    // FileReader отдаёт байты отдельной задачей, а не микрозадачей — ждём её
    await new Promise((resolve) => setTimeout(resolve, 20));
  });
}

describe("base64 для файла", () => {
  it("байты cp1251 уходят как есть — кодировку распознаёт сервер", () => {
    const байты = new Uint8Array([0xd0, 0xe5, 0xe3]);  // «Рег» в cp1251
    expect(Array.from(atob(toBase64(байты)), (с) => с.charCodeAt(0))).toEqual([0xd0, 0xe5, 0xe3]);
  });

  it("мегабайтный файл не упирается в предел аргументов функции", () => {
    const байты = new Uint8Array(1_000_000).fill(65);
    expect(atob(toBase64(байты))).toHaveLength(1_000_000);
  });
});

describe("база знаний", () => {
  it("загруженный документ появляется в списке", async () => {
    await выбрать(new File([new Uint8Array([0xd0, 0xe5, 0xe3])], "Регламент.txt"));
    expect(загружено).toEqual([{ filename: "Регламент.txt", content: "0OXj" }]);
    expect(container.textContent).toContain("Регламент");
    expect(container.textContent).toContain("42 символов");
  });

  it("отказ сервера показывается, а не глотается", async () => {
    отказ = new Error("Пока принимаются только файлы .txt и .md.");
    await выбрать(new File(["%PDF"], "договор.pdf"));
    expect(container.querySelector(".banner.error")?.textContent).toBe("Пока принимаются только файлы .txt и .md.");
  });

  it("удаление спрашивает подтверждение и убирает документ из списка", async () => {
    await выбрать(new File(["текст"], "Черновик.md"));
    vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true);
    const кнопка = () => Array.from(container.querySelectorAll("button")).find((b) => b.textContent === "Удалить")!;

    await act(async () => кнопка().click());
    expect(container.textContent).toContain("Черновик");  // передумал — документ на месте

    await act(async () => кнопка().click());
    expect(container.textContent).not.toContain("Черновик");
  });
});
