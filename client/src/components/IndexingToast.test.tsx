/**
 * Окошко про индексацию в углу. Сама индексация подменена: проверяется, когда
 * окошко видно и что в нём написано.
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { IndexProgress } from "../llm/search";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let индексация: (onProgress: (p: IndexProgress) => void) => Promise<number> = async () => 0;

vi.mock("../api/rest", () => ({ api: { searchPending: vi.fn(), searchIndex: vi.fn(), searchQuery: vi.fn() } }));
vi.mock("../llm/settings", () => ({ loadLlmSettings: () => ({ embedModel: "bge-m3" }) }));
vi.mock("../llm/search", () => ({
  indexPending: vi.fn((_api: unknown, _s: unknown, _m: string, onProgress: (p: IndexProgress) => void) =>
    индексация(onProgress)),
}));

const { startIndexing } = await import("../llm/indexing");
const { default: IndexingToast } = await import("./IndexingToast");

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

/** Индексация, которую тест отпускает сам: так ловится «идёт прямо сейчас». */
function медленная() {
  let отпустить = () => {};
  const дождались = new Promise<void>((resolve) => {
    отпустить = resolve;
  });
  let шаг: (p: IndexProgress) => void = () => {};
  индексация = async (onProgress) => {
    шаг = onProgress;
    await дождались;
    return 100;
  };
  return { отпустить: () => отпустить(), шаг: (p: IndexProgress) => шаг(p) };
}

const показать = (наЭкранеБазы = false) =>
  act(async () => root.render(<IndexingToast наЭкранеБазы={наЭкранеБазы} />));

describe("окошко про индексацию", () => {
  it("пока ничего не считается, окошка нет", async () => {
    await показать();
    expect(container.querySelector(".toast")).toBeNull();
  });

  it("во время индексации говорит, что и сколько осталось", async () => {
    const запуск = медленная();
    await показать();
    let обещание!: Promise<number>;
    await act(async () => {
      обещание = startIndexing("bge-m3", 100);
      запуск.шаг({ chunksDone: 40, chunksTotal: 100, source: "Регламент отдела" });
    });

    const окошко = container.querySelector(".toast");
    expect(окошко?.textContent).toContain("«Регламент отдела»");
    expect(окошко?.textContent).toContain("кусок 40 из 100");
    expect(окошко?.textContent).toContain("осталось");

    запуск.отпустить();
    await act(async () => {
      await обещание;
    });
  });

  it("на экране базы знаний не мешает: там своя полоса с подробностями", async () => {
    const запуск = медленная();
    await показать(true);
    let обещание!: Promise<number>;
    await act(async () => {
      обещание = startIndexing("bge-m3", 100);
      запуск.шаг({ chunksDone: 40, chunksTotal: 100, source: "Регламент отдела" });
    });

    expect(container.querySelector(".toast")).toBeNull();

    запуск.отпустить();
    await act(async () => {
      await обещание;
    });
  });

  it("кончилась индексация — окошко ушло само", async () => {
    const запуск = медленная();
    await показать();
    let обещание!: Promise<number>;
    await act(async () => {
      обещание = startIndexing("bge-m3", 100);
      запуск.шаг({ chunksDone: 40, chunksTotal: 100, source: "Регламент отдела" });
    });
    expect(container.querySelector(".toast")).not.toBeNull();

    запуск.отпустить();
    await act(async () => {
      await обещание;
    });
    expect(container.querySelector(".toast")).toBeNull();
  });
});
