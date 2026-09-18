/**
 * Один хозяин у индексации. Сама индексация подменена: здесь проверяется не
 * счёт векторов (он в search.test.ts), а то, что запуск ровно один и что его
 * видно со стороны — из любого места приложения.
 */
import { afterEach, describe, expect, it, vi } from "vitest";

import { indexState, isIndexing, startIndexing, subscribe } from "./indexing";
import type { IndexProgress } from "./search";

let индексация: (onProgress: (p: IndexProgress) => void) => Promise<number> = async () => 0;
let запусков = 0;

vi.mock("../api/rest", () => ({ api: { searchPending: vi.fn(), searchIndex: vi.fn(), searchQuery: vi.fn() } }));
vi.mock("./settings", () => ({ loadLlmSettings: () => ({ embedModel: "bge-m3" }) }));
vi.mock("./search", () => ({
  indexPending: vi.fn((_api: unknown, _s: unknown, _m: string, onProgress: (p: IndexProgress) => void) => {
    запусков += 1;
    return индексация(onProgress);
  }),
}));

/** Индексация, которую тест отпускает сам: так ловится «идёт прямо сейчас». */
function медленная(результат: () => Promise<number> | number) {
  let отпустить = () => {};
  const дождались = new Promise<void>((resolve) => {
    отпустить = resolve;
  });
  let шаг: (p: IndexProgress) => void = () => {};
  индексация = async (onProgress) => {
    шаг = onProgress;
    await дождались;
    return результат();
  };
  return { отпустить: () => отпустить(), шаг: (p: IndexProgress) => шаг(p) };
}

afterEach(() => {
  запусков = 0;
  индексация = async () => 0;
});

describe("один хозяин у индексации", () => {
  it("второй вызов не начинает вторую индексацию, а ждёт идущую", async () => {
    const запуск = медленная(() => 7);
    const первый = startIndexing("bge-m3", 10);
    const второй = startIndexing("bge-m3", 10);

    expect(запусков).toBe(1);          // сервер не спрошен второй раз
    expect(первый).toBe(второй);       // и ждут оба одного и того же
    запуск.отпустить();
    expect(await первый).toBe(7);
    expect(await второй).toBe(7);
  });

  it("после окончания можно запустить снова", async () => {
    const первый = медленная(() => 1);
    const обещание = startIndexing("bge-m3");
    первый.отпустить();
    await обещание;
    expect(isIndexing()).toBe(false);

    const второй = медленная(() => 2);
    const ещё = startIndexing("bge-m3");
    второй.отпустить();
    expect(await ещё).toBe(2);
    expect(запусков).toBe(2);
  });

  it("прогресс виден подписчику, а по окончании гаснет", async () => {
    const запуск = медленная(() => 3);
    const снимки: (IndexProgress | null)[] = [];
    const отписаться = subscribe(() => снимки.push(indexState().progress));

    const обещание = startIndexing("bge-m3", 3);
    запуск.шаг({ chunksDone: 1, chunksTotal: 3, source: "Регламент" });
    запуск.отпустить();
    await обещание;

    expect(снимки[0]).toEqual({ chunksDone: 0, chunksTotal: 3, source: "" });  // подсказка из сводки
    expect(снимки[1]).toEqual({ chunksDone: 1, chunksTotal: 3, source: "Регламент" });
    expect(снимки.at(-1)).toBeNull();
    отписаться();
  });

  it("подписчик может запустить следующую индексацию прямо в конце прошлой", async () => {
    // Так будет устроен автозапуск: он узнаёт об окончании и решает, не пора ли
    // считать дальше. Если признак «идёт» снять после рассылки, а не до, такой
    // подписчик получит уже завершённое обещание и решит, что всё посчитано.
    const первый = медленная(() => 1);
    let ответ: number | null = null;
    const отписаться = subscribe(() => {
      if (indexState().progress === null && запусков === 1) {
        медленная(() => 2).отпустить();
        void startIndexing("bge-m3").then((с) => {
          ответ = с;
        });
      }
    });

    const обещание = startIndexing("bge-m3");
    первый.отпустить();
    await обещание;
    await new Promise((r) => setTimeout(r, 0));  // дать второму запуску завершиться
    отписаться();

    expect(запусков).toBe(2);
    expect(ответ).toBe(2);
  });

  it("отписавшийся больше не тревожится", async () => {
    const запуск = медленная(() => 1);
    let звонков = 0;
    const отписаться = subscribe(() => {
      звонков += 1;
    });
    отписаться();
    const обещание = startIndexing("bge-m3");
    запуск.отпустить();
    await обещание;
    expect(звонков).toBe(0);
  });

  it("ошибка ложится в состояние, а обещание не отказывает", async () => {
    // Запуск, которого никто не ждёт, отказом обещания уронил бы приложение —
    // а такой запуск есть: индексация умеет начинаться сама
    const запуск = медленная(() => Promise.reject(new Error("Модель недоступна: ollama pull bge-m3")));
    const обещание = startIndexing("bge-m3");
    запуск.отпустить();

    expect(await обещание).toBe(0);
    expect(indexState().progress).toBeNull();
    // Точка дописана: без неё приписка склеилась бы с командой в одно предложение
    expect(indexState().error).toBe(
      "Модель недоступна: ollama pull bge-m3. Уже посчитанное сохранено — можно продолжить.",
    );
    expect(isIndexing()).toBe(false);
  });

  it("новый запуск гасит ошибку прошлого", async () => {
    const упавший = медленная(() => Promise.reject(new Error("Модель недоступна")));
    const первый = startIndexing("bge-m3");
    упавший.отпустить();
    await первый;
    expect(indexState().error).not.toBe("");

    const удачный = медленная(() => 5);
    const второй = startIndexing("bge-m3");
    expect(indexState().error).toBe("");
    удачный.отпустить();
    await второй;
  });
});
