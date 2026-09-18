/**
 * Один хозяин у индексации. Сама индексация подменена: здесь проверяется не
 * счёт векторов (он в search.test.ts), а то, что запуск ровно один и что его
 * видно со стороны — из любого места приложения.
 */
import { act } from "react";
import { createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  autoIndex,
  indexState,
  isIndexing,
  setMeetingLive,
  startIndexing,
  subscribe,
  useAutoIndexing,
} from "./indexing";
import type { IndexProgress } from "./search";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let индексация: (onProgress: (p: IndexProgress) => void) => Promise<number> = async () => 0;
let запусков = 0;

let сводокСпрошено = 0;
let сводка = {
  model: "bge-m3",
  meetings: [{ id: 1, title: "Планёрка", status: "waiting", chunks: 0, chunks_waiting: 8, chunks_other_models: 0, started_at: null }],
  documents: [],
};
vi.mock("../api/rest", () => ({
  api: {
    searchPending: vi.fn(), searchIndex: vi.fn(), searchQuery: vi.fn(),
    knowledgeStatus: vi.fn(async () => {
      сводокСпрошено += 1;
      return сводка;
    }),
  },
}));
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
  сводка = {
    model: "bge-m3",
    meetings: [{ id: 1, title: "Планёрка", status: "waiting", chunks: 0, chunks_waiting: 8, chunks_other_models: 0, started_at: null }],
    documents: [],
  };
  запусков = 0;
  сводокСпрошено = 0;
  индексация = async () => 0;
  setMeetingLive(false);
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

  it("сама индексация во время встречи не начинается", async () => {
    // Эмбеддинги и подсказки считает одна Ollama, очередь к ней одна: документ
    // на сотню кусков занимает её на полминуты, и подсказка придёт после него
    setMeetingLive(true);
    await autoIndex("bge-m3");
    expect(запусков).toBe(0);
    expect(isIndexing()).toBe(false);
    expect(сводокСпрошено).toBe(0);  // и сервер во время встречи не тревожим
  });

  it("после встречи считает", async () => {
    setMeetingLive(true);
    await autoIndex("bge-m3");
    setMeetingLive(false);
    const запуск = медленная(() => 4);
    await autoIndex("bge-m3");
    запуск.отпустить();
    await new Promise((r) => setTimeout(r, 0));
    expect(запусков).toBe(1);
  });

  it("без выбранной модели эмбеддингов сама не лезет к Ollama", async () => {
    await autoIndex("");
    expect(запусков).toBe(0);
  });

  it("считать нечего — к модели не обращается", async () => {
    сводка = { ...сводка, meetings: [] };
    await autoIndex("bge-m3");
    expect(запусков).toBe(0);
  });

  it("пересчёт после смены модели сама не начинает — это решение человека", async () => {
    // Он стоит столько же, сколько индексация всей базы с нуля, и на экране
    // «База знаний» человеку показано, сколько это займёт. Тихо занять Ollama
    // на полчаса вместо него — ровно то «зависание», от которого экран и спасал
    сводка = {
      ...сводка,
      meetings: [{ ...сводка.meetings[0], chunks_other_models: 8 }],
    };
    await autoIndex("qwen3-embedding:0.6b");
    expect(запусков).toBe(0);
  });

  it("просьбу человека встреча не отменяет: он видит, чего просит", async () => {
    setMeetingLive(true);
    const запуск = медленная(() => 2);
    const обещание = startIndexing("bge-m3");
    запуск.отпустить();
    expect(await обещание).toBe(2);
    expect(запусков).toBe(1);
  });

  /** Приложение с включённым автозапуском: те же два признака, что в App.
   *  Имя латиницей не от хорошей жизни: eslint признаёт компонентом только имя
   *  с латинской заглавной буквы, а с кириллической «К» ругается, что хук
   *  вызван вне компонента. */
  function приложение(серверНаСвязи: boolean, идётВстреча: boolean) {
    return createElement(function Root() {
      useAutoIndexing(серверНаСвязи, идётВстреча);
      return null;
    });
  }

  it("приложение досчитывает само, как только сервер отозвался", async () => {
    const узел = document.createElement("div");
    let корень: Root;
    await act(async () => {
      корень = createRoot(узел);
      корень.render(приложение(false, false));      // сервер ещё молчит
    });
    expect(запусков).toBe(0);

    await act(async () => корень.render(приложение(true, false)));
    expect(запусков).toBe(1);
    await act(async () => корень!.unmount());
  });

  it("во время встречи не считает, а по её окончании берётся сам", async () => {
    const узел = document.createElement("div");
    let корень: Root;
    await act(async () => {
      корень = createRoot(узел);
      корень.render(приложение(true, true));        // идёт встреча
    });
    expect(запусков).toBe(0);
    await autoIndex("bge-m3");                      // и загрузка документа тоже ждёт
    expect(запусков).toBe(0);

    await act(async () => корень!.render(приложение(true, false)));  // встреча кончилась
    expect(запусков).toBe(1);
    await act(async () => корень!.unmount());
  });
});
