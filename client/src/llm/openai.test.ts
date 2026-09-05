import { afterEach, describe, expect, it, vi } from "vitest";

import { LlmError } from "./ollama";
import { OpenAiClient, parseReset } from "./openai";

afterEach(() => vi.unstubAllGlobals());

const GROQ = "https://api.groq.com/openai/v1";

function ответ(body: unknown, { status = 200, headers = {} } = {}): Response {
  return {
    ok: status < 400,
    status,
    statusText: "",
    headers: new Headers(headers),
    json: async () => body,
  } as unknown as Response;
}

/** Поток событий провайдера: строки SSE одним куском. */
function поток(строки: string[], headers = {}): Response {
  const данные = new TextEncoder().encode(строки.join("\n") + "\n");
  let отдано = false;
  return {
    ok: true,
    status: 200,
    headers: new Headers(headers),
    body: {
      getReader: () => ({
        read: async () =>
          отдано ? { done: true, value: undefined } : ((отдано = true), { done: false, value: данные }),
      }),
    },
  } as unknown as Response;
}

const клиент = (opts = {}) => new OpenAiClient({ baseUrl: GROQ + "/", apiKey: "ключ", ...opts });

/** Мок fetch с объявленными параметрами: без них у него пустой тип аргументов,
 *  и обращение к mock.calls[0][1] не проходит проверку типов. */
function перехват(ответить: () => Response) {
  const mock = vi.fn(async (_url: string, _init?: RequestInit) => ответить());
  vi.stubGlobal("fetch", mock);
  return mock;
}

type Перехват = ReturnType<typeof перехват>;
const адрес = (m: Перехват) => m.mock.calls[0]![0];
const заголовки = (m: Перехват) => m.mock.calls[0]![1]!.headers as Record<string, string>;
const телоЗапроса = (m: Перехват) => JSON.parse(String(m.mock.calls[0]![1]!.body));

describe("parseReset", () => {
  it.each([
    ["1.2s", 1.2],
    ["120ms", 0.12],
    ["2m59.56s", 179.56],
    ["", 0],
    [null, 0],
  ])("«%s» → %s секунд", (raw, ожидание) => {
    expect(parseReset(raw as string | null)).toBeCloseTo(ожидание, 2);
  });
});

describe("OpenAiClient.generate", () => {
  it("шлёт ключ заголовком и возвращает текст", async () => {
    const mock = перехват(() => ответ({ choices: [{ message: { content: " Готово. " } }] }));

    await expect(клиент().generate("gpt-oss", "вопрос")).resolves.toBe("Готово.");
    expect(заголовки(mock).Authorization).toBe("Bearer ключ");
    expect(адрес(mock)).toBe(`${GROQ}/chat/completions`);
  });

  it("системный промпт уходит отдельным сообщением", async () => {
    const mock = перехват(() => ответ({ choices: [{ message: { content: "х" } }] }));
    await клиент().generate("m", "вопрос", { system: "ты помощник" });
    expect(телоЗапроса(mock).messages).toEqual([
      { role: "system", content: "ты помощник" },
      { role: "user", content: "вопрос" },
    ]);
  });

  it("причину отказа берёт из error.message, а не из кода", async () => {
    vi.stubGlobal("fetch", vi.fn(async () =>
      ответ({ error: { message: "Limit 8000, Requested 16324" } }, { status: 429 })));
    await expect(клиент().generate("m", "п")).rejects.toThrow(/Limit 8000, Requested 16324/);
  });

  it("недоступный адрес объясняет словами", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    await expect(клиент().generate("m", "п")).rejects.toThrow(LlmError);
  });
});

describe("OpenAiClient.generate потоком", () => {
  it("разделяет мысли и ответ, склеивает текст", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => поток([
      'data: {"choices":[{"delta":{"reasoning":"думаю…"}}]}',
      'data: {"choices":[{"delta":{"content":"Демо "}}]}',
      'data: {"choices":[{"delta":{"content":"во вторник."}}]}',
      "data: [DONE]",
    ])));
    const куски: [string, string][] = [];
    const текст = await клиент().generate("m", "п", {
      onDelta: (kind, chunk) => куски.push([kind, chunk]),
    });
    expect(текст).toBe("Демо во вторник.");
    expect(куски).toEqual([
      ["reasoning", "думаю…"],
      ["content", "Демо "],
      ["content", "во вторник."],
    ]);
  });

  it("служебные строки потока не ломают разбор", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => поток([
      ": ping",
      "data: не-json",
      'data: {"choices":[{"delta":{"content":"ок"}}]}',
      "data: [DONE]",
    ])));
    await expect(клиент().generate("m", "п", { onDelta: () => {} })).resolves.toBe("ок");
  });
});

describe("OpenAiClient.models", () => {
  it("отсеивает не-текстовые и слишком короткий контекст", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ответ({ data: [
      { id: "gpt-oss-120b", context_window: 131072 },
      { id: "whisper-large", input_modalities: ["audio"], output_modalities: ["text"] },
      { id: "prompt-guard", context_window: 512 },
      { id: "без-полей" },
    ] })));
    const модели = (await клиент().models()).map((m) => m.id);
    // «без-полей» остаётся: судить не по чему, а молча урезать список опаснее
    expect(модели).toEqual(["gpt-oss-120b", "без-полей"]);
  });
});

describe("OpenAiClient.tokenLimit", () => {
  it("берёт лимит из заголовка ответа", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ответ({}, {
      headers: { "x-ratelimit-limit-tokens": "8000" },
    })));
    await expect(клиент().tokenLimit("m")).resolves.toBe(8000);
  });

  it("сетевая ошибка не ломает сохранение настроек", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("сеть"); }));
    await expect(клиент().tokenLimit("m")).resolves.toBeNull();
  });
});

describe("OpenAiClient.waitForBudget", () => {
  it("ждёт ровно столько, сколько обещал провайдер, плюс секунду", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ответ({}, {
      headers: { "x-ratelimit-remaining-tokens": "100", "x-ratelimit-reset-tokens": "5s" },
    })));
    const c = клиент();
    await c.tokenLimit("m"); // запоминаем остаток из заголовков

    const паузы: number[] = [];
    await c.waitForBudget(5000, async (ms) => { паузы.push(ms); });
    expect(паузы).toHaveLength(1);
    expect(паузы[0]).toBeGreaterThan(5000);
    expect(паузы[0]).toBeLessThan(6500);
  });

  it("влезающий запрос не ждёт вовсе", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ответ({}, {
      headers: { "x-ratelimit-remaining-tokens": "9000", "x-ratelimit-reset-tokens": "5s" },
    })));
    const c = клиент();
    await c.tokenLimit("m");
    const паузы: number[] = [];
    await c.waitForBudget(100, async (ms) => { паузы.push(ms); });
    expect(паузы).toEqual([]);
  });
});

describe("ожидание лимита перед запросом", () => {
  it("запрос дожидается восстановления окна, а не получает отказ", async () => {
    // Сервер ждал столько, сколько провайдер обещал в заголовках. При переносе
    // это едва не потерялось: функция ожидания была написана и никем не вызвана.
    const mock = перехват(() => ответ(
      { choices: [{ message: { content: "ответ" } }] },
      { headers: { "x-ratelimit-remaining-tokens": "10", "x-ratelimit-reset-tokens": "3s" } },
    ));
    const c = new OpenAiClient({ baseUrl: GROQ, apiKey: "ключ", tpmLimit: 8000 });

    await c.generate("m", "первый запрос");   // из ответа узнали: осталось 10 токенов
    const паузы: number[] = [];
    vi.spyOn(c, "waitForBudget").mockImplementation(async (need, _sleep) => {
      паузы.push(need);
    });
    await c.generate("m", "второй запрос подлиннее, чтобы точно не влез в остаток");

    expect(паузы).toHaveLength(1);
    // Цена запроса — вход плюс резерв под ответ (провайдер списывает и его)
    expect(паузы[0]).toBeGreaterThan(8000 * 0.25);
    expect(mock).toHaveBeenCalledTimes(2);
  });

  it("ждём ровно то, что обещал провайдер", async () => {
    перехват(() => ответ({}, {
      headers: { "x-ratelimit-remaining-tokens": "5", "x-ratelimit-reset-tokens": "4s" },
    }));
    const c = new OpenAiClient({ baseUrl: GROQ, apiKey: "ключ" });
    await c.tokenLimit("m");

    const паузы: number[] = [];
    await c.waitForBudget(9999, async (ms) => { паузы.push(ms); });
    expect(паузы[0]).toBeGreaterThan(4000);
    expect(паузы[0]).toBeLessThan(5500);
  });
});
