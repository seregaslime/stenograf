import { afterEach, describe, expect, it, vi } from "vitest";

import { LlmError, OllamaClient } from "./ollama";

afterEach(() => vi.unstubAllGlobals());

function stub(handler: (url: string, init?: RequestInit) => unknown) {
  const mock = vi.fn(async (url: string, init?: RequestInit) => handler(url, init) as Response);
  vi.stubGlobal("fetch", mock);
  return mock;
}

function ok(body: unknown) {
  return { ok: true, status: 200, json: async () => body } as unknown as Response;
}

const клиент = () => new OllamaClient({ url: "http://192.168.3.10:11434/" });

describe("OllamaClient.generate", () => {
  it("отдаёт текст ответа и не шлёт num_predict", async () => {
    // num_predict у qwen3 съедают размышления, и ответ приходит пустым —
    // грабли, на которые в проекте уже наступали
    const mock = stub(() => ok({ response: "Демо во вторник." }));
    await expect(клиент().generate("qwen3:4b", "вопрос")).resolves.toBe("Демо во вторник.");

    const тело = JSON.parse(mock.mock.calls[0]![1]!.body as string);
    expect(тело.options.num_predict).toBeUndefined();
    expect(тело.think).toBeUndefined(); // отключать размышления тоже нельзя
    expect(тело.stream).toBe(false);
  });

  it("вырезает размышления, если модель вывалила их в текст", async () => {
    stub(() => ok({ response: "<think>надо посчитать</think>Ответ." }));
    await expect(клиент().generate("qwen3:4b", "в")).resolves.toBe("Ответ.");
  });

  it("режет хвостовой слэш в адресе, а не склеивает двойной", async () => {
    const mock = stub(() => ok({ response: "" }));
    await клиент().generate("m", "p");
    expect(mock.mock.calls[0]![0]).toBe("http://192.168.3.10:11434/api/generate");
  });

  it("на недоступный адрес объясняет, что делать", async () => {
    stub(() => {
      throw new TypeError("Failed to fetch");
    });
    await expect(клиент().generate("m", "p")).rejects.toThrow(/Проверьте адрес/);
  });

  it("на 404 подсказывает скачать именно эту модель", async () => {
    stub(() => ({ ok: false, status: 404, text: async () => "" }) as unknown as Response);
    await expect(клиент().generate("qwen3:8b", "p")).rejects.toThrow(
      "Модель «qwen3:8b» не найдена. Скачайте её: ollama pull qwen3:8b",
    );
  });
});

describe("OllamaClient.embed", () => {
  it("шлёт всю пачку одним запросом", async () => {
    const mock = stub(() => ok({ embeddings: [[1], [2], [3]] }));
    await expect(клиент().embed("bge-m3", ["а", "б", "в"])).resolves.toHaveLength(3);
    expect(mock).toHaveBeenCalledTimes(1);
  });

  it("пустой список не ходит в сеть", async () => {
    const mock = stub(() => ok({}));
    await expect(клиент().embed("bge-m3", [])).resolves.toEqual([]);
    expect(mock).not.toHaveBeenCalled();
  });

  it("замечает, что модель не для эмбеддингов", async () => {
    stub(() => ok({ embeddings: [[1]] }));
    await expect(клиент().embed("qwen3:4b", ["а", "б"])).rejects.toThrow(LlmError);
  });
});

describe("OllamaClient.models", () => {
  it("возвращает имена скачанных моделей", async () => {
    stub(() => ok({ models: [{ name: "qwen3:4b" }, { name: "bge-m3:latest" }] }));
    await expect(клиент().models()).resolves.toEqual(["qwen3:4b", "bge-m3:latest"]);
  });

  it("недоступный адрес — пустой список, а не исключение", async () => {
    // Список моделей спрашивают на экране настроек, пока человек печатает адрес:
    // сыпать исключениями на каждый недонабранный адрес неправильно
    stub(() => {
      throw new TypeError("Failed to fetch");
    });
    await expect(клиент().models()).resolves.toEqual([]);
  });
});
