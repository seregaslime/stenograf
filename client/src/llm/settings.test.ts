import { afterEach, describe, expect, it } from "vitest";

import { llmReady, loadLlmSettings, saveLlmSettings } from "./settings";
import { setSetting } from "../store";

afterEach(() => localStorage.clear());

describe("настройки моделей у клиента", () => {
  it("без сохранённого — локальная модель на этой же машине", () => {
    const s = loadLlmSettings();
    expect(s.provider).toBe("local");
    expect(s.ollamaUrl).toBe("http://127.0.0.1:11434");
    expect(s.localSummaryModel).toBe("qwen3:4b");
  });

  it("сохранённое возвращается как есть", () => {
    saveLlmSettings({
      provider: "api",
      apiBaseUrl: "https://api.groq.com/openai/v1",
      apiKey: "ключ",
      apiSummaryModel: "gpt-oss-120b",
      tpmLimits: { "gpt-oss-120b": 8000 },
    });
    const s = loadLlmSettings();
    expect(s.provider).toBe("api");
    expect(s.apiKey).toBe("ключ");
    expect(s.tpmLimits).toEqual({ "gpt-oss-120b": 8000 });
  });

  it("частичное сохранение не стирает остальное", () => {
    saveLlmSettings({ apiKey: "ключ", apiBaseUrl: "адрес" });
    saveLlmSettings({ provider: "api" });
    expect(loadLlmSettings().apiKey).toBe("ключ");
  });

  it("мусор в лимитах не ломает настройки", () => {
    // Настройки должны открываться, даже когда в хранилище невесть что
    setSetting("llm.tpmLimits", "{это не json");
    expect(loadLlmSettings().tpmLimits).toEqual({});
  });

  it("нечисловые лимиты отбрасываются", () => {
    setSetting("llm.tpmLimits", '{"a":8000,"b":"много"}');
    expect(loadLlmSettings().tpmLimits).toEqual({ a: 8000 });
  });
});

describe("llmReady", () => {
  it("локальная модель готова с адресом и именем", () => {
    expect(llmReady(loadLlmSettings())).toBe(true);
  });

  it("локальная без адреса — не готова", () => {
    saveLlmSettings({ ollamaUrl: "" });
    expect(llmReady(loadLlmSettings())).toBe(false);
  });

  it("внешний API без ключа — не готов", () => {
    saveLlmSettings({ provider: "api", apiBaseUrl: "адрес", apiSummaryModel: "модель" });
    expect(llmReady(loadLlmSettings())).toBe(false);
  });

  it("внешний API со всем нужным — готов", () => {
    saveLlmSettings({
      provider: "api",
      apiBaseUrl: "адрес",
      apiKey: "ключ",
      apiSummaryModel: "модель",
    });
    expect(llmReady(loadLlmSettings())).toBe(true);
  });
});
