/**
 * Настройки моделей на стороне приложения.
 *
 * Раньше они жили на сервере, и это было неправильно по трём причинам: с
 * сервера деплоя внешний API недоступен вовсе, один ключ на сервере — это один
 * лимит токенов в минуту на всех, а счёт приходит владельцу ключа. Теперь
 * адрес модели, ключ и выбор моделей у каждого свои.
 *
 * Хранилище — то же, где лежит адрес сервера и токен доступа.
 */
import { getSetting, setSetting } from "../store";
import type { LlmSettings, Provider } from "./router";

const ПРЕФИКС = "llm.";

/** Значения по умолчанию: локальная модель на этой же машине. */
const ПО_УМОЛЧАНИЮ: LlmSettings = {
  provider: "local",
  ollamaUrl: "http://127.0.0.1:11434",
  localSummaryModel: "qwen3:4b",
  localHintsModel: "qwen3:1.7b",
  apiBaseUrl: "",
  apiKey: "",
  apiSummaryModel: "",
  apiHintsModel: "",
  embedModel: "bge-m3",
  tpmLimits: {},
};

export function loadLlmSettings(): LlmSettings {
  const строка = (ключ: string, запас: string) => getSetting(ПРЕФИКС + ключ, запас).trim();
  return {
    provider: (строка("provider", ПО_УМОЛЧАНИЮ.provider) as Provider) || "local",
    ollamaUrl: строка("ollamaUrl", ПО_УМОЛЧАНИЮ.ollamaUrl),
    localSummaryModel: строка("localSummaryModel", ПО_УМОЛЧАНИЮ.localSummaryModel),
    localHintsModel: строка("localHintsModel", ПО_УМОЛЧАНИЮ.localHintsModel),
    apiBaseUrl: строка("apiBaseUrl", ""),
    apiKey: строка("apiKey", ""),
    apiSummaryModel: строка("apiSummaryModel", ""),
    apiHintsModel: строка("apiHintsModel", ""),
    embedModel: строка("embedModel", ПО_УМОЛЧАНИЮ.embedModel),
    tpmLimits: разобратьЛимиты(getSetting(ПРЕФИКС + "tpmLimits", "")),
  };
}

export function saveLlmSettings(settings: Partial<LlmSettings>): void {
  for (const [ключ, значение] of Object.entries(settings)) {
    if (значение === undefined) continue;
    setSetting(
      ПРЕФИКС + ключ,
      ключ === "tpmLimits" ? JSON.stringify(значение) : String(значение),
    );
  }
}

/**
 * Измеренные лимиты токенов в минуту. Битое значение — пустой список, а не
 * исключение: настройки должны открываться даже когда в хранилище мусор.
 */
function разобратьЛимиты(raw: string): Record<string, number> {
  if (!raw) return {};
  try {
    const данные = JSON.parse(raw) as Record<string, unknown>;
    return Object.fromEntries(
      Object.entries(данные)
        .filter(([, значение]) => typeof значение === "number")
        .map(([модель, значение]) => [модель, значение as number]),
    );
  } catch {
    return {};
  }
}

/** Настроены ли модели настолько, чтобы вообще пробовать генерацию. */
export function llmReady(settings: LlmSettings = loadLlmSettings()): boolean {
  return settings.provider === "api"
    ? Boolean(settings.apiBaseUrl && settings.apiKey && settings.apiSummaryModel)
    : Boolean(settings.ollamaUrl && settings.localSummaryModel);
}
