/**
 * Выбор модели по роли: протокол и подсказки могут считаться разными моделями,
 * а вызывающий код не должен знать, какая сейчас активна и чем она отличается.
 *
 * Перенесено с сервера (llm/router.py). Числа бюджетов не тронуты — они
 * подбирались под контекст локальной модели и под тарифные лимиты провайдеров.
 */
import { OllamaClient } from "./ollama";
import { OpenAiClient, type Delta } from "./openai";

export type Provider = "local" | "api";
export type Role = "summary" | "hints";

export interface LlmSettings {
  provider: Provider;
  /** Локальная модель: адрес сервера моделей и модели по ролям. */
  ollamaUrl: string;
  localSummaryModel: string;
  localHintsModel: string;
  /** Внешний API: адрес, ключ, модели по ролям. */
  apiBaseUrl: string;
  apiKey: string;
  apiSummaryModel: string;
  apiHintsModel: string;
  /** Измеренные лимиты токенов в минуту по моделям. */
  tpmLimits?: Record<string, number>;
  keepAlive?: string;
}

/** Сколько контекста не жалко отдать модели и насколько подробный промпт. */
export interface Budget {
  /** Потолок транскрипта в символах. 0 — без ограничения. */
  summaryChars: number;
  hintsChars: number;
  /** Развёрнутый промпт: больше секций и примеров. */
  detailed: boolean;
  /** Потолок одного запроса протокола в токенах. 0 — тарифного лимита нет. */
  summaryTokens: number;
}

/** У локальной модели окно маленькое, у API — большое. */
const LOCAL_SUMMARY_CHARS = 12_000;
const API_SUMMARY_CHARS = 200_000;
const LOCAL_HINTS_CHARS = 2_500;
const API_HINTS_CHARS = 40_000;

/** Запасной лимит, если измерить не вышло, и доля, оставляемая модели на ответ. */
const TPM_FALLBACK = 6_000;
const OUTPUT_SHARE = 0.25;

/** Символов на токен для русского текста — грубая, но проверенная оценка. */
export const CHARS_PER_TOKEN = 2.5;

export class LlmRouter {
  constructor(private settings: LlmSettings) {}

  get provider(): Provider {
    return this.settings.provider;
  }

  modelFor(role: Role): string {
    const s = this.settings;
    if (s.provider === "api") {
      return role === "summary" ? s.apiSummaryModel : s.apiHintsModel;
    }
    return role === "summary" ? s.localSummaryModel : s.localHintsModel;
  }

  /**
   * Бюджет считается на каждый вызов, а не в конструкторе: провайдера могут
   * переключить между встречей и пересозданием протокола, и глубина промпта
   * должна поменяться сразу.
   */
  get budget(): Budget {
    if (this.settings.provider !== "api") {
      return {
        summaryChars: LOCAL_SUMMARY_CHARS,
        hintsChars: LOCAL_HINTS_CHARS,
        detailed: false,
        summaryTokens: 0, // тарифного лимита нет, режем по символам
      };
    }
    return {
      summaryChars: API_SUMMARY_CHARS,
      hintsChars: API_HINTS_CHARS,
      detailed: true,
      summaryTokens: this.tpmLimit(this.settings.apiSummaryModel),
    };
  }

  /**
   * Сколько токенов можно ОТПРАВИТЬ модели за минуту. Из минутного лимита
   * вычитаем резерв под ответ: провайдер засчитывает в лимит и его тоже. Без
   * вычета запрос на 5400 токенов при лимите 8000 получал отказ
   * «Requested 8476» — недостающие три тысячи и были местом под ответ.
   */
  tpmLimit(model: string): number {
    const limit = this.settings.tpmLimits?.[model] || TPM_FALLBACK;
    return limit <= 0 ? 0 : Math.trunc(limit * (1 - OUTPUT_SHARE));
  }

  /** Запрос к активной модели в роли. onDelta работает только у внешнего API. */
  async generate(
    role: Role,
    prompt: string,
    { system, temperature = 0.4, onDelta }: {
      system?: string; temperature?: number; onDelta?: Delta;
    } = {},
  ): Promise<string> {
    const model = this.modelFor(role);
    if (this.settings.provider === "api") {
      const client = new OpenAiClient({
        baseUrl: this.settings.apiBaseUrl,
        apiKey: this.settings.apiKey,
      });
      return client.generate(model, prompt, { system, temperature, onDelta });
    }
    // У локальной модели свой протокол, потокового чтения там нет: вызывающий
    // просто получит ответ целиком, как раньше получал от сервера.
    const client = new OllamaClient({
      url: this.settings.ollamaUrl,
      keepAlive: this.settings.keepAlive,
    });
    return client.generate(model, prompt, { system, temperature });
  }
}
