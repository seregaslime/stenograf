/**
 * Клиент OpenAI-совместимого API (chat/completions) на стороне приложения.
 *
 * Ключ и адрес живут у человека, а не на сервере: с сервера деплоя внешний API
 * вообще недоступен (провайдер отдаёт 403 по его адресу), а один общий ключ —
 * это один лимит токенов в минуту на всех сразу.
 *
 * Как и клиент локальной модели, не знает ни про DOM, ни про Electron: должен
 * запускаться в Node, иначе сквозные тесты не проверят генерацию.
 */
import { LlmError } from "./ollama";

export interface OpenAiOptions {
  baseUrl: string;
  apiKey?: string;
  /** Минимальный контекст модели, чтобы она попала в список (окно подсказок ~16k). */
  minContextTokens?: number;
  /**
   * Минутный лимит токенов этой модели. Нужен, чтобы перед запросом дождаться
   * восстановления окна вместо отказа провайдера.
   */
  tpmLimit?: number;
}

/** Кусок потокового ответа: текст ответа и ход мыслей идут раздельно. */
export type Delta = (kind: "content" | "reasoning", text: string) => void;

export interface ModelInfo {
  id: string;
  context?: number;
}

/** «1.2s», «120ms», «2m59.56s» → секунды. Формат провайдера, не ISO. */
export function parseReset(raw: string | null): number {
  if (!raw) return 0;
  const factors: Record<string, number> = { ms: 0.001, s: 1, m: 60, h: 3600 };
  let total = 0;
  for (const [, value, unit] of raw.matchAll(/(\d+(?:\.\d+)?)(ms|s|m|h)/g)) {
    total += parseFloat(value) * factors[unit];
  }
  return total;
}

/** Причина отказа человеческими словами: провайдер кладёт её в error.message. */
function detail(body: unknown, fallback: string): string {
  const message = (body as { error?: { message?: string } })?.error?.message;
  return message || fallback.slice(0, 200);
}

const THINK = /<think>[\s\S]*?<\/think>/g;

/** Символов на токен для русского текста — та же оценка, что и в роутере. */
const CHARS_PER_TOKEN = 2.5;

/** Долю минутного лимита провайдер списывает за ответ, даже если модель её не
 *  использует, — резервируем заранее. */
const OUTPUT_SHARE = 0.25;

export class OpenAiClient {
  /**
   * Остаток минутного лимита по данным провайдера. Спать фиксированную минуту
   * недостаточно: окно скользящее, и после крупного запроса за 60 секунд
   * восстанавливается не всё.
   */
  private remaining: number | null = null;
  private resetS = 0;
  private checkedAt = 0;

  constructor(private opts: OpenAiOptions) {}

  private get base(): string {
    return this.opts.baseUrl.replace(/\/+$/, "");
  }

  private get headers(): Record<string, string> {
    return {
      "Content-Type": "application/json",
      ...(this.opts.apiKey ? { Authorization: `Bearer ${this.opts.apiKey}` } : {}),
    };
  }

  private rememberLimits(response: Response): void {
    const raw = response.headers.get("x-ratelimit-remaining-tokens");
    if (raw === null) return;
    const left = parseInt(raw, 10);
    if (Number.isNaN(left)) return;
    this.remaining = left;
    this.resetS = parseReset(response.headers.get("x-ratelimit-reset-tokens"));
    this.checkedAt = Date.now();
  }

  /**
   * Ждёт, пока провайдер восстановит лимит под запрос такого размера. Спрашиваем
   * у того, кто знает: сколько осталось и когда вернётся, сказано в заголовках
   * прошлого ответа.
   */
  async waitForBudget(need: number, sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))): Promise<void> {
    if (this.remaining === null || need <= this.remaining) return;
    const wait = this.resetS - (Date.now() - this.checkedAt) / 1000;
    if (wait <= 0) return;
    await sleep((wait + 1) * 1000); // секунда сверху: их часы и наши не совпадают
    this.remaining = null; // после ожидания оценка устарела
  }

  /**
   * Пригодные модели. Судим ТОЛЬКО по тому, что прислал провайдер: зашитые
   * списки имён у другого провайдера окажутся неверными. Отсеиваем не
   * текст→текст и слишком маленький контекст; если полей нет — не прячем,
   * судить не по чему.
   */
  async models(): Promise<ModelInfo[]> {
    const minimum = this.opts.minContextTokens ?? 16_000;
    const response = await fetch(`${this.base}/models`, {
      headers: this.headers,
      signal: AbortSignal.timeout(10_000),
    }).catch(() => null);
    if (!response?.ok) return [];
    const data = (await response.json()) as { data?: Record<string, unknown>[] };
    const suitable = (raw: Record<string, unknown>): boolean => {
      const input = raw.input_modalities as string[] | undefined;
      const output = raw.output_modalities as string[] | undefined;
      if (input && !input.includes("text")) return false;
      if (output && !output.includes("text")) return false;
      const context = (raw.context_window ?? raw.context_length) as number | undefined;
      return !(context !== undefined && context < minimum);
    };
    return (data.data ?? [])
      .filter(suitable)
      .map((raw) => ({
        id: String(raw.id),
        context: (raw.context_window ?? raw.context_length) as number | undefined,
      }));
  }

  /**
   * Сколько токенов в минуту разрешено модели, или null. В списке моделей
   * лимита нет — провайдер сообщает его только заголовком ответа, поэтому шлём
   * самый дешёвый запрос: одно слово и один токен в ответе. Ответ с ошибкой
   * тоже годится, заголовки приходят и с ним.
   */
  async tokenLimit(model: string): Promise<number | null> {
    if (!this.base || !model) return null;
    const response = await fetch(`${this.base}/chat/completions`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "1" }],
        max_tokens: 1,
        temperature: 0,
      }),
      signal: AbortSignal.timeout(8_000),
    }).catch(() => null);
    // Сеть у пользователя нестабильна — это не повод ломать сохранение настроек
    if (!response) return null;
    this.rememberLimits(response);
    const raw = response.headers.get("x-ratelimit-limit-tokens");
    const limit = raw ? parseInt(raw, 10) : NaN;
    return Number.isNaN(limit) ? null : limit;
  }

  async generate(
    model: string,
    prompt: string,
    { system, temperature = 0.4, onDelta }: {
      system?: string; temperature?: number; onDelta?: Delta;
    } = {},
  ): Promise<string> {
    const body = {
      model,
      messages: [
        ...(system ? [{ role: "system", content: system }] : []),
        { role: "user", content: prompt },
      ],
      temperature,
      stream: Boolean(onDelta),
    };
    // Сколько этот запрос будет стоить: вход плюс место под ответ, которое
    // провайдер списывает независимо от того, воспользуется им модель или нет.
    const cost =
      Math.trunc(((system?.length ?? 0) + prompt.length) / CHARS_PER_TOKEN) +
      Math.trunc((this.opts.tpmLimit ?? 0) * OUTPUT_SHARE);
    await this.waitForBudget(cost);

    const response = await fetch(`${this.base}/chat/completions`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify(body),
    }).catch(() => {
      throw new LlmError(`Модель недоступна по адресу ${this.base}. Проверьте адрес и ключ.`);
    });
    this.rememberLimits(response);
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      throw new LlmError(`Провайдер отказал (${response.status}): ${detail(body, response.statusText)}`);
    }
    return onDelta ? this.readStream(response, onDelta) : this.readWhole(response);
  }

  private async readWhole(response: Response): Promise<string> {
    const data = (await response.json()) as {
      choices?: { message?: { content?: string } }[];
    };
    return (data.choices?.[0]?.message?.content ?? "").replace(THINK, "").trim();
  }

  /** Разбирает поток событий: куски ответа и ход мыслей приходят раздельно. */
  private async readStream(response: Response, onDelta: Delta): Promise<string> {
    const reader = response.body?.getReader();
    if (!reader) throw new LlmError("Провайдер обещал поток, но не прислал тела ответа");
    const decoder = new TextDecoder();
    let tail = "";
    let text = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      tail += decoder.decode(value, { stream: true });
      const lines = tail.split("\n");
      tail = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") return text.replace(THINK, "").trim();
        let chunk: { choices?: { delta?: Record<string, string> }[] };
        try {
          chunk = JSON.parse(payload);
        } catch {
          continue; // провайдер иногда шлёт служебные lines — они не наши
        }
        const delta = chunk.choices?.[0]?.delta ?? {};
        if (delta.reasoning) onDelta("reasoning", delta.reasoning);
        if (delta.content) {
          text += delta.content;
          onDelta("content", delta.content);
        }
      }
    }
    return text.replace(THINK, "").trim();
  }
}
