/**
 * Клиент локальной модели (Ollama). Раньше жил на сервере — теперь приложение
 * ходит к модели само: у каждого свой адрес, и сервер про модели не знает.
 *
 * Ни DOM, ни Electron здесь не используются намеренно: этот модуль должен
 * запускаться и в Node, иначе сквозные тесты не смогут проверить генерацию.
 */

/** Ответ Ollama на /api/generate — берём только то, чем пользуемся. */
interface GenerateResponse {
  response?: string;
  /** Размышления qwen3 приходят отдельным полем — в текст ответа они не лезут. */
  thinking?: string;
  eval_count?: number;
  eval_duration?: number;
}

export class LlmError extends Error {}

/** Вырезаем <think>…</think> на случай, если модель всё же вывалила мысли в текст. */
const THINK = /<think>[\s\S]*?<\/think>/g;

export interface OllamaOptions {
  url: string;
  /**
   * Сколько Ollama держит модель загруженной после запроса. По умолчанию
   * полчаса: холодная загрузка стоит около 22 секунд (замер 04.09.2026 на
   * qwen3:1.7b), и платить их за каждый вопрос после паузы незачем. Машине с
   * теснотой по памяти значение задают явно — там выгрузка важнее скорости.
   */
  keepAlive?: string;
}

export class OllamaClient {
  constructor(private opts: OllamaOptions) {}

  private get base(): string {
    return this.opts.url.replace(/\/+$/, "");
  }

  private async post<T>(
    path: string, body: unknown, model: string, timeoutMs = 600_000,
  ): Promise<T> {
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), timeoutMs);
    let response: Response;
    try {
      response = await fetch(this.base + path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: abort.signal,
      });
    } catch {
      // Две разные беды с разными советами. «Не ответила за столько-то» значит,
      // что адрес верный и модель считает — совет «проверьте адрес» тут только
      // уводит в сторону.
      throw new LlmError(
        abort.signal.aborted
          ? `Модель не ответила за ${Math.round(timeoutMs / 60_000)} мин. Возможно, она слишком велика для этой машины.`
          : `Модель недоступна по адресу ${this.base}. Проверьте адрес в настройках и что Ollama запущена.`,
      );
    } finally {
      clearTimeout(timer);
    }
    if (response.status === 404) {
      // Имя подставляем: «скачайте <модель>» человек выполнит не думая,
      // а «скачайте нужную модель» заставит идти искать, какую именно.
      throw new LlmError(`Модель «${model}» не найдена. Скачайте её: ollama pull ${model}`);
    }
    if (!response.ok) {
      const текст = await response.text().catch(() => "");
      throw new LlmError(`Модель вернула ошибку ${response.status}: ${текст.slice(0, 200)}`);
    }
    return (await response.json()) as T;
  }

  /** Какие модели скачаны. Пустой список — адрес отвечает, но моделей нет. */
  async models(): Promise<string[]> {
    try {
      const response = await fetch(`${this.base}/api/tags`, {
        signal: AbortSignal.timeout(5_000),
      });
      if (!response.ok) return [];
      const data = (await response.json()) as { models?: { name: string }[] };
      return (data.models ?? []).map((m) => m.name);
    } catch {
      return [];
    }
  }

  /**
   * Модели, которыми можно считать эмбеддинги.
   *
   * Ollama сама говорит, что умеет модель (`/api/show` → capabilities), и без
   * этого в списке рядом с bge-m3 стояли бы qwen3:4b и прочие разговорные:
   * эмбеддинги они выдадут, но поиск по ним будет хуже, а человек об этом не
   * узнает. Суффикс «:latest» убираем — Ollama отдаёт его в списке, но в
   * настройках модель записана как «bge-m3», и без этого она выглядела бы
   * ненайденной.
   *
   * Старая Ollama про capabilities не знает и отдаёт пустой список у всех:
   * тогда показываем все модели, а не пустоту, из которой нечего выбрать.
   */
  async embeddingModels(): Promise<string[]> {
    const все = await this.models();
    const умеют = await Promise.all(все.map((имя) => this.capabilities(имя)));
    const эмбеддинги = все.filter((_, i) => умеют[i].includes("embedding"));
    const список = эмбеддинги.length || умеют.some((c) => c.length) ? эмбеддинги : все;
    return [...new Set(список.map((имя) => имя.replace(/:latest$/, "")))];
  }

  private async capabilities(model: string): Promise<string[]> {
    try {
      const response = await fetch(`${this.base}/api/show`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model }),
        signal: AbortSignal.timeout(5_000),
      });
      if (!response.ok) return [];
      return ((await response.json()) as { capabilities?: string[] }).capabilities ?? [];
    } catch {
      return [];
    }
  }

  /** Отвечает ли адрес вообще. Пустой список моделей — не признак: у живой
   *  Ollama их может быть ноль, если ничего не скачано. */
  async reachable(): Promise<boolean> {
    try {
      const response = await fetch(`${this.base}/api/tags`, {
        signal: AbortSignal.timeout(5_000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Генерация. У qwen3 НЕ отключаем размышления и НЕ задаём num_predict:
   * мысли считаются в тот же лимит, и с потолком ответ приходит пустым —
   * грабли, на которые в проекте уже наступали.
   */
  async generate(
    model: string,
    prompt: string,
    { system, temperature = 0.4 }: { system?: string; temperature?: number } = {},
  ): Promise<string> {
    const data = await this.post<GenerateResponse>("/api/generate", {
      model,
      prompt,
      ...(system ? { system } : {}),
      stream: false,
      keep_alive: this.opts.keepAlive ?? "30m",
      options: { temperature },
    }, model);
    return (data.response ?? "").replace(THINK, "").trim();
  }

  /**
   * Векторы для списка текстов. Одним запросом на всю пачку: на каждый запрос
   * приходится загрузка модели в память, и по одному тексту за раз индексация
   * встречи занимала бы минуты вместо секунд.
   */
  async embed(model: string, texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    const data = await this.post<{ embeddings?: number[][] }>("/api/embed", {
      model,
      input: texts,
      keep_alive: this.opts.keepAlive ?? "30m",
    }, model);
    const vectors = data.embeddings ?? [];
    if (vectors.length !== texts.length) {
      throw new LlmError(
        `Модель вернула ${vectors.length} векторов на ${texts.length} текстов — она не для эмбеддингов?`,
      );
    }
    return vectors;
  }
}
