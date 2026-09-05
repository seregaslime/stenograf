/**
 * Подсказки во время встречи: когда спрашивать модель, что ей показывать и что
 * делать с ответом. Перенесено с сервера (ws.py) — там это жило в живой сессии,
 * потому что модель звал сервер.
 *
 * Ни DOM, ни React: движок должен запускаться в Node, иначе сквозные тесты не
 * смогут проверить подсказки.
 *
 * Числа не тронуты, все подбирались на живых встречах.
 */
import { buildAnswerPrompt } from "./prompts/answer";
import { buildHintPrompt, parseHint } from "./prompts/hint";
import { LlmError } from "./ollama";
import type { LlmRouter } from "./router";
import { CHARS_PER_TOKEN } from "./router";
import { similarityRatio } from "./similarity";

/** Минимум между подсказками, чтобы не частить. */
const MIN_GAP_S = 15;
/** Сколько нового текста накопить, прежде чем дёргать модель. */
const MIN_NEW_CHARS = 15;
/** Меньше разговора — рано подсказывать. */
const MIN_CONTEXT_CHARS = 80;
/** Короче — считаем, что модель промолчала. */
const MIN_LEN_CHARS = 12;
/** Сколько последних подсказок помнить против повторов и насколько похоже — дубль. */
const MEMORY = 3;
const DUP_RATIO = 0.85;
/** После SKIP пауза короче обычной, серия подряд растит её линейно. */
const SKIP_GAP_S = 8;
const SKIP_MAX_GAP_S = 45;
/** Столько ошибок подряд — подсказки выключаются; потолок паузы между повторами. */
const MAX_FAILS = 5;
const MAX_BACKOFF_S = 120;
const TEMPERATURE = 0.4;
/** Реплик в памяти окна и подсказок в журнале контекста. */
const RECENT_MAXLEN = 400;
const HINT_LOG_MAXLEN = 20;

export interface HintEngineOptions {
  llm: LlmRouter;
  mode: string | null;
  title: string;
  /** Участники строкой — их считает вызывающий, он знает имена. */
  participants: () => string;
  onHint: (text: string) => void;
  onError: (message: string) => void;
  /** Печать ответа по мере генерации — только для кнопки «Подсказать сейчас». */
  onDelta?: (text: string) => void;
  /** Часы: в тестах подменяются, чтобы не ждать по-настоящему. */
  now?: () => number;
}

export class HintEngine {
  private lines: string[] = [];
  private linesTotal = 0;
  private hintedAtLine = 0;
  /** Где именно ассистент подсказывал: [номер реплики, текст]. */
  private hintLog: [number, string][] = [];
  private recentHints: string[] = [];
  private charsSinceHint = 0;
  private lastHintAt = 0;
  private hintGapS = MIN_GAP_S;
  private skipStreak = 0;
  private failStreak = 0;
  private backoffUntil = 0;
  private inFlight = false;
  private explicitInFlight = false;
  /** Выключаются сами после серии ошибок; человек может включить заново. */
  enabled = false;

  constructor(private opts: HintEngineOptions) {}

  private get now(): number {
    return (this.opts.now ?? (() => Date.now() / 1000))();
  }

  /** Новая реплика разговора. */
  push(name: string, text: string): void {
    this.lines.push(`${name}: ${text}`);
    if (this.lines.length > RECENT_MAXLEN) this.lines.shift();
    this.linesTotal += 1;
    this.charsSinceHint += text.length;
  }

  /**
   * Пора ли подсказывать: не в периоде бэкоффа, накопилось нового текста и
   * прошёл минимальный интервал (он растёт при серии SKIP).
   */
  private shouldHint(now: number): boolean {
    return (
      now >= this.backoffUntil &&
      this.charsSinceHint >= MIN_NEW_CHARS &&
      now - this.lastHintAt >= this.hintGapS
    );
  }

  /** Вызывается по таймеру: сама решает, дёргать модель или нет. */
  async tick(): Promise<void> {
    if (!this.enabled || this.inFlight || !this.shouldHint(this.now)) return;
    await this.emit(false);
  }

  /**
   * Модель промолчала — это не ошибка, а норма. Счётчик текста уже сброшен
   * (материал модель посмотрела и признала непригодным), но окно не трогаем: на
   * следующей попытке она увидит и старое, и новое.
   */
  private onSkip(): void {
    this.skipStreak += 1;
    this.hintGapS = Math.min(SKIP_GAP_S * this.skipStreak, SKIP_MAX_GAP_S);
  }

  /** Почти-дубль недавней подсказки (сравнение без учёта регистра). */
  private isDuplicate(hint: string): boolean {
    const candidate = hint.toLowerCase();
    return this.recentHints.some(
      (prev) => similarityRatio(candidate, prev.toLowerCase()) >= DUP_RATIO,
    );
  }

  /**
   * Делит разговор на (контекст, новое) по границе прошлой подсказки.
   *
   * Свои прошлые подсказки вплетаются туда, где они прозвучали, ВСЕГДА — в том
   * числе по кнопке: «смотреть на весь разговор» не значит «забыть свои
   * ответы», иначе модель снова видит закрытые вопросы открытыми.
   */
  private splitWindow(budgetChars: number, force: boolean): [string, string] {
    const firstNo = this.linesTotal - this.lines.length;
    const pending = this.hintLog.filter(([no]) => no > firstNo);
    const rendered: string[] = [];
    let newFrom: number | null = null;

    for (const [offset, line] of this.lines.entries()) {
      const no = firstNo + offset;
      while (pending.length > 0 && pending[0][0] <= no) {
        rendered.push(`  [ты подсказал: ${pending.shift()![1]}]`);
      }
      if (newFrom === null && no >= this.hintedAtLine) newFrom = rendered.length;
      rendered.push(line);
    }
    for (const [, text] of pending) rendered.push(`  [ты подсказал: ${text}]`);

    // Делить не на что: первая подсказка, нового не появилось, или нажата
    // кнопка — там человек просит посмотреть на разговор целиком.
    if (force || !newFrom) return ["", rendered.join("\n").slice(-budgetChars)];

    const newText = rendered.slice(newFrom).join("\n").slice(-budgetChars);
    const left = Math.max(0, budgetChars - newText.length);
    return [left ? rendered.slice(0, newFrom).join("\n").slice(-left) : "", newText];
  }

  /**
   * Сколько символов разговора влезает в один запрос.
   *
   * Размер промпта не забиваем числом, а меряем сборкой с пустым транскриптом:
   * поменяются промпты — пересчитается само. Меряем вариантом с разрешённым
   * молчанием, он длиннее, то есть оценка с запасом в безопасную сторону.
   */
  private windowChars(): number {
    const budget = this.opts.llm.budget;
    const hintsTokens = this.opts.llm.tpmLimit(this.opts.llm.modelFor("hints"));
    if (this.opts.llm.provider !== "api" || !hintsTokens) return budget.hintsChars;

    const { system, prompt } = buildHintPrompt({
      mode: this.opts.mode,
      transcript: "",
      earlier: "",
      previous: this.recentHints.join("\n") || "—",
      title: this.opts.title,
      participants: this.opts.participants(),
      detailed: budget.detailed,
      allowSkip: true,
    });
    const overhead = (system.length + prompt.length) / CHARS_PER_TOKEN;
    const free = Math.trunc((hintsTokens - overhead) * CHARS_PER_TOKEN);
    // Если бюджета не хватает даже на инструкции, подсказки на этом тарифе
    // невозможны: отправляем минимум и даём провайдеру объяснить, что не так.
    return Math.max(MIN_CONTEXT_CHARS, Math.min(budget.hintsChars, free));
  }

  /**
   * Подсказка. force — кнопка «Подсказать сейчас»: молчать нельзя, дедуп
   * отключён, а на каждый отказ человек получает внятный ответ вместо тишины.
   */
  async emit(force: boolean): Promise<void> {
    if (force) {
      if (this.explicitInFlight) {
        this.opts.onError("Подсказка уже готовится…");
        return;
      }
    } else if (this.inFlight) {
      return;
    }

    const [earlier, window] = this.splitWindow(this.windowChars(), force);
    if (window.length + earlier.length < MIN_CONTEXT_CHARS) {
      if (force) this.opts.onError("Пока слишком мало разговора для подсказки.");
      return; // счётчики не трогаем — контекст копится дальше
    }

    this.inFlight = true;
    this.explicitInFlight = force;
    this.charsSinceHint = 0;
    this.lastHintAt = this.now;
    try {
      const { system, prompt } = buildHintPrompt({
        mode: this.opts.mode,
        transcript: window,
        earlier,
        previous: this.recentHints.join("\n") || "—",
        title: this.opts.title,
        participants: this.opts.participants(),
        detailed: this.opts.llm.budget.detailed,
        allowSkip: !force,
      });
      // Печатаем только по кнопке. Автоподсказку стримить нечем: модель вправе
      // промолчать, и молчание приходит словом SKIP в тексте ответа — человек
      // увидел бы, как в панели появляется «SKIP» и исчезает.
      const raw = await this.opts.llm.generate("hints", prompt, {
        system,
        temperature: TEMPERATURE,
        onDelta:
          force && this.opts.onDelta
            ? (kind, chunk) => kind === "content" && this.opts.onDelta!(chunk)
            : undefined,
      });
      this.failStreak = 0;
      const hint = parseHint(raw, MIN_LEN_CHARS);
      // Модель этот текст посмотрела — двигаем границу, что бы она ни ответила.
      // Иначе отвергнутый фрагмент вернётся на следующей попытке, и так по
      // кругу, пока модель не надумает подсказку на пустом месте.
      this.hintedAtLine = this.linesTotal;
      if (hint === null) {
        if (force) this.opts.onError("Модель не нашла, что подсказать. Попробуйте позже.");
        else this.onSkip();
        return;
      }
      this.skipStreak = 0;
      this.hintGapS = MIN_GAP_S;
      if (force || !this.isDuplicate(hint)) {
        this.recentHints.push(hint);
        if (this.recentHints.length > MEMORY) this.recentHints.shift();
        this.hintLog.push([this.linesTotal, hint]);
        if (this.hintLog.length > HINT_LOG_MAXLEN) this.hintLog.shift();
        this.opts.onHint(hint);
      }
    } catch (exc) {
      // Один сбой не выключает подсказки: наращиваем бэкофф и гасим только
      // после нескольких ошибок подряд (важно при нестабильной сети).
      this.failStreak += 1;
      this.backoffUntil =
        this.now + Math.min(MIN_GAP_S * 2 ** this.failStreak, MAX_BACKOFF_S);
      if (this.failStreak >= MAX_FAILS) {
        this.enabled = false;
        this.opts.onError("Подсказки приостановлены после нескольких ошибок связи с моделью.");
      } else if (this.failStreak === 1) {
        this.opts.onError((exc as Error).message);
      }
    } finally {
      this.inFlight = false;
      this.explicitInFlight = false;
    }
  }

  /**
   * Ответ на вопрос участника. Отличие от подсказки принципиальное: там модель
   * сама решает, о чём говорить, и вправе промолчать. Здесь спросил человек —
   * ответ обязателен, а тему задавать модели не надо.
   */
  async answer(
    question: string,
    quoted: string,
    onDelta?: (text: string) => void,
  ): Promise<string> {
    const [earlier] = [this.lines.join("\n").slice(-this.windowChars())];
    const { system, prompt } = buildAnswerPrompt({
      mode: this.opts.mode,
      question,
      quoted,
      earlier,
      title: this.opts.title,
      participants: this.opts.participants(),
    });
    const raw = await this.opts.llm.generate("hints", prompt, {
      system,
      temperature: TEMPERATURE,
      onDelta: onDelta ? (kind, chunk) => kind === "content" && onDelta(chunk) : undefined,
    });
    const text = raw.trim();
    if (!text) throw new LlmError("Модель вернула пустой ответ.");
    return text;
  }
}
