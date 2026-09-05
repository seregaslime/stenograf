/**
 * Сквозная проверка обеих половин: сервер распознаёт речь, приложение
 * составляет протокол.
 *
 * Раньше такой прогон жил только на сервере — он же и считал протокол. После
 * переезда одной стороны мало: серверный e2e доходит до транскрипта и на этом
 * заканчивается, а всё, ради чего человек включает приложение, происходит
 * дальше. Поэтому сквозная проверка теперь здесь: настоящий uvicorn с
 * настоящими моделями распознавания, настоящий WebSocket, синтезированный
 * голос — и настоящий клиентский код, собирающий протокол.
 *
 * Модель языка подменена намеренно: проверяется конвейер, а не качество
 * формулировок. Живая модель — в ручном чек-листе TESTING.md.
 *
 * Запуск (медленный, минуты):  npm run test:e2e
 */
import { execFileSync, spawn, type ChildProcess } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

import { LlmRouter, type LlmSettings } from "../llm/router";
import { generateSummary } from "../llm/summary";
import type { SegmentDto } from "../types";

const PORT = 8767;
const BASE = `http://127.0.0.1:${PORT}`;
const SERVER_DIR = resolve(__dirname, "../../../server");
const PYTHON = join(SERVER_DIR, ".venv/bin/python");
const МОДЕЛИ = join(SERVER_DIR, "data/models");

const НАСТРОЙКИ: LlmSettings = {
  provider: "local",
  ollamaUrl: "http://127.0.0.1:1", // сюда никто не пойдёт: модель подменена
  localSummaryModel: "qwen3:4b",
  localHintsModel: "qwen3:1.7b",
  apiBaseUrl: "",
  apiKey: "",
  apiSummaryModel: "",
  apiHintsModel: "",
  embedModel: "bge-m3",
};

let сервер: ChildProcess | null = null;

/** Речь голосом macOS → PCM16 16 кГц, как его шлёт приложение с микрофона. */
function произнести(текст: string, голос = "Milena"): Buffer {
  const wav = join(mkdtempSync(join(tmpdir(), "e2e-")), "voice.wav");
  execFileSync("say", ["-v", голос, "-o", wav, "--data-format=LEI16@16000", текст]);
  const файл = readFileSync(wav);
  // Заголовок wav фиксированной длины: нам нужны только сэмплы
  return файл.subarray(44);
}

async function здоровье(): Promise<{ asr?: { loaded: boolean } } | null> {
  try {
    return await fetch(`${BASE}/api/health`).then((r) => r.json());
  } catch {
    return null;
  }
}

async function дождаться(проверка: () => Promise<boolean>, секунд: number, что: string) {
  const до = Date.now() + секунд * 1000;
  while (Date.now() < до) {
    if (await проверка()) return;
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error(`не дождались: ${что}`);
}

beforeAll(async () => {
  if (!existsSync(PYTHON) || !existsSync(МОДЕЛИ)) return; // тесты сами себя пропустят
  const данные = mkdtempSync(join(tmpdir(), "e2e-data-"));
  symlinkSync(МОДЕЛИ, join(данные, "models"));
  сервер = spawn(PYTHON, ["-m", "uvicorn", "app.main:app", "--port", String(PORT)], {
    cwd: SERVER_DIR,
    env: {
      ...process.env,
      STENOGRAF_DATA_DIR: данные,
      STENOGRAF_PRELOAD_ASR: "true",
      // Синтетические голоса macOS ближе живых — порог поднят, как и в
      // серверном e2e: проверяем правила, а не подбор числа.
      STENOGRAF_SPEAKER_MATCH_THRESHOLD: "0.45",
    },
    stdio: "ignore",
  });
  await дождаться(async () => Boolean((await здоровье())?.asr?.loaded), 180, "загрузку моделей");
}, 200_000);

afterAll(() => сервер?.kill());

const доступно = existsSync(PYTHON) && existsSync(МОДЕЛИ) && process.platform === "darwin";

describe.skipIf(!доступно)("встреча целиком: сервер распознаёт, приложение пишет протокол", () => {
  it("транскрипт приходит с сервера, протокол уходит на сервер", async () => {
    const segments: SegmentDto[] = [];
    const ws = new WebSocket(`ws://127.0.0.1:${PORT}/ws/live`);
    let meetingId = 0;

    await new Promise<void>((готово, ошибка) => {
      ws.onerror = () => ошибка(new Error("сокет не открылся"));
      ws.onopen = () => готово();
    });
    ws.onmessage = (кадр) => {
      const событие = JSON.parse(String(кадр.data));
      if (событие.type === "ready") meetingId = событие.meeting_id;
      if (событие.type === "segment") segments.push(событие.segment);
    };
    ws.send(JSON.stringify({
      type: "start", title: "e2e: протокол считает клиент",
      record_audio: false, summarize: false, meeting_mode: "work",
    }));
    await дождаться(async () => meetingId > 0, 20, "начало встречи");

    // Канал 0 — микрофон: [байт канала] + PCM, как шлёт приложение
    const pcm = произнести("Давайте перенесём демонстрацию на вторник. Разбор пришлю в пятницу.");
    for (let i = 0; i < pcm.length; i += 3200) {
      const кусок = pcm.subarray(i, i + 3200);
      const кадр = new Uint8Array(1 + кусок.length);
      кадр[0] = 0;
      кадр.set(кусок, 1);
      ws.send(кадр);
      await new Promise((r) => setTimeout(r, 20)); // темп, близкий к реальному
    }
    ws.send(JSON.stringify({ type: "stop" }));
    await дождаться(async () => segments.length > 0, 90, "распознанные реплики");
    ws.close();

    // Дальше — работа приложения: транскрипт, промпт, модель, отправка протокола
    const llm = new LlmRouter(НАСТРОЙКИ);
    let увиденное = "";
    vi.spyOn(llm, "generate").mockImplementation(async (_role, prompt) => {
      увиденное = prompt;
      return "## Краткий итог\nПеренесли демонстрацию на вторник.";
    });

    const протокол = await generateSummary(llm, {
      segments,
      title: "e2e: протокол считает клиент",
      date: "05.09.2026 12:00",
      mode: "work",
    });
    // Модель увидела настоящий распознанный текст, а не заготовку
    expect(увиденное).toContain("Расшифровка:");
    expect(увиденное.toLowerCase()).toContain("вторник");

    const ответ = await fetch(`${BASE}/api/meetings/${meetingId}/summary`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: протокол, model: "e2e-подстановка" }),
    });
    expect(ответ.status).toBe(200);

    // И то же самое глазами сервера: протокол сохранён и попадает в выгрузку
    const встреча = await fetch(`${BASE}/api/meetings/${meetingId}`).then((r) => r.json());
    expect(встреча.summary).toContain("Перенесли демонстрацию на вторник");
    expect(встреча.summary_model).toBe("e2e-подстановка");
    expect(встреча.status).toBe("done");

    const выгрузка = await fetch(`${BASE}/api/meetings/${meetingId}/export?fmt=md`).then((r) =>
      r.text(),
    );
    expect(выгрузка).toContain("Перенесли демонстрацию на вторник");
  }, 240_000);
});
