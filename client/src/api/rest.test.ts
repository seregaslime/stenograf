import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "./rest";
import { setSetting } from "../store";

afterEach(() => {
  vi.unstubAllGlobals();
  setSetting("serverToken", "");
});

function stubFetch(response: Partial<Response> & { json: () => Promise<unknown> }) {
  vi.stubGlobal("fetch", vi.fn(async () => response));
}

describe("rest.request", () => {
  it("возвращает разобранный JSON при успехе", async () => {
    stubFetch({ ok: true, json: async () => ({ status: "ok" }) });
    await expect(api.health()).resolves.toEqual({ status: "ok" });
  });

  it("бросает body.detail при ошибке", async () => {
    stubFetch({ ok: false, status: 400, json: async () => ({ detail: "плохо" }) });
    await expect(api.health()).rejects.toThrow("плохо");
  });

  it("падает на 'HTTP <код>', если тело не JSON", async () => {
    stubFetch({
      ok: false,
      status: 503,
      json: async () => {
        throw new Error("не JSON");
      },
    });
    await expect(api.health()).rejects.toThrow("HTTP 503");
  });
});

describe("rest.request: токен доступа", () => {
  /** Параметры объявлены явно: без них у мока пустой тип аргументов, и
   *  обращение к calls[0][1] не проходит проверку типов. */
  function stubAndCapture() {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      void init;
      return { ok: true, json: async () => ({}) };
    });
    vi.stubGlobal("fetch", fetchMock);
    return fetchMock;
  }

  function headersOf(fetchMock: ReturnType<typeof stubAndCapture>) {
    return (fetchMock.mock.calls[0]![1].headers ?? {}) as Record<string, string>;
  }

  it("не шлёт заголовок, пока токен не задан", async () => {
    const fetchMock = stubAndCapture();
    await api.health();
    expect(headersOf(fetchMock).Authorization).toBeUndefined();
  });

  it("шлёт токен схемой Bearer", async () => {
    setSetting("serverToken", "секретный-токен");
    const fetchMock = stubAndCapture();
    await api.health();
    expect(headersOf(fetchMock).Authorization).toBe("Bearer секретный-токен");
  });

  it("не теряет тип содержимого у запросов со своим телом", async () => {
    // Раньше «...init» затирал заголовки целиком — POST уходил без Content-Type
    setSetting("serverToken", "т");
    const fetchMock = stubAndCapture();
    await api.setAsr("gigaam", "v3_e2e_rnnt");
    const headers = headersOf(fetchMock);
    expect(headers["Content-Type"]).toBe("application/json");
    expect(headers.Authorization).toBe("Bearer т");
  });
});

describe("rest: файлы забираются с токеном, а не адресом", () => {
  /**
   * Аудио «звучания» и экспорт протокола раньше отдавались адресом —
   * `new Audio(url)` и `<a href>`. Заголовок Authorization браузерное API туда
   * подставить не даёт, и на сервере с заведёнными людьми оба пути отвечали
   * 401: «не удалось воспроизвести звучание» и молчаливо пустой экспорт.
   * Проверять это на личном сервере было нельзя — там токен не требуется.
   */
  function stubFile(headers: Record<string, string> = {}) {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      void init;
      return {
        ok: true,
        headers: { get: (имя: string) => headers[имя.toLowerCase()] ?? null },
        blob: async () => new Blob(["данные"]),
      };
    });
    vi.stubGlobal("fetch", fetchMock);
    return fetchMock;
  }

  it("звучание голоса запрашивается с заголовком", async () => {
    setSetting("serverToken", "секретный-токен");
    const fetchMock = stubFile();
    await api.voiceprintAudio(3, 7);
    const [адрес, init] = fetchMock.mock.calls[0]!;
    expect(адрес).toContain("/api/speakers/3/voiceprints/7/audio");
    expect((init.headers as Record<string, string>).Authorization)
      .toBe("Bearer секретный-токен");
  });

  it("имя файла берётся из ответа сервера, а не собирается заново", async () => {
    const fetchMock = stubFile({
      "content-disposition": 'attachment; filename="meeting_12.md"',
    });
    const { имя } = await api.exportFile(12, "md");
    expect(имя).toBe("meeting_12.md");
    expect(fetchMock.mock.calls[0]![0]).toContain("fmt=md");
  });

  it("отказ сервера объясняется словами, а не пустым файлом", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: false,
      status: 401,
      json: async () => ({ detail: "Нужен токен доступа" }),
    })));
    await expect(api.exportFile(1, "txt")).rejects.toThrow("Нужен токен доступа");
  });
});
