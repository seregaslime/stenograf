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
