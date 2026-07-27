import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "./rest";

afterEach(() => vi.unstubAllGlobals());

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
