import { beforeEach, describe, expect, it } from "vitest";

import { DEFAULT_SERVER_URL, getServerUrl, getSetting, isDebugMode, setSetting } from "./store";

beforeEach(() => localStorage.clear());

describe("store", () => {
  it("getServerUrl падает на дефолт без сохранённого адреса", () => {
    expect(getServerUrl()).toBe(DEFAULT_SERVER_URL);
  });

  it("getServerUrl срезает завершающие слеши", () => {
    setSetting("serverUrl", "http://host:8765///");
    expect(getServerUrl()).toBe("http://host:8765");
  });

  it("getSetting/setSetting работают через префикс stenograf.", () => {
    setSetting("foo", "bar");
    expect(getSetting("foo")).toBe("bar");
    expect(localStorage.getItem("stenograf.foo")).toBe("bar");
    expect(getSetting("missing", "def")).toBe("def");
  });

  it("isDebugMode отражает флаг '1'", () => {
    expect(isDebugMode()).toBe(false);
    setSetting("debug", "1");
    expect(isDebugMode()).toBe(true);
    setSetting("debug", "0");
    expect(isDebugMode()).toBe(false);
  });
});
