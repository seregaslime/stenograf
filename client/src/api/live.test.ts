import { describe, expect, it, vi } from "vitest";

import { LiveClient } from "./live";
import { setSetting } from "../store";

function withFakeSocket(readyState: number) {
  const client = new LiveClient(
    () => {},
    () => {},
  );
  const sent: ArrayBuffer[] = [];
  (client as unknown as { ws: unknown }).ws = {
    readyState,
    send: (b: ArrayBuffer) => sent.push(b),
  };
  return { client, sent };
}

describe("LiveClient.sendAudio", () => {
  it("ставит байт канала первым и копирует PCM следом", () => {
    const { client, sent } = withFakeSocket(WebSocket.OPEN);
    const pcm = new Uint8Array([10, 20, 30, 40]).buffer;
    client.sendAudio(1, pcm);
    expect(sent).toHaveLength(1);
    const frame = new Uint8Array(sent[0]);
    expect(frame[0]).toBe(1); // канал = system
    expect(Array.from(frame.slice(1))).toEqual([10, 20, 30, 40]);
  });

  it("молчит, если сокет не открыт", () => {
    const { client, sent } = withFakeSocket(WebSocket.CLOSED);
    client.sendAudio(0, new Uint8Array([1, 2]).buffer);
    expect(sent).toHaveLength(0);
  });
});

describe("LiveClient.start", () => {
  it("передаёт тип встречи серверу (от него зависят промпты)", () => {
    const client = new LiveClient(
      () => {},
      () => {},
    );
    const sent: string[] = [];
    (client as unknown as { ws: unknown }).ws = {
      readyState: WebSocket.OPEN,
      send: (text: string) => sent.push(text),
    };
    client.start({
      title: "Собес",
      record_audio: false,
      summarize: true,
      meeting_mode: "interview",
    });
    expect(JSON.parse(sent[0])).toMatchObject({
      type: "start",
      title: "Собес",
      meeting_mode: "interview",
    });
  });
});

describe("LiveClient: токен первым кадром", () => {
  /** Сокет-заглушка: запоминает отправленное и сам зовёт onopen. */
  function fakeSocketClass(sent: string[]) {
    return class {
      static OPEN = 1;
      readyState = 1;
      binaryType = "";
      onopen: (() => void) | null = null;
      onerror: (() => void) | null = null;
      onmessage: (() => void) | null = null;
      onclose: (() => void) | null = null;
      constructor() {
        queueMicrotask(() => this.onopen?.());
      }
      send(data: string) {
        sent.push(data);
      }
    };
  }

  it("шлёт auth сразу после открытия, до всего остального", async () => {
    setSetting("serverToken", "токен-встречи");
    const sent: string[] = [];
    vi.stubGlobal("WebSocket", fakeSocketClass(sent));
    const client = new LiveClient(
      () => {},
      () => {},
    );
    await client.connect();
    expect(JSON.parse(sent[0])).toEqual({ type: "auth", token: "токен-встречи" });
    setSetting("serverToken", "");
    vi.unstubAllGlobals();
  });

  it("шлёт кадр и с пустым токеном: отказ должен прийти сразу, а не по таймауту", async () => {
    setSetting("serverToken", "");
    const sent: string[] = [];
    vi.stubGlobal("WebSocket", fakeSocketClass(sent));
    const client = new LiveClient(
      () => {},
      () => {},
    );
    await client.connect();
    expect(JSON.parse(sent[0])).toEqual({ type: "auth", token: "" });
    vi.unstubAllGlobals();
  });
});
