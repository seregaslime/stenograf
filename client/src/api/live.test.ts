import { describe, expect, it } from "vitest";

import { LiveClient } from "./live";

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
