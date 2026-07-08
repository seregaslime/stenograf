import { getServerUrl } from "../store";
import type { LiveEvent } from "../types";

export type Channel = 0 | 1; // 0 = микрофон, 1 = системный звук

/** WebSocket-клиент живой встречи: аудио-кадры туда, события транскрипта обратно. */
export class LiveClient {
  private ws: WebSocket | null = null;

  constructor(
    private onEvent: (event: LiveEvent) => void,
    private onClose: () => void,
  ) {}

  connect(): Promise<void> {
    const url = getServerUrl().replace(/^http/, "ws") + "/ws/live";
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";
      ws.onopen = () => {
        this.ws = ws;
        resolve();
      };
      ws.onerror = () => reject(new Error(`Не удалось подключиться к ${url}`));
      ws.onmessage = (message) => {
        try {
          this.onEvent(JSON.parse(message.data as string) as LiveEvent);
        } catch {
          /* бинарных сообщений от сервера нет */
        }
      };
      ws.onclose = () => {
        this.ws = null;
        this.onClose();
      };
    });
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  start(options: { title: string; record_audio: boolean; hints: boolean }): void {
    this.sendJson({ type: "start", ...options });
  }

  stop(): void {
    this.sendJson({ type: "stop" });
  }

  setHints(enabled: boolean): void {
    this.sendJson({ type: "hints", enabled });
  }

  sendAudio(channel: Channel, pcm: ArrayBuffer): void {
    if (!this.connected) return;
    const frame = new Uint8Array(1 + pcm.byteLength);
    frame[0] = channel;
    frame.set(new Uint8Array(pcm), 1);
    this.ws!.send(frame.buffer);
  }

  close(): void {
    this.ws?.close();
  }

  private sendJson(payload: unknown): void {
    if (this.connected) this.ws!.send(JSON.stringify(payload));
  }
}
