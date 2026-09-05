import { getServerUrl, getToken } from "../store";
import type { LiveEvent, MeetingMode } from "../types";

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
        // Токен уходит первым кадром, а не заголовком: заголовки браузерному
        // WebSocket задать нельзя, а в адресе токен попал бы в журналы сервера.
        // Шлём кадр всегда, даже с пустым токеном: закрытый сервер ответит
        // отказом сразу, а не через десять секунд ожидания — иначе интерфейс
        // показывает начавшуюся встречу, и человек всё это время говорит впустую.
        // Личный сервер этот кадр просто пропустит.
        this.sendJson({ type: "auth", token: getToken() });
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

  start(options: {
    title: string;
    record_audio: boolean;
    summarize: boolean;
    meeting_mode: MeetingMode;
  }): void {
    this.sendJson({ type: "start", ...options });
  }

  stop(): void {
    this.sendJson({ type: "stop" });
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
