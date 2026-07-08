// Захват звука. Весь пайплайн работает на 16 кГц mono PCM16:
// AudioContext создаётся сразу с sampleRate 16000 (Chromium ресемплирует сам),
// AudioWorklet (public/pcm-worklet.js) нарезает поток на кадры по 100 мс.
import { platform } from "../store";

export interface CaptureCallbacks {
  onChunk: (pcm: ArrayBuffer) => void;
  onLevel: (level: number) => void;
}

export interface CaptureHandle {
  stop: () => void;
}

export class AudioEngine {
  private context: AudioContext | null = null;

  private async ensureContext(): Promise<AudioContext> {
    if (!this.context) {
      this.context = new AudioContext({ sampleRate: 16000 });
      await this.context.audioWorklet.addModule("pcm-worklet.js");
    }
    if (this.context.state === "suspended") await this.context.resume();
    return this.context;
  }

  /** Микрофон — голос владельца. Эхоподавление включено, чтобы звук из колонок
   *  (голоса собеседников) не дублировался в канал микрофона. */
  async startMic(deviceId: string | undefined, callbacks: CaptureCallbacks): Promise<CaptureHandle> {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    return this.attach(stream, callbacks);
  }

  /** Системный звук.
   *  macOS: виртуальное устройство BlackHole выбирается как обычный аудио-вход.
   *  Windows: getDisplayMedia + WASAPI loopback (см. electron/main.cjs). */
  async startSystem(deviceId: string | undefined, callbacks: CaptureCallbacks): Promise<CaptureHandle> {
    if (platform() === "win32") {
      const stream = await navigator.mediaDevices.getDisplayMedia({ audio: true, video: true });
      stream.getVideoTracks().forEach((track) => track.stop());
      if (stream.getAudioTracks().length === 0) {
        throw new Error("Система не отдала звук — разрешите захват аудио");
      }
      return this.attach(stream, callbacks);
    }
    if (!deviceId) {
      throw new Error("Выберите устройство системного звука (BlackHole) в настройках встречи");
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: { exact: deviceId },
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    return this.attach(stream, callbacks);
  }

  private async attach(stream: MediaStream, callbacks: CaptureCallbacks): Promise<CaptureHandle> {
    const context = await this.ensureContext();
    const source = context.createMediaStreamSource(stream);
    const worklet = new AudioWorkletNode(context, "pcm16");
    worklet.port.onmessage = (event: MessageEvent<{ pcm: ArrayBuffer; level: number }>) => {
      callbacks.onChunk(event.data.pcm);
      callbacks.onLevel(event.data.level);
    };
    source.connect(worklet);
    // worklet никуда не подключаем — звук не должен идти в колонки
    return {
      stop: () => {
        stream.getTracks().forEach((track) => track.stop());
        source.disconnect();
        worklet.disconnect();
        worklet.port.onmessage = null;
      },
    };
  }

  async close(): Promise<void> {
    await this.context?.close();
    this.context = null;
  }
}

/** Список аудио-входов. Сначала просим доступ, иначе labels будут пустыми. */
export async function listAudioInputs(): Promise<MediaDeviceInfo[]> {
  try {
    const probe = await navigator.mediaDevices.getUserMedia({ audio: true });
    probe.getTracks().forEach((track) => track.stop());
  } catch {
    /* нет разрешения — вернём то, что есть */
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((device) => device.kind === "audioinput");
}

/** Похоже ли устройство на виртуальный loopback-драйвер (BlackHole и аналоги). */
export function looksLikeLoopback(device: MediaDeviceInfo): boolean {
  return /blackhole|loopback|soundflower|vb-?cable|virtual/i.test(device.label);
}
