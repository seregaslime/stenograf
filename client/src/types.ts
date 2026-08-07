export interface SpeakerRef {
  id: number;
  name: string;
  is_self: boolean;
}

export interface SegmentDto {
  id: number;
  meeting_id: number;
  channel: "mic" | "system";
  start_s: number;
  end_s: number;
  text: string;
  similarity: number | null;
  speaker: SpeakerRef | null;
}

export interface MeetingListItem {
  id: number;
  title: string;
  status: "live" | "summarizing" | "done";
  started_at: string | null;
  ended_at: string | null;
  segments_count: number;
  has_summary: boolean;
}

/** Тип встречи: под каждый свой фокус подсказок и свои секции протокола. */
export type MeetingMode = "work" | "interview" | "negotiation";

export const MEETING_MODE_LABELS: Record<MeetingMode, string> = {
  work: "Рабочая встреча / планёрка",
  interview: "Собеседование (подсказки вам как соискателю)",
  negotiation: "Переговоры",
};

export interface MeetingDetail {
  id: number;
  title: string;
  status: "live" | "summarizing" | "done";
  started_at: string | null;
  ended_at: string | null;
  record_audio: boolean;
  meeting_mode: MeetingMode;
  summary: string | null;
  summary_model: string | null;
  summary_error: string | null;
  segments: SegmentDto[];
}

export interface VoiceprintDto {
  id: number;
  count: number; // из скольких реплик усреднён
  audio_duration_s: number | null; // null — отпечаток без аудио (создан до v0.4)
}

export interface SpeakerDto {
  id: number;
  name: string;
  is_self: boolean;
  meetings_count: number;
  segments_count: number;
  voiceprints_count: number;
  created_at: string | null;
  voiceprints: VoiceprintDto[];
}

export interface HealthDto {
  status: string;
  version: string;
  asr: { engine: string; model: string; loaded: boolean };
  diarization: { loaded: boolean };
  ollama: { reachable: boolean; models: string[] };
  llm: { provider: "local" | "api"; api_configured: boolean };
  summary_model: string;
  hints_model: string;
}

export interface LlmStateDto {
  provider: "local" | "api";
  api_configured: boolean;
  api_base_url: string;
  /** Чем заполнить поле адреса, если ничего не сохранено (задаётся сервером). */
  api_base_url_default: string;
  /** Токенов в минуту по моделям — измеряется при сохранении настроек.
   *  Именно этот лимит ограничивает подсказки, а не размер контекста. */
  api_tpm_limits: Record<string, number>;
  reachable: boolean;
  models: string[];
  models_info: LlmModelInfo[];
  models_rejected: number;
  summary_model: string; // модели активного провайдера (строка статуса)
  hints_model: string;
  api_summary_model: string; // модели API — отдельно, для формы настроек
  api_hints_model: string;
}

/** Модель API с размером контекстного окна (его сообщает сам провайдер). */
export interface LlmModelInfo {
  id: string;
  context_window: number | null;
}

export interface LlmModelsDto {
  reachable: boolean;
  models: string[];
  models_info: LlmModelInfo[];
  /** Сколько моделей провайдера не подошло — их в списке нет. */
  models_rejected: number;
}

export interface LlmSettings {
  provider: string;
  api_base_url?: string;
  api_key?: string;
  summary_model?: string;
  hints_model?: string;
}

export interface AsrStateDto {
  engine: string;
  model: string;
  loaded: boolean;
  loading: boolean;
  error: string | null;
  engines: { faster_whisper: boolean; mlx: boolean; gigaam: boolean };
  models_by_engine: Record<string, string[]>;
}

export type LiveEvent =
  | { type: "ready"; meeting_id: number; title: string; meeting_mode?: MeetingMode }
  | { type: "segment"; segment: SegmentDto }
  | { type: "speaker_new"; speaker: { id: number; name: string } }
  | { type: "hint"; text: string }
  | { type: "hint_error"; message: string }
  /** Ответ на вопрос участника из окна чата (команда ask). */
  | { type: "answer"; text: string }
  | { type: "answer_error"; message: string }
  | { type: "stopped"; meeting_id: number }
  | { type: "error"; message: string };

declare global {
  interface Window {
    stenograf?: {
      platform: string;
      enableLoopbackAudio?: () => Promise<void>;
      disableLoopbackAudio?: () => Promise<void>;
      getScreenPermission?: () => Promise<string>; // granted | denied | not-determined | restricted | unknown
      openScreenSettings?: () => Promise<void>;
    };
  }
}
