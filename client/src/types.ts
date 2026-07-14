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

export interface MeetingDetail {
  id: number;
  title: string;
  status: "live" | "summarizing" | "done";
  started_at: string | null;
  ended_at: string | null;
  record_audio: boolean;
  summary: string | null;
  summary_model: string | null;
  summary_error: string | null;
  segments: SegmentDto[];
}

export interface SpeakerSampleDto {
  id: number;
  duration_s: number;
}

export interface SpeakerDto {
  id: number;
  name: string;
  is_self: boolean;
  meetings_count: number;
  segments_count: number;
  voiceprints_count: number;
  created_at: string | null;
  samples: SpeakerSampleDto[];
  voiceprints: { id: number; count: number }[];
}

export interface HealthDto {
  status: string;
  version: string;
  asr: { engine: string; model: string; loaded: boolean };
  diarization: { loaded: boolean };
  ollama: { reachable: boolean; models: string[] };
  summary_model: string;
  hints_model: string;
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
  | { type: "ready"; meeting_id: number; title: string }
  | { type: "segment"; segment: SegmentDto }
  | { type: "speaker_new"; speaker: { id: number; name: string } }
  | { type: "hint"; text: string }
  | { type: "hint_error"; message: string }
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
