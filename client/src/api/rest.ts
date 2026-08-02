import { getServerUrl } from "../store";
import type {
  AsrStateDto,
  HealthDto,
  LlmModelsDto,
  LlmSettings,
  LlmStateDto,
  MeetingDetail,
  MeetingListItem,
  SpeakerDto,
} from "../types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(getServerUrl() + path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body.detail) detail = body.detail;
    } catch {
      /* тело не JSON — оставляем код */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<HealthDto>("/api/health"),

  asr: () => request<AsrStateDto>("/api/asr"),
  setAsr: (engine: string, model: string) =>
    request<AsrStateDto>("/api/asr", {
      method: "POST",
      body: JSON.stringify({ engine, model }),
    }),

  llm: () => request<LlmStateDto>("/api/llm"),
  setLlm: (settings: LlmSettings) =>
    request<LlmStateDto>("/api/llm", {
      method: "POST",
      body: JSON.stringify(settings),
    }),
  // Список моделей у API по введённым (ещё не сохранённым) кредам
  probeModels: (api_base_url: string, api_key?: string) =>
    request<LlmModelsDto>("/api/llm/models", {
      method: "POST",
      body: JSON.stringify({ api_base_url, api_key }),
    }),

  meetings: () => request<MeetingListItem[]>("/api/meetings"),
  meeting: (id: number) => request<MeetingDetail>(`/api/meetings/${id}`),
  deleteMeeting: (id: number) =>
    request<{ deleted: number }>(`/api/meetings/${id}`, { method: "DELETE" }),
  summarize: (id: number) =>
    request<{ status: string }>(`/api/meetings/${id}/summarize`, { method: "POST" }),
  exportUrl: (id: number, fmt: "md" | "txt") =>
    `${getServerUrl()}/api/meetings/${id}/export?fmt=${fmt}`,

  speakers: () => request<SpeakerDto[]>("/api/speakers"),
  renameSpeaker: (id: number, name: string) =>
    request<{ id: number; name: string }>(`/api/speakers/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ name }),
    }),
  deleteSpeaker: (id: number) =>
    request<{ deleted: number; unassigned_segments: number }>(`/api/speakers/${id}`, {
      method: "DELETE",
    }),
  deleteVoiceprint: (speakerId: number, printId: number) =>
    request<{ deleted: number }>(`/api/speakers/${speakerId}/voiceprints/${printId}`, {
      method: "DELETE",
    }),
  mergeSpeakers: (ids: [number, number]) =>
    request<{ target_id: number; name: string; moved_segments: number }>(
      "/api/speakers/merge",
      { method: "POST", body: JSON.stringify({ speaker_ids: ids }) },
    ),
  voiceprintAudioUrl: (speakerId: number, printId: number) =>
    `${getServerUrl()}/api/speakers/${speakerId}/voiceprints/${printId}/audio`,
};
