import { getServerUrl } from "../store";
import type { HealthDto, MeetingDetail, MeetingListItem, SpeakerDto } from "../types";

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
  mergeSpeakers: (sourceId: number, targetId: number) =>
    request<{ target_id: number; moved_segments: number }>("/api/speakers/merge", {
      method: "POST",
      body: JSON.stringify({ source_id: sourceId, target_id: targetId }),
    }),
  sampleUrl: (sampleId: number) => `${getServerUrl()}/api/samples/${sampleId}`,
};
