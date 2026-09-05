import { getServerUrl, getToken } from "../store";
import type {
  AsrStateDto,
  HealthDto,
  LlmModelsDto,
  LlmSettings,
  OllamaModelsDto,
  LlmStateDto,
  MeetingDetail,
  MeetingListItem,
  SearchAnswer,
  SearchHit,
  PendingMeetingDto,
  SpeakerDto,
  SummarySaved,
} from "../types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getToken();
  const response = await fetch(getServerUrl() + path, {
    ...init,
    // Заголовки собираем после разбора init: раньше «...init» затирал их
    // целиком, и вызов со своими заголовками терял и тип содержимого, и токен.
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(init?.headers ?? {}),
    },
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
  // То же для Ollama: какие модели скачаны по ещё не сохранённому адресу
  probeOllama: (ollama_url: string) =>
    request<OllamaModelsDto>("/api/llm/ollama/models", {
      method: "POST",
      body: JSON.stringify({ ollama_url }),
    }),

  // Поиск по смыслу среди прошлых встреч; сервер сам доиндексирует новые
  /** Что осталось проиндексировать ЭТОЙ моделью: векторы считает приложение. */
  searchPending: (model: string) =>
    request<{ meetings: PendingMeetingDto[] }>(
      `/api/search/pending?model=${encodeURIComponent(model)}`,
    ),

  searchIndex: (body: {
    model: string;
    meeting_id: number;
    chunks: (PendingMeetingDto["chunks"][number] & { vector: number[] })[];
  }) =>
    request<{ meeting_id: number; chunks: number }>("/api/search/index", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  /** Поиск по уже посчитанному вектору вопроса: сравнение делает сервер. */
  searchQuery: (body: { model: string; vector: number[]; limit?: number }) =>
    request<{ results: SearchHit[] }>("/api/search/query", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  search: (q: string, limit?: number) =>
    request<{ results: SearchHit[] }>(
      `/api/search?q=${encodeURIComponent(q)}` + (limit ? `&limit=${limit}` : ""),
    ),

  // Ответ модели по найденному — отдельным запросом: поиск отвечает за доли
  // секунды, модель за секунды, и цитаты успевают появиться раньше ответа
  searchAnswer: (q: string, limit?: number) =>
    request<SearchAnswer>("/api/search/answer", {
      method: "POST",
      body: JSON.stringify({ q, limit }),
    }),

  meetings: () => request<MeetingListItem[]>("/api/meetings"),
  meeting: (id: number) => request<MeetingDetail>(`/api/meetings/${id}`),
  deleteMeeting: (id: number) =>
    request<{ deleted: number }>(`/api/meetings/${id}`, { method: "DELETE" }),
  /** Отдать серверу готовый протокол (его теперь составляет клиент) или
   *  причину неудачи — чтобы встреча не висела в «составляется» молча. */
  saveSummary: (id: number, body: { text?: string; error?: string; model?: string }) =>
    request<SummarySaved>(`/api/meetings/${id}/summary`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

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
  // source_id — какого профиля больше нет: целевой сервер выбирает сам
  // («Вы» → человеческое имя → больше реплик), и клиент заранее его не знает.
  mergeSpeakers: (ids: [number, number]) =>
    request<{
      target_id: number; name: string; moved_segments: number;
      source_id: number; was_named: string[];
    }>(
      "/api/speakers/merge",
      { method: "POST", body: JSON.stringify({ speaker_ids: ids }) },
    ),
  voiceprintAudioUrl: (speakerId: number, printId: number) =>
    `${getServerUrl()}/api/speakers/${speakerId}/voiceprints/${printId}/audio`,
};
