/**
 * Поиск по прошлым встречам со стороны приложения: считать векторы своей
 * моделью, отдать их серверу, спросить у него ближайшие и ответить по ним.
 *
 * Разделение проведено по границе «что зависит от модели». Нарезка разговора и
 * сравнение векторов остались на сервере: нарезка про содержимое встречи,
 * сравнение — скалярное произведение, модель для него не нужна. Здесь только
 * то, для чего нужна модель.
 */
import { OllamaClient } from "./ollama";
import { buildSearchAnswerPrompt } from "./prompts/searchAnswer";
import type { LlmRouter } from "./router";
import type { LlmSettings } from "./router";
import type { SearchHit } from "../types";

export interface PendingMeeting {
  meeting_id: number;
  title: string;
  chunks: { first_segment_id: number; last_segment_id: number; start_s: number; text: string }[];
}

/** Что умеет сервер: отдать неиндексированное, принять векторы, найти по вектору. */
export interface SearchApi {
  pending(model: string): Promise<{ meetings: PendingMeeting[] }>;
  index(body: {
    model: string;
    meeting_id: number;
    chunks: (PendingMeeting["chunks"][number] & { vector: number[] })[];
  }): Promise<{ chunks: number }>;
  query(body: { model: string; vector: number[]; limit?: number }): Promise<{ results: SearchHit[] }>;
}

/**
 * Эмбеддинги считает Ollama, а не выбранный провайдер: модель эмбеддингов — не
 * разговорная, у внешних API это отдельная услуга с отдельной тарификацией, и
 * смешивать их в одну настройку значило бы врать в интерфейсе.
 */
function embedder(settings: LlmSettings): OllamaClient {
  return new OllamaClient({ url: settings.ollamaUrl, keepAlive: settings.keepAlive });
}

/**
 * Досчитывает векторы для встреч, у которых их нет. Возвращает, сколько кусков
 * посчитано.
 *
 * Ленивая индексация, как и была: встреча могла пройти до появления поиска, а
 * модель — смениться. Проверка дешёвая, пересчёт идёт только там, где не хватает.
 */
export async function indexPending(
  api: SearchApi,
  settings: LlmSettings,
  model: string,
  onProgress: (готово: number, всего: number) => void = () => {},
): Promise<number> {
  const { meetings } = await api.pending(model);
  if (meetings.length === 0) return 0;

  let посчитано = 0;
  for (const [индекс, встреча] of meetings.entries()) {
    onProgress(индекс, meetings.length);
    const векторы = await embedder(settings).embed(
      model,
      встреча.chunks.map((к) => к.text),
    );
    // Кусок уходит обратно вместе со своим вектором: пересчитывать нарезку на
    // сервере нельзя — встречу могли дописать, и вектор лёг бы к чужому тексту.
    await api.index({
      model,
      meeting_id: встреча.meeting_id,
      chunks: встреча.chunks.map((к, i) => ({ ...к, vector: векторы[i] })),
    });
    посчитано += встреча.chunks.length;
  }
  onProgress(meetings.length, meetings.length);
  return посчитано;
}

/** Ближайшие куски к вопросу. Пустой вопрос — пустая выдача, без похода к модели. */
export async function searchMeetings(
  api: SearchApi,
  settings: LlmSettings,
  model: string,
  question: string,
  limit?: number,
): Promise<SearchHit[]> {
  const текст = question.trim();
  if (!текст) return [];
  const [вектор] = await embedder(settings).embed(model, [текст]);
  const { results } = await api.query({ model, vector: вектор, limit });
  return results;
}

/**
 * Ответ модели по найденным фрагментам плюс сами фрагменты.
 *
 * Цитаты возвращаются вместе с ответом и всегда показываются рядом: модель не
 * помнит встреч, она пересказывает показанное, и проверить её можно только по
 * тому, что видно на экране.
 */
export async function answerFromMeetings(
  api: SearchApi,
  llm: LlmRouter,
  settings: LlmSettings,
  model: string,
  question: string,
  limit?: number,
): Promise<{ answer: string; results: SearchHit[] }> {
  const results = await searchMeetings(api, settings, model, question, limit);
  if (results.length === 0) return { answer: "", results: [] };

  const { system, prompt } = buildSearchAnswerPrompt(question, results);
  // Ролью отвечает модель протокола: ответ по нескольким фрагментам ближе к
  // резюме, чем к реплике на лету.
  const answer = await llm.generate("summary", prompt, { system, temperature: 0.3 });
  return { answer: answer.trim(), results };
}
