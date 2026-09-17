/**
 * Поиск по прошлым встречам и документам базы знаний со стороны приложения: считать векторы своей
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

/** Документ базы знаний, которому нужны векторы: у его кусков только текст. */
export interface PendingDocument {
  document_id: number;
  title: string;
  chunks: { text: string }[];
}

/** Что умеет сервер: отдать неиндексированное, принять векторы, найти по вектору. */
export interface SearchApi {
  pending(model: string): Promise<{ meetings: PendingMeeting[]; documents?: PendingDocument[] }>;
  index(
    body:
      | {
        model: string;
        meeting_id: number;
        chunks: (PendingMeeting["chunks"][number] & { vector: number[] })[];
      }
      | { model: string; document_id: number; chunks: { text: string; vector: number[] }[] },
  ): Promise<{ chunks: number }>;
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

/** Сколько кусков отправлять модели за раз. Замер 17.09.2026, bge-m3 на ПК,
 *  256 кусков: одним запросом 62–63 с, пачками по 32 — 61–65 с. Пачки ничего не
 *  стоят, а прогресс по ним обновляется раз в ~8 секунд вместо одного раза на
 *  весь документ, который на мегабайте считается 7 минут молча. */
export const EMBED_BATCH = 32;

/** Где сейчас индексация: куски всех источников и текущий источник. */
export interface IndexProgress {
  chunksDone: number;
  chunksTotal: number;
  /** Название встречи или документа, который считается сейчас; "" — готово. */
  source: string;
}

/**
 * Досчитывает векторы для встреч и документов, у которых их нет. Возвращает,
 * сколько кусков посчитано.
 *
 * Ленивая индексация, как и была: встреча могла пройти до появления поиска, а
 * модель — смениться. Проверка дешёвая, пересчёт идёт только там, где не хватает.
 * Упала посредине (модель недоступна) — повторный вызов продолжит с того
 * источника, на котором упала: уже отправленные сервер больше не отдаёт.
 */
export async function indexPending(
  api: SearchApi,
  settings: LlmSettings,
  model: string,
  onProgress: (progress: IndexProgress) => void = () => {},
): Promise<number> {
  const { meetings, documents = [] } = await api.pending(model);
  // Документы — после встреч: большой документ считается минутами, а новая
  // встреча в поиске нужнее вчерашнего регламента.
  const очередь = [
    ...meetings.map((в) => ({ title: в.title, texts: в.chunks.map((к) => к.text), meeting: в })),
    ...documents.map((д) => ({ title: д.title, texts: д.chunks.map((к) => к.text), document: д })),
  ];
  const всего = очередь.reduce((сумма, и) => сумма + и.texts.length, 0);
  if (всего === 0) return 0;

  const модель = embedder(settings);
  let готово = 0;
  for (const источник of очередь) {
    const векторы: number[][] = [];
    for (let i = 0; i < источник.texts.length; i += EMBED_BATCH) {
      onProgress({ chunksDone: готово, chunksTotal: всего, source: источник.title });
      const пачка = источник.texts.slice(i, i + EMBED_BATCH);
      векторы.push(...(await модель.embed(model, пачка)));
      готово += пачка.length;
    }
    // Источник уходит целиком, а не пачками: сервер заменяет его куски разом,
    // и встреча с половиной векторов считалась бы проиндексированной с дырой.
    // Кусок — вместе со своим вектором: пересчитывать нарезку на сервере нельзя,
    // встречу могли дописать, и вектор лёг бы к чужому тексту.
    if ("meeting" in источник && источник.meeting) {
      await api.index({
        model,
        meeting_id: источник.meeting.meeting_id,
        chunks: источник.meeting.chunks.map((к, i) => ({ ...к, vector: векторы[i] })),
      });
    } else if ("document" in источник && источник.document) {
      await api.index({
        model,
        document_id: источник.document.document_id,
        chunks: источник.document.chunks.map((к, i) => ({ text: к.text, vector: векторы[i] })),
      });
    }
  }
  onProgress({ chunksDone: всего, chunksTotal: всего, source: "" });
  return всего;
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
 * Ответ модели по уже найденным фрагментам.
 *
 * Отдельно от поиска намеренно: цитаты показываются сразу, а ответ догоняет —
 * человек читает, пока модель думает. Поэтому вызывающий сначала ищет, потом
 * зовёт это.
 *
 * Отвечает модель протокола: ответ по нескольким фрагментам ближе к резюме,
 * чем к реплике на лету. Решение живёт здесь одно на всех — иначе выбор роли
 * разъедется между страницей и модулем.
 */
export async function answerByFragments(
  llm: LlmRouter,
  question: string,
  results: SearchHit[],
): Promise<string> {
  if (results.length === 0) return "";
  const { system, prompt } = buildSearchAnswerPrompt(question, results);
  const answer = await llm.generate("summary", prompt, { system, temperature: 0.3 });
  return answer.trim();
}
