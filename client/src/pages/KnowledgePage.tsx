import { useEffect, useRef, useState } from "react";

import { api } from "../api/rest";
import { indexPending, type IndexProgress, type SearchApi } from "../llm/search";
import { loadLlmSettings } from "../llm/settings";
import type { KnowledgeSourceStatus, KnowledgeStatusDto } from "../types";

/** Скорость bge-m3 через Ollama на ПК, кусков в секунду — замер 17.09.2026.
 *  Нужна только для первой оценки до начала индексации: дальше время считается
 *  по тому, как быстро идёт на самом деле. Узкое место — Ollama, не видеокарта:
 *  на каждый текст уходит ~0.2 с накладных, какой бы он ни был длины. */
const CHUNKS_PER_SECOND_ESTIMATE = 4;

const STATUS_LABEL: Record<KnowledgeSourceStatus, string> = {
  indexed: "найдётся поиском",
  waiting: "ждёт индексации",
  not_ready: "встреча не закончена",
  empty: "текста нет",
};

export function formatDuration(seconds: number): string {
  if (seconds < 60) return "меньше минуты";
  const минуты = Math.round(seconds / 60);
  return минуты < 60 ? `около ${минуты} мин` : `около ${Math.floor(минуты / 60)} ч ${минуты % 60} мин`;
}

/** Сколько осталось: по скорости, с которой идёт на самом деле; до первых кусков — по замеру. */
export function remainingSeconds(done: number, total: number, elapsedMs: number): number {
  const скорость = done > 0 && elapsedMs > 0 ? done / (elapsedMs / 1000) : CHUNKS_PER_SECOND_ESTIMATE;
  return (total - done) / скорость;
}

/**
 * База знаний (пункты 5б и 7): что найдётся поиском, что ждёт индексации, и
 * индексация с прогрессом.
 *
 * Раньше индексация пряталась в первом поиске: новая встреча или документ
 * считались под крутилкой на кнопке «Найти», а смена модели эмбеддингов
 * превращала поиск в получасовое «зависание». Здесь она идёт открыто.
 */
export default function KnowledgePage() {
  const [status, setStatus] = useState<KnowledgeStatusDto | null>(null);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState<IndexProgress | null>(null);
  const начало = useRef(0);
  const model = loadLlmSettings().embedModel;
  // Перечитать состояние — после индексации; счётчик вместо функции в зависимостях эффекта
  const [версия, перечитать] = useState(0);

  useEffect(() => {
    api.knowledgeStatus(model).then(setStatus).catch((exc: Error) => setError(exc.message));
  }, [model, версия]);

  const источники = status ? [...status.meetings, ...status.documents] : [];
  const найдётся = источники.filter((и) => и.status === "indexed").length;
  const ждут = источники.filter((и) => и.status === "waiting");
  const кусковЖдёт = ждут.reduce((сумма, и) => сумма + и.chunks_waiting, 0);

  async function index() {
    const searchApi: SearchApi = {
      pending: (m) => api.searchPending(m),
      index: (body) => api.searchIndex(body),
      query: (body) => api.searchQuery(body),
    };
    setError("");
    начало.current = Date.now();
    setProgress({ chunksDone: 0, chunksTotal: кусковЖдёт, source: "" });
    try {
      await indexPending(searchApi, loadLlmSettings(), model, setProgress);
    } catch (exc) {
      // Сохранённое не пропадает: повторный запуск продолжит со следующего источника
      setError(`${(exc as Error).message} Уже посчитанное сохранено — можно продолжить.`);
    } finally {
      setProgress(null);
      перечитать((в) => в + 1);
    }
  }

  return (
    <div className="content">
      <h1>База знаний</h1>
      <p className="page-sub">
        Всё, по чему ищет поиск: встречи и документы. Чтобы кусок нашёлся, для него
        считается вектор моделью эмбеддингов — это и есть индексация.
      </p>
      {error && <div className="banner error">{error}</div>}

      {status && (
        <div className="card settings-block">
          <div className="meta">Модель эмбеддингов: {model}</div>
          <div style={{ marginTop: 6 }}>
            Найдётся поиском: <strong>{найдётся}</strong> из {источники.length}
            {ждут.length > 0 && (
              <>
                {" "}· ждут индексации: <strong>{ждут.length}</strong> ({кусковЖдёт} кусков,{" "}
                {formatDuration(remainingSeconds(0, кусковЖдёт, 0))})
              </>
            )}
          </div>
          {progress ? (
            <div className="banner info" style={{ marginTop: 10 }}>
              <span className="spinner" /> Кусок {progress.chunksDone} из {progress.chunksTotal}
              {progress.source && ` · «${progress.source}»`} · осталось{" "}
              {formatDuration(remainingSeconds(progress.chunksDone, progress.chunksTotal, Date.now() - начало.current))}
            </div>
          ) : (
            <button
              className="btn primary"
              style={{ marginTop: 10 }}
              disabled={ждут.length === 0}
              onClick={() => void index()}
            >
              {ждут.length === 0 ? "Всё проиндексировано" : "Проиндексировать"}
            </button>
          )}
        </div>
      )}

      {источники.length > 0 && (
        <div className="card settings-block">
          {status?.meetings.map((встреча) => (
            <SourceRow key={`m${встреча.id}`} kind="Встреча" source={встреча} />
          ))}
          {status?.documents.map((документ) => (
            <SourceRow key={`d${документ.id}`} kind="Документ" source={документ} />
          ))}
        </div>
      )}
    </div>
  );
}

function SourceRow({ kind, source }: { kind: string; source: KnowledgeStatusDto["meetings"][number] | KnowledgeStatusDto["documents"][number] }) {
  const кусков = source.status === "indexed" ? source.chunks : source.chunks_waiting;
  return (
    <div className="list-item" style={{ marginTop: 8 }} data-status={source.status}>
      <div className="grow">
        <div>{source.title}</div>
        <div className="meta">
          {kind} · {STATUS_LABEL[source.status]}
          {кусков > 0 && ` · ${кусков} кусков`}
        </div>
      </div>
    </div>
  );
}
