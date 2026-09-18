import { useEffect, useRef, useState } from "react";

import { api } from "../api/rest";
import { startIndexing, useIndexing } from "../llm/indexing";
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

/** Число со словом в нужной форме: plural(3, ["кусок", "куска", "кусков"]) → «3 куска». */
export function plural(n: number, [один, два, пять]: [string, string, string]): string {
  const десятки = n % 100, единицы = n % 10;
  if (десятки >= 11 && десятки <= 14) return `${n} ${пять}`;
  if (единицы === 1) return `${n} ${один}`;
  if (единицы >= 2 && единицы <= 4) return `${n} ${два}`;
  return `${n} ${пять}`;
}

const КУСКИ: [string, string, string] = ["кусок", "куска", "кусков"];

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
  const [uploading, setUploading] = useState(false);
  const выбор = useRef<HTMLInputElement>(null);
  const model = loadLlmSettings().embedModel;
  // Индексация общая на приложение: она могла начаться на этом экране, в поиске
  // на истории или сама — а показать её нужно везде одинаково
  const индексация = useIndexing();
  const progress = индексация.progress;
  const идёт = progress !== null;
  // Перечитать состояние — после индексации; счётчик вместо функции в зависимостях эффекта
  const [версия, перечитать] = useState(0);

  useEffect(() => {
    api.knowledgeStatus(model).then(setStatus).catch((exc: Error) => setError(exc.message));
  }, [model, версия]);

  // Сводка «найдётся N из M» меняется, когда индексация заканчивается, — чьей
  // бы она ни была: своей, поисковой или самостоятельной. Спрашивать сервер на
  // её начале незачем: в этот момент у него ничего не изменилось.
  const шлаИндексация = useRef(false);
  useEffect(() => {
    if (шлаИндексация.current && !идёт) перечитать((в) => в + 1);
    шлаИндексация.current = идёт;
  }, [идёт]);

  const источники = status ? [...status.meetings, ...status.documents] : [];
  const найдётся = источники.filter((и) => и.status === "indexed").length;
  const ждут = источники.filter((и) => и.status === "waiting");
  const кусковЖдёт = ждут.reduce((сумма, и) => сумма + и.chunks_waiting, 0);
  const другойМоделью = ждут.filter((и) => и.chunks_other_models > 0);

  async function upload(file: File) {
    setUploading(true);
    setError("");
    try {
      await api.uploadDocument(file.name, new Uint8Array(await file.arrayBuffer()));
      перечитать((в) => в + 1);
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setUploading(false);
      if (выбор.current) выбор.current.value = "";  // тот же файл можно выбрать снова
    }
  }

  async function remove(id: number, title: string) {
    if (!confirm(`Удалить документ «${title}» из базы знаний?`)) return;
    try {
      await api.deleteDocument(id);
      перечитать((в) => в + 1);
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  async function index() {
    setError("");
    await startIndexing(model, кусковЖдёт);
  }

  return (
    <div className="content">
      <h1>База знаний</h1>
      <p className="page-sub">
        Всё, по чему ищет поиск: встречи и документы. Чтобы кусок нашёлся, для него
        считается вектор моделью эмбеддингов — это и есть индексация.
      </p>
      {(error || индексация.error) && (
        <div className="banner error">{error || индексация.error}</div>
      )}

      {/* Сменили модель эмбеддингов — векторы прежней поиск не видит. Молча
          пересчитывать при первом поиске значило бы получасовое «зависание»;
          здесь человек видит, сколько это займёт, и решает сам. */}
      {другойМоделью.length > 0 && !progress && (
        <div className="banner warn">
          {другойМоделью.length === 1
            ? "Один источник посчитан другой моделью эмбеддингов"
            : `${plural(другойМоделью.length, ["источник", "источника", "источников"])} посчитаны другой моделью эмбеддингов`}
          , а выбрана «{model}». Поиск их не найдёт, пока они не пересчитаны. Пересчёт
          заменит прежние векторы: вернётесь к старой модели — считать придётся заново.
        </div>
      )}

      {status && (
        <div className="card settings-block">
          <div className="meta">Модель эмбеддингов: {model}</div>
          <div style={{ marginTop: 6 }}>
            Найдётся поиском: <strong>{найдётся}</strong> из {источники.length}
            {ждут.length > 0 && (
              <>
                {" "}· ждут индексации: <strong>{ждут.length}</strong> ({plural(кусковЖдёт, КУСКИ)},{" "}
                {formatDuration(remainingSeconds(0, кусковЖдёт, 0))})
              </>
            )}
          </div>
          {progress ? (
            <div className="banner info" style={{ marginTop: 10 }}>
              <span className="spinner" /> Кусок {progress.chunksDone} из {progress.chunksTotal}
              {progress.source && ` · «${progress.source}»`} · осталось{" "}
              {formatDuration(remainingSeconds(progress.chunksDone, progress.chunksTotal, Date.now() - индексация.startedAt))}
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

      <div className="card settings-block">
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ flex: 1 }} className="hint">
            Свои документы — регламенты, ТЗ, заметки в .txt, .md, .docx и .pdf —
            поиск найдёт вместе со встречами
          </span>
          <input
            ref={выбор}
            type="file"
            accept=".txt,.md,.docx,.pdf"
            hidden
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void upload(file);
            }}
          />
          <button className="btn small" disabled={uploading} onClick={() => выбор.current?.click()}>
            {uploading ? <span className="spinner" /> : "Загрузить документ"}
          </button>
        </div>
        {status?.meetings.map((встреча) => (
          <SourceRow key={`m${встреча.id}`} kind="Встреча" source={встреча} />
        ))}
        {status?.documents.map((документ) => (
          <SourceRow
            key={`d${документ.id}`}
            kind="Документ"
            source={документ}
            onDelete={() => void remove(документ.id, документ.title)}
          />
        ))}
      </div>
    </div>
  );
}

function SourceRow({
  kind,
  source,
  onDelete,
}: {
  kind: string;
  source: KnowledgeStatusDto["meetings"][number] | KnowledgeStatusDto["documents"][number];
  /** Только у документа: встречи удаляются в истории вместе с транскриптом. */
  onDelete?: () => void;
}) {
  const кусков = source.status === "indexed" ? source.chunks : source.chunks_waiting;
  return (
    <div className="list-item" style={{ marginTop: 8 }} data-status={source.status}>
      <div className="grow">
        <div>{source.title}</div>
        <div className="meta">
          {kind} · {STATUS_LABEL[source.status]}
          {кусков > 0 && ` · ${plural(кусков, КУСКИ)}`}
          {source.chunks_other_models > 0 && source.status !== "indexed" && " · посчитан другой моделью"}
        </div>
      </div>
      {onDelete && (
        <button className="btn small danger" onClick={onDelete}>
          Удалить
        </button>
      )}
    </div>
  );
}
