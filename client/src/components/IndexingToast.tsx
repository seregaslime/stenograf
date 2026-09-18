import { useIndexing } from "../llm/indexing";
import { formatDuration, remainingSeconds } from "../pages/KnowledgePage";

/**
 * Окошко в углу, пока идёт индексация.
 *
 * Зачем: считать векторы приложение теперь начинает само, и без этого окошка
 * человек видел бы только последствия — Ollama занята, вентилятор шумит, а
 * причина не названа. Окошко отвечает на «что происходит» и «сколько ещё».
 *
 * На «Базе знаний» не показывается: там та же индексация уже расписана
 * подробно, со списком источников, и вторая плашка о том же поверх первой —
 * шум.
 *
 * Гаснет само: индексация кончилась — состояние опустело — окошка нет. Своей
 * кнопки «закрыть» у него нет намеренно, закрывать нечего.
 */
export default function IndexingToast({ наЭкранеБазы }: { наЭкранеБазы: boolean }) {
  const { progress, startedAt } = useIndexing();
  if (!progress || наЭкранеБазы) return null;

  const осталось = remainingSeconds(progress.chunksDone, progress.chunksTotal, Date.now() - startedAt);
  return (
    <div className="toast">
      <span className="spinner" />
      <div>
        <div>
          Индексация{progress.source && `: «${progress.source}»`}
        </div>
        <div className="meta">
          кусок {progress.chunksDone} из {progress.chunksTotal}
          {осталось > 0 && ` · осталось ${formatDuration(осталось)}`}
        </div>
      </div>
    </div>
  );
}
