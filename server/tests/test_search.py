"""Нарезка разговора на куски и устойчивость поиска к чужим размерностям.

Векторы считает приложение, здесь их нет вовсе. На сервере осталось то, для
чего модель не нужна: нарезка (она про содержимое встречи) и сравнение готовых
векторов. Качество поиска проверяется не здесь, а замером на эталоне
(scripts/eval_search.py) — тест отвечает «работает/сломано», а не «насколько
хорошо».
"""
import numpy as np

from app import search
from app.db.models import Chunk, Meeting, Segment


def _segment(i: int, text: str, start: float = 0.0) -> Segment:
    return Segment(id=i, meeting_id=1, channel="mic", start_s=start, end_s=start + 1, text=text)


# ------------------------------------------------------------------ нарезка

def test_chunks_glue_replicas_until_limit():
    """Короткие реплики склеиваются в кусок: вектор от «Да-да, согласен» — шум."""
    сегменты = [_segment(i, "тридцать символов ровно тут да", i) for i in range(1, 5)]
    куски = search.build_chunks(сегменты, max_chars=60)
    assert len(куски) == 2
    assert куски[0]["first_segment_id"] == 1 and куски[0]["last_segment_id"] == 2


def test_chunk_keeps_replica_whole():
    """Граница куска проходит по реплике, а не по символу: разрезанная посреди
    фразы реплика теряет смысл вместе с вектором."""
    длинная = "а" * 200
    куски = search.build_chunks([_segment(1, длинная), _segment(2, "хвост")], max_chars=50)
    assert куски[0]["text"] == длинная
    assert куски[1]["text"] == "хвост"


def test_chunk_skips_empty_segments():
    """Пустые реплики в кусок не попадают — иначе в тексте копятся лишние пробелы."""
    куски = search.build_chunks([_segment(1, "  "), _segment(2, "есть текст")], max_chars=500)
    assert len(куски) == 1 and куски[0]["text"] == "есть текст"


def test_no_chunks_from_silence():
    """Встреча без единой реплики кусков не даёт и к модели не ходит."""
    assert search.build_chunks([], max_chars=500) == []


# ------------------------------------------------- чужие размерности в базе

def _положить(db, вектор: list[float], meeting_id: int = 1) -> None:
    """Кусок со всей роднёй: встреча и сегмент, на который он ссылается.

    Сегмент здесь не для красоты — на него смотрит внешний ключ. Пока база была
    SQLite, она ключи по умолчанию не проверяла, и кусок спокойно ссылался в
    пустоту; PostgreSQL такую вставку отбивает.
    """
    встреча = Meeting(id=meeting_id, title="Планёрка", status="done")
    db.add(встреча)
    сегмент = _segment(meeting_id, "про деньги и сроки")
    сегмент.meeting_id = meeting_id
    db.add(сегмент)
    db.flush()
    db.add(Chunk(
        meeting_id=meeting_id, model="bge-m3", text="про деньги и сроки",
        first_segment_id=сегмент.id, last_segment_id=сегмент.id, start_s=0.0,
        vector=вектор,
    ))
    db.flush()


def test_чужая_размерность_не_роняет_поиск(db_session):
    """Модель сменили, а пересчёт не прошёл целиком: старые векторы трёхмерные,
    новый запрос пятимерный. pgvector отказывается сравнивать векторы разной
    длины — ошибкой посреди запроса человека, если не отсеять их заранее."""
    _положить(db_session, [1.0, 0.0, 0.0])
    найдено = search.search_by_vector(
        db_session, "bge-m3", [1.0, 0.0, 0.0, 0.0, 0.0], limit=5,
    )
    assert найдено == []


def test_уцелевшие_куски_ищутся_несмотря_на_чужие(db_session):
    """Модель перекачали новой версией, пересчёт прошёл наполовину: чужие по
    длине куски отсеиваются, а свои находятся, — вместо отказа всего запроса."""
    _положить(db_session, [1.0, 0.0, 0.0], meeting_id=1)
    _положить(db_session, [1.0, 1.0, 1.0, 1.0, 1.0], meeting_id=2)

    найдено = search.search_by_vector(db_session, "bge-m3", [1.0, 0.0, 0.0], limit=5)
    assert [к["meeting_id"] for к in найдено] == [1]


def test_база_ранжирует_так_же_как_прежний_перебор_в_numpy(db_session):
    """Переезд на pgvector не должен поменять выдачу: те же векторы — тот же
    порядок и те же близости, что давало перемножение матрицы в numpy.

    Векторы случайные, но с фиксированным зерном: проверяется арифметика
    сравнения, а не смысл, и порядок должен совпасть до последнего куска.
    """
    rng = np.random.default_rng(17)
    векторы = rng.standard_normal((12, 16)).astype(np.float32)
    for номер, вектор in enumerate(векторы, start=1):
        _положить(db_session, list(вектор / np.linalg.norm(вектор)), meeting_id=номер)
    запрос = rng.standard_normal(16).astype(np.float32)

    найдено = search.search_by_vector(db_session, "bge-m3", list(запрос), limit=5)

    нормированные = векторы / np.linalg.norm(векторы, axis=1, keepdims=True)
    близости = нормированные @ (запрос / np.linalg.norm(запрос))
    ожидаемые = np.argsort(-близости)[:5] + 1
    assert [к["meeting_id"] for к in найдено] == list(ожидаемые)
    assert [к["similarity"] for к in найдено] == [round(float(близости[i - 1]), 3) for i in ожидаемые]
