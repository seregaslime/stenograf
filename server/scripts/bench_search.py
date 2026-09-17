"""Поиск по встречам через индекс HNSW против точного перебора: скорость и точность.

Поиск (app/search.py) идёт через индекс HNSW: он быстрый на больших объёмах, но
приблизительный — изредка отдаёт не самые близкие куски. Бенчмарк меряет ту
самую функцию, которую зовёт сервер, двумя способами на одних данных:
  - перебор — индекс запрещён, база сравнивает вопрос со всеми кусками. Это
    точный ответ и заодно время, которое было бы без индекса;
  - индекс — при нескольких значениях hnsw.ef_search (сколько кандидатов
    смотрит индекс): больше кандидатов — точнее, но медленнее.
Точность (recall@5) — сколько из пяти кусков перебора индекс тоже нашёл.
По этим числам выбирается search.EF_SEARCH.

Векторы синтетические, 1024 числа как у bge-m3, но не равномерный шум: у
настоящих эмбеддингов разговоры группируются по темам, а на равномерном шуме
индекс теряет точность заметно сильнее, чем на живых данных. Поэтому векторы —
темы (центры) плюс разброс вокруг них. Точность всё равно оценка, а не факт:
на живых векторах её надо перепроверить.

База — тестовая (stenograf_test), куски складываются под отдельным именем
модели и удаляются в конце. Рабочая база не трогается. Числа зависят от машины
и от кэша базы (shared_buffers): индекс, не влезающий в кэш, даёт редкие
медленные запросы — это видно по p95.

Запуск:
    .venv/bin/python scripts/bench_search.py
    .venv/bin/python scripts/bench_search.py --sizes 1000,10000 --ef 40,100 --queries 20
"""
import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

# До импорта app: бенчмарк пишет сотни мегабайт векторов, рабочей базе они ни к чему
os.environ.setdefault(
    "STENOGRAF_DATABASE_URL",
    "postgresql+psycopg://stenograf:stenograf@127.0.0.1:5432/stenograf_test",
)

from pgvector.psycopg import register_vector  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app import search  # noqa: E402
from app.db.database import engine, init_db, session_scope  # noqa: E402
from app.db.models import Chunk, Meeting, Segment  # noqa: E402

DIM = 1024  # bge-m3
MODEL = "bench-bge-m3"
TOP_K = 5
TOPICS = 300
SPREAD = 0.9  # разброс вокруг темы: близость внутри темы ~0.5–0.6, как у соседних кусков разговора
# Прогрев длинный: индекс на 50 тысяч кусков весит ~400 МБ и в кэш базы (128 МБ
# по умолчанию) не помещается — первые запросы читают его страницы с диска.
WARMUP = 10


def make_vectors(n: int, rng: np.random.Generator, centers: np.ndarray) -> np.ndarray:
    v = centers[rng.integers(0, len(centers), n)] + SPREAD * rng.standard_normal((n, DIM))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def add_chunks(vectors: np.ndarray, first: int, meeting_id: int, segment_id: int) -> None:
    """Двоичный COPY: сотня тысяч INSERT по одному шла бы десятки минут.

    start_s у каждого куска свой — по нему куски узнаются в выдаче при подсчёте
    точности: текст и встреча у всех одинаковые.
    """
    raw = engine.raw_connection()
    try:
        conn = raw.driver_connection
        register_vector(conn)
        with conn.cursor() as cur:
            with cur.copy(
                "COPY chunks (meeting_id, first_segment_id, last_segment_id, start_s, text, model, vector) "
                "FROM STDIN WITH (FORMAT BINARY)"
            ) as copy:
                copy.set_types(["int4", "int4", "int4", "float8", "text", "varchar", "vector"])
                for i, v in enumerate(vectors, start=first):
                    copy.write_row((meeting_id, segment_id, segment_id, float(i), "кусок", MODEL, v))
        conn.commit()
    finally:
        raw.close()


def run_sql(sql: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(sql))


def measure(queries: np.ndarray, *, exact: bool) -> tuple[list[float], list[set[float]]]:
    """Время и найденные куски на каждый вопрос. Первые WARMUP — прогрев кэша."""
    times, found = [], []
    for i, q in enumerate(queries):
        with session_scope() as db:
            if exact:
                db.execute(text("SET LOCAL enable_indexscan = off"))
            start = time.perf_counter()
            rows = search.search_by_vector(db, MODEL, q.tolist(), TOP_K)
            elapsed = (time.perf_counter() - start) * 1000
        if i >= WARMUP:
            times.append(elapsed)
            found.append({r["start_s"] for r in rows})
    return times, found


def med_p95(times: list[float]) -> str:
    return f"{statistics.median(times):>6.1f} /{np.percentile(times, 95):>6.1f} мс"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sizes", default="10000,50000,100000",
                        help="объёмы кусков через запятую, по возрастанию")
    parser.add_argument("--ef", default="40,100,200", help="значения hnsw.ef_search через запятую")
    parser.add_argument("--queries", type=int, default=30, help="запросов на каждый объём")
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    efs = [int(e) for e in args.ef.split(",")]

    init_db()
    rng = np.random.default_rng(42)
    centers = rng.standard_normal((TOPICS, DIM))
    with session_scope() as db:
        meeting = Meeting(title="бенчмарк поиска", status="done")
        db.add(meeting)
        db.flush()
        segment = Segment(meeting_id=meeting.id, channel="mic", start_s=0, end_s=1, text="кусок")
        db.add(segment)
        db.flush()
        meeting_id, segment_id = meeting.id, segment.id

    index = f"{search.INDEX_PREFIX}{DIM}"
    print(f"{'кусков':>7} | {'перебор мед/p95':>15} | {'ef':>4} {'индекс мед/p95':>15} "
          f"{'точность':>8} | сборка, размер индекса")
    default_ef = search.EF_SEARCH
    try:
        loaded = 0
        for size in sizes:
            # Индекс снимается на время заливки и строится заново — как его строит
            # миграция по уже лежащим векторам; вставка через живой индекс шла бы
            # в разы дольше и мерила бы не то.
            run_sql(f"DROP INDEX IF EXISTS {index}")
            add_chunks(make_vectors(size - loaded, rng, centers), loaded, meeting_id, segment_id)
            loaded = size
            start = time.perf_counter()
            with session_scope() as db:
                search.ensure_index(db, DIM)
            build_s = time.perf_counter() - start
            run_sql("ANALYZE chunks")
            with engine.connect() as conn:
                index_size = conn.execute(
                    text(f"SELECT pg_size_pretty(pg_relation_size('{index}'))")).scalar()

            queries = make_vectors(args.queries + WARMUP, rng, centers)
            # Индекс меряется первым: перебор прогоняет через кэш всю таблицу и
            # вытесняет из него страницы индекса. На сервере перебора не бывает,
            # и в первом прогоне этот порядок сделал индекс «медленным, как перебор».
            by_ef = {}
            for ef in efs:
                search.EF_SEARCH = ef
                by_ef[ef] = measure(queries, exact=False)
            exact_ms, exact_found = measure(queries, exact=True)
            for n, ef in enumerate(efs):
                index_ms, index_found = by_ef[ef]
                recall = statistics.mean(len(a & b) / TOP_K for a, b in zip(index_found, exact_found))
                head = (f"{size:>7} | {med_p95(exact_ms)}" if n == 0 else f"{'':>7} | {'':>15}")
                tail = f"{build_s:.0f} с, {index_size}" if n == 0 else ""
                print(f"{head} | {ef:>4} {med_p95(index_ms)} {recall:>8.2f} | {tail}", flush=True)
    finally:
        search.EF_SEARCH = default_ef
        run_sql(f"DROP INDEX IF EXISTS {index}")
        with session_scope() as db:
            db.query(Chunk).filter(Chunk.model == MODEL).delete()
            db.query(Segment).filter(Segment.id == segment_id).delete()
            db.query(Meeting).filter(Meeting.id == meeting_id).delete()
        # Удалённые строки лежат в таблице мёртвыми, пока их не вычистят: после
        # пары прогонов перебор на тысяче кусков шёл 547 мс вместо 6 — продирался
        # через сотни тысяч удалённых векторов. Следующий замер врал бы.
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("VACUUM FULL ANALYZE chunks"))


if __name__ == "__main__":
    main()
