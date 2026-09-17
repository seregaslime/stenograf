"""Когда поиску по встречам понадобится индекс: точный перебор против HNSW.

Поиск сравнивает векторы в базе расширением pgvector без индекса — база
перебирает все куски человека (app/search.py). Индекс HNSW быстрее на больших
объёмах, но приблизительный: изредка отдаёт не самые близкие куски. Бенчмарк
отвечает на вопрос, с какого числа кусков перебор перестаёт укладываться во
время, которое человек не замечает, и сколько точности индекс за это берёт.

Что меряется на каждом объёме:
  - настоящий запрос поиска (search.search_by_vector) — медиана и p95;
  - запрос через индекс HNSW — медиана, p95 и время построения индекса;
  - точность индекса (recall@5): сколько из настоящих пяти ближайших кусков
    индекс вернул.

Векторы синтетические, 1024 числа как у bge-m3, но не равномерный шум: у
настоящих эмбеддингов разговоры группируются по темам, а на равномерном шуме
индекс теряет точность заметно сильнее, чем на живых данных. Поэтому векторы —
темы (центры) плюс разброс вокруг них. Точность всё равно оценка, а не факт:
на живых векторах её надо перепроверить.

База — тестовая (stenograf_test), куски складываются под отдельным именем
модели и удаляются в конце. Рабочая база не трогается.

Запуск:
    .venv/bin/python scripts/bench_search.py
    .venv/bin/python scripts/bench_search.py --sizes 1000,10000 --queries 20
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


def make_vectors(n: int, rng: np.random.Generator, centers: np.ndarray) -> np.ndarray:
    v = centers[rng.integers(0, len(centers), n)] + SPREAD * rng.standard_normal((n, DIM))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def add_chunks(vectors: np.ndarray, meeting_id: int, segment_id: int) -> None:
    """Двоичный COPY: миллион строк INSERT по одной шёл бы часами."""
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
                for v in vectors:
                    copy.write_row((meeting_id, segment_id, segment_id, 0.0, "кусок", MODEL, v))
        conn.commit()
    finally:
        raw.close()


def timed(fn) -> tuple[float, object]:
    start = time.perf_counter()
    result = fn()
    return (time.perf_counter() - start) * 1000, result


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def exact_ids(q: np.ndarray) -> list[int]:
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(text(
            "SELECT id FROM chunks WHERE model = :m AND vector_dims(vector) = :d "
            "ORDER BY vector <#> CAST(:q AS vector) LIMIT :k"),
            {"m": MODEL, "d": DIM, "q": str(q.tolist()), "k": TOP_K})]


# Колонка без размерности, а индексу HNSW размерность нужна: индекс строится по
# выражению с приведением и только по одной модели. Запрос обязан повторить
# выражение дословно — иначе планировщик индекс не возьмёт.
INDEX_SQL = (f"CREATE INDEX bench_hnsw ON chunks USING hnsw ((vector::vector({DIM})) vector_ip_ops) "
             f"WHERE model = '{MODEL}'")
INDEX_QUERY = (f"SELECT id FROM chunks WHERE model = '{MODEL}' "
               f"ORDER BY vector::vector({DIM}) <#> CAST(:q AS vector({DIM})) LIMIT {TOP_K}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sizes", default="1000,10000,50000,100000",
                        help="объёмы кусков через запятую, по возрастанию")
    parser.add_argument("--queries", type=int, default=30, help="запросов на каждый объём")
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

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

    print(f"{'кусков':>8} | {'перебор мед':>11} {'p95':>6} | {'индекс мед':>10} {'p95':>6} "
          f"{'сборка':>8} | {'точность':>8} | план")
    try:
        loaded = 0
        for size in sizes:
            add_chunks(make_vectors(size - loaded, rng, centers), meeting_id, segment_id)
            loaded = size
            with engine.begin() as conn:
                conn.execute(text("ANALYZE chunks"))
            queries = make_vectors(args.queries + 3, rng, centers)

            exact_ms = []
            with session_scope() as db:
                for i, q in enumerate(queries):
                    ms, _ = timed(lambda: search.search_by_vector(db, MODEL, q.tolist(), TOP_K))
                    if i >= 3:  # первые — прогрев кэша страниц
                        exact_ms.append(ms)

            build_ms, _ = timed(lambda: _run(INDEX_SQL))
            index_ms, recalls = [], []
            with engine.connect() as conn:
                plan = " ".join(r[0] for r in conn.execute(
                    text("EXPLAIN " + INDEX_QUERY), {"q": str(queries[0].tolist())}))
                for i, q in enumerate(queries):
                    ms, rows = timed(lambda: conn.execute(text(INDEX_QUERY), {"q": str(q.tolist())}).all())
                    if i >= 3:
                        index_ms.append(ms)
                        recalls.append(len({r[0] for r in rows} & set(exact_ids(q))) / TOP_K)
            _run("DROP INDEX bench_hnsw")

            print(f"{size:>8} | {statistics.median(exact_ms):>9.1f}мс {percentile(exact_ms, 95):>6.1f} | "
                  f"{statistics.median(index_ms):>8.1f}мс {percentile(index_ms, 95):>6.1f} "
                  f"{build_ms / 1000:>7.1f}с | {statistics.mean(recalls):>8.2f} | "
                  f"{'индекс' if 'bench_hnsw' in plan else 'перебор'}", flush=True)
    finally:
        _run("DROP INDEX IF EXISTS bench_hnsw")
        with session_scope() as db:
            db.query(Chunk).filter(Chunk.model == MODEL).delete()
            db.query(Segment).filter(Segment.id == segment_id).delete()
            db.query(Meeting).filter(Meeting.id == meeting_id).delete()


def _run(sql: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(sql))


if __name__ == "__main__":
    main()
