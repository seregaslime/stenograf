"""Поиск по прошлым встречам: куски разговора, их векторы и подбор ближайших.

Зачем: договорённости растворяются. Через месяц никто не помнит, на какой
встрече решили перенести сроки, а протоколы приходится открывать по одному.

Как: каждый кусок разговора превращается в вектор (эмбеддинг), вопрос — тоже,
и мы берём куски с наибольшей близостью. Это ровно то, что SpeakerRegistry
делает с голосами (скалярное произведение L2-нормированных векторов), только
сравниваются не тембры, а смыслы.

Векторы считает приложение: у каждого своя модель эмбеддингов и свой адрес.
Здесь осталось то, для чего модель не нужна, — нарезка разговора на куски и
сравнение готовых векторов.

Сравнивает база, расширением pgvector (требование куратора от 17.09.2026). До
этого векторы лежали байтами, и сервер на каждый запрос поднимал в память все
куски человека и перемножал их в numpy. На сотнях кусков разницы в скорости
нет; она в том, что с базой знаний (пункт 5а) кусков станет на порядки больше,
а память сервера — 3.9 ГБ на всё. Индекса пока нет: база перебирает куски
точно, приблизительный индекс HNSW — после замера на настоящем объёме.
"""
import logging

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .config import Settings
from .db.models import Chunk, Meeting, Segment

log = logging.getLogger(__name__)

# Повторная индексация одной встречи безопасна: store_vectors сначала удаляет
# прежние куски. Раньше здесь стоял замок — эмбеддинги считались тут же, и два
# одновременных поиска успевали посчитать одну встречу дважды.


def build_chunks(segments: list[Segment], max_chars: int) -> list[dict]:
    """Склеивает подряд идущие реплики в куски примерно по max_chars символов.

    Границы кусков — по репликам, а не по символам: разрезанная посреди фразы
    реплика теряет смысл, а вместе с ним и вектор. Одна длинная реплика
    становится куском целиком, даже если она длиннее лимита.
    """
    куски: list[dict] = []
    текущие: list[Segment] = []
    длина = 0

    def закрыть() -> None:
        nonlocal текущие, длина
        if not текущие:
            return
        куски.append({
            "first_segment_id": текущие[0].id,
            "last_segment_id": текущие[-1].id,
            "start_s": текущие[0].start_s,
            "text": " ".join(s.text.strip() for s in текущие if s.text.strip()),
        })
        текущие, длина = [], 0

    for сегмент in segments:
        if not (сегмент.text or "").strip():
            continue
        текущие.append(сегмент)
        длина += len(сегмент.text)
        if длина >= max_chars:
            закрыть()
    закрыть()
    return [к for к in куски if к["text"]]


# Индекс HNSW строится на каждую длину вектора отдельно: индексу нужна одна
# длина, а в колонке лежат векторы разных моделей эмбеддингов. Длиннее 2000
# чисел pgvector индекс не строит — вектор должен уместиться в страницу индекса
# (8 КБ по 4 байта на число). Такие модели (qwen3-embedding:4b — 2560 чисел)
# ищутся перебором: медленнее на больших объёмах, но точно.
INDEX_MAX_DIMS = 2000
INDEX_PREFIX = "chunks_vector_hnsw_"


def ensure_index(db: Session, dims: int) -> None:
    """Индекс HNSW для векторов длины dims, если его ещё нет.

    Зовётся, когда приходят векторы: так индекс появляется у модели любой длины,
    которую человек выберет, а не только у тех, что предусмотрели заранее.
    Первые векторы новой длины — это единицы кусков, и строится он мгновенно;
    по уже лежащим векторам индексы строит миграция.

    Длина вписывается в текст запроса числом, а не параметром: это целое из
    len() присланного вектора, подставлять туда нечего. Замок нужен против
    двух одновременных индексаций — второй CREATE INDEX IF NOT EXISTS, начатый
    до конца первого, падает на уникальности имени.
    """
    dims = int(dims)
    if not 0 < dims <= INDEX_MAX_DIMS:
        return
    db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": 7_300_000 + dims})
    db.execute(text(
        f"CREATE INDEX IF NOT EXISTS {INDEX_PREFIX}{dims} ON chunks "
        f"USING hnsw ((vector::vector({dims})) vector_ip_ops) "
        f"WHERE vector_dims(vector) = {dims}"
    ))


def _normalized(vector: list[float]) -> np.ndarray:
    """L2-нормированный float32: после нормировки близость — обычное скалярное
    произведение, без деления на длины при каждом поиске."""
    v = np.asarray(vector, dtype=np.float32)
    норма = float(np.linalg.norm(v))
    return v / норма if норма else v


def pending_chunks(db: Session, cfg: Settings, model: str,
                   owner_id: int | None = None) -> list[dict]:
    """Встречи, которым нужны векторы ЭТОЙ модели, с готовыми кусками разговора.

    Модель приходит от приложения, а не берётся из настроек сервера: считает
    векторы теперь оно, у каждого своя модель эмбеддингов и свой адрес. Сервер
    перестал знать, чем считают, и спрашивать его об этом больше нельзя.

    Нарезка осталась здесь: она про содержимое встречи, а не про модель, и
    одинакова для всех.
    """
    готовые = set(db.scalars(
        select(Chunk.meeting_id).where(Chunk.model == model).distinct()
    ))
    запрос = select(Meeting).where(Meeting.status == "done")
    if owner_id is not None:
        запрос = запрос.where(Meeting.owner_id == owner_id)

    ждут = []
    for meeting in db.scalars(запрос):
        if meeting.id in готовые:
            continue
        segments = list(db.scalars(
            select(Segment).where(Segment.meeting_id == meeting.id).order_by(Segment.start_s)
        ))
        куски = build_chunks(segments, cfg.search_chunk_chars)
        if куски:
            ждут.append({"meeting_id": meeting.id, "title": meeting.title, "chunks": куски})
    return ждут


def store_vectors(db: Session, model: str, meeting: Meeting, куски: list[dict]) -> int:
    """Кладёт присланные векторы. Прежние куски этой встречи удаляются.

    Кусок приходит вместе со своим вектором, а не пересчитывается здесь по
    номерам: у приложения и сервера нарезка могла бы разойтись на одну реплику
    (встречу дописали между запросами), и вектор лёг бы к чужому тексту —
    молча, потому что размерность совпала бы.
    """
    for старый in db.scalars(select(Chunk).where(Chunk.meeting_id == meeting.id)):
        db.delete(старый)
    for dims in {len(кусок["vector"]) for кусок in куски}:
        ensure_index(db, dims)
    for кусок in куски:
        db.add(Chunk(
            meeting_id=meeting.id,
            model=model,
            vector=_normalized(кусок["vector"]),
            first_segment_id=кусок["first_segment_id"],
            last_segment_id=кусок["last_segment_id"],
            start_s=кусок["start_s"],
            text=кусок["text"],
        ))
    db.flush()
    return len(куски)


def search_by_vector(db: Session, model: str, вектор: list[float], limit: int,
                     owner_id: int | None = None) -> list[dict]:
    """Ближайшие куски к уже посчитанному вектору.

    Сравнение векторов модели не требует — это скалярное произведение. Поэтому
    считать эмбеддинги сервер разучился, а искать по ним умеет по-прежнему: так
    по сети едут килобайты вопроса, а не мегабайты матрицы.
    """
    q = _normalized(вектор)
    params = {"model": model, "q": str(q.tolist()), "limit": limit, "owner_id": owner_id}
    with_index = len(q) <= INDEX_MAX_DIMS
    if with_index:
        # Индекс находит ближайших среди ВСЕХ кусков этой длины, а чужая модель и
        # чужие встречи отсеиваются уже после. Обычный поиск по индексу отдал бы
        # hnsw.ef_search кандидатов (40) и остановился — после отсева могло не
        # остаться ни одного. Итеративный поиск продолжает, пока не наберёт
        # limit, но отдаёт кандидатов в неточном порядке — поэтому пересортировка.
        db.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
    строки = db.execute(text(search_sql(len(q), owner_id is not None)), params).all()
    return [_result(строка) for строка in sorted(строки, key=lambda r: r.distance)]


def search_sql(dims: int, by_owner: bool) -> str:
    """Запрос поиска. Отдельно — чтобы тест мог спросить у базы его план.

    Приведение к vector(dims) и условие на длину повторяют индекс дословно —
    иначе база его не узнает. Длина вписана числом, а не параметром: драйвер
    после пяти одинаковых запросов переходит на общий план, где параметр
    неизвестен, и база молча уходила бы в перебор. Подставлять туда нечего:
    это len() присланного вектора.

    Векторы другой длины (пересчёт модели не дошёл до конца) отсеиваются до
    сравнения: сравнивать их база отказывается с ошибкой. <#> — скалярное
    произведение со знаком минус: по возрастанию идут самые близкие.
    """
    dims = int(dims)
    колонка = f"c.vector::vector({dims})" if dims <= INDEX_MAX_DIMS else "c.vector"
    владелец = "AND m.owner_id = :owner_id" if by_owner else ""
    return (
        f"SELECT c.meeting_id, m.title, m.started_at, c.start_s, c.text, "
        f"{колонка} <#> CAST(:q AS vector) AS distance "
        f"FROM chunks c JOIN meetings m ON m.id = c.meeting_id "
        f"WHERE c.model = :model AND vector_dims(c.vector) = {dims} {владелец} "
        f"ORDER BY distance LIMIT :limit"
    )


def _result(row) -> dict:
    return {
        "meeting_id": row.meeting_id,
        "meeting_title": row.title,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "start_s": row.start_s,
        "text": row.text,
        "similarity": round(-float(row.distance), 3),
    }
