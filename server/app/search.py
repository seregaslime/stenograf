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
а память сервера — 3.9 ГБ на всё.

Поиск идёт через индекс HNSW (решение Сергея и куратора, 17.09.2026). Что он
даёт и чего стоит — замер scripts/bench_search.py, векторы по 1024 числа:
  - скорость поиска: на Маке на максимуме (8 ядер, кэш базы 1.5–2 ГБ) индекс и
    перебор равны — 100 тысяч кусков по 4–6 мс, 300 тысяч по 12–13 мс. Перебор
    так быстр, потому что векторы лежат в строке и база делит его на процессы;
  - сборка — один раз, при миграции по уже лежащим векторам; новые куски
    встраиваются в готовый индекс по одному. Время сборки упирается в память
    maintenance_work_mem: 100 тысяч — 21 с при 2.5 ГБ и 26 минут при 64 МБ на
    4 ядрах; 300 тысяч — 14.5 минут, индекс (2.3 ГБ) в память сборки не влез.
    Пока индекс собирается, запись кусков ждёт;
  - место: 781 МБ на 100 тысяч кусков, 2.3 ГБ на 300 тысяч;
  - точность: индекс приблизительный, отсюда EF_SEARCH ниже.
"""
import logging
import re

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .config import Settings
from .db.models import Chunk, Document, Meeting, Segment

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


# Конец предложения: точка, вопрос, восклицание или многоточие и пробел за ними
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def document_chunks(text: str, max_chars: int) -> list[str]:
    """Режет документ на куски примерно по max_chars символов.

    Границы — по абзацам (пустая строка), как у встречи по репликам: абзац в
    регламенте — законченная мысль, и разрезанный посредине он теряет смысл
    вместе с вектором. Короткие абзацы склеиваются — заголовок в одну строку
    сам по себе ничего не значит. Абзац длиннее max_chars режется по
    предложениям, а предложение без точек, длиннее max_chars, — по словам:
    в выгрузках из таблиц и логов бывает и такое.
    """
    части: list[str] = []
    for абзац in re.split(r"\n\s*\n", text):
        абзац = " ".join(абзац.split())
        if not абзац:
            continue
        if len(абзац) <= max_chars:
            части.append(абзац)
            continue
        for предложение in _SENTENCE_END.split(абзац):
            while len(предложение) > max_chars:
                разрез = предложение.rfind(" ", 0, max_chars)
                разрез = разрез if разрез > 0 else max_chars
                части.append(предложение[:разрез].strip())
                предложение = предложение[разрез:].strip()
            if предложение:
                части.append(предложение)

    куски: list[str] = []
    for часть in части:
        if куски and len(куски[-1]) + 1 + len(часть) <= max_chars:
            куски[-1] = f"{куски[-1]}\n{часть}"
        else:
            куски.append(часть)
    return куски


# Индекс HNSW строится на каждую длину вектора отдельно: индексу нужна одна
# длина, а в колонке лежат векторы разных моделей эмбеддингов. Длиннее 2000
# чисел pgvector индекс не строит — вектор должен уместиться в страницу индекса
# (8 КБ по 4 байта на число). Такие модели (qwen3-embedding:4b — 2560 чисел)
# ищутся перебором: медленнее на больших объёмах, но точно.
INDEX_MAX_DIMS = 2000
INDEX_PREFIX = "chunks_vector_hnsw_"
# Сколько кандидатов смотрит индекс (у pgvector по умолчанию 40). Больше —
# точнее. Замер scripts/bench_search.py, доля настоящих пяти ближайших, которую
# нашёл индекс: 300 тысяч кусков — 0.84 при 40, 0.93 при 100, 1.00 при 200;
# 100 тысяч — 0.97 при 40, 1.00 при 100 и 200. Медленнее при 200 поиск не стал
# (13 мс против 24 при 100 — в пределах прогрева кэша между прогонами).
EF_SEARCH = 200


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


def pending_documents(db: Session, cfg: Settings, model: str,
                      owner_id: int | None = None) -> list[dict]:
    """Документы, которым нужны векторы ЭТОЙ модели, с нарезанными кусками.

    Как pending_chunks для встреч: считает векторы приложение, нарезка — здесь.
    """
    готовые = set(db.scalars(
        select(Chunk.document_id).where(Chunk.model == model, Chunk.document_id.is_not(None)).distinct()
    ))
    запрос = select(Document)
    if owner_id is not None:
        запрос = запрос.where(Document.owner_id == owner_id)
    ждут = []
    for документ in db.scalars(запрос.order_by(Document.id)):
        if документ.id in готовые:
            continue
        куски = [{"text": кусок} for кусок in document_chunks(документ.text, cfg.search_chunk_chars)]
        if куски:
            ждут.append({"document_id": документ.id, "title": документ.title, "chunks": куски})
    return ждут


def store_vectors(db: Session, model: str, source: Meeting | Document, куски: list[dict]) -> int:
    """Кладёт присланные векторы встречи или документа. Прежние куски источника удаляются.

    Кусок приходит вместе со своим вектором, а не пересчитывается здесь по
    номерам: у приложения и сервера нарезка могла бы разойтись на одну реплику
    (встречу дописали между запросами), и вектор лёг бы к чужому тексту —
    молча, потому что размерность совпала бы.
    """
    своё = (Chunk.document_id if isinstance(source, Document) else Chunk.meeting_id) == source.id
    for старый in db.scalars(select(Chunk).where(своё)):
        db.delete(старый)
    for dims in {len(кусок["vector"]) for кусок in куски}:
        ensure_index(db, dims)
    for кусок in куски:
        if isinstance(source, Document):
            откуда = {"document_id": source.id}
        else:
            откуда = {"meeting_id": source.id, "first_segment_id": кусок["first_segment_id"],
                      "last_segment_id": кусок["last_segment_id"], "start_s": кусок["start_s"]}
        db.add(Chunk(model=model, vector=_normalized(кусок["vector"]), text=кусок["text"], **откуда))
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
        # EF_SEARCH кандидатов и остановился — после отсева могло не
        # остаться ни одного. Итеративный поиск продолжает, пока не наберёт
        # limit, но отдаёт кандидатов в неточном порядке — поэтому пересортировка.
        db.execute(text(f"SET LOCAL hnsw.ef_search = {int(EF_SEARCH)}"))
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
    # Владелец — у встречи или у документа, смотря откуда кусок: источник ровно
    # один (ограничение chunks_one_source), второе соединение даёт NULL.
    владелец = "AND COALESCE(m.owner_id, d.owner_id) = :owner_id" if by_owner else ""
    return (
        f"SELECT c.meeting_id, m.title AS meeting_title, m.started_at, c.start_s, "
        f"c.document_id, d.title AS document_title, c.text, "
        f"{колонка} <#> CAST(:q AS vector) AS distance "
        f"FROM chunks c LEFT JOIN meetings m ON m.id = c.meeting_id "
        f"LEFT JOIN documents d ON d.id = c.document_id "
        f"WHERE c.model = :model AND vector_dims(c.vector) = {dims} {владелец} "
        f"ORDER BY distance LIMIT :limit"
    )


def _result(row) -> dict:
    return {
        # Кусок встречи или документа: поля другого источника — null
        "meeting_id": row.meeting_id,
        "meeting_title": row.meeting_title,
        "document_id": row.document_id,
        "document_title": row.document_title,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "start_s": row.start_s,
        "text": row.text,
        "similarity": round(-float(row.distance), 3),
    }
