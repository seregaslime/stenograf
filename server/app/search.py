"""Поиск по прошлым встречам: куски разговора, их векторы и подбор ближайших.

Зачем: договорённости растворяются. Через месяц никто не помнит, на какой
встрече решили перенести сроки, а протоколы приходится открывать по одному.

Как: каждый кусок разговора превращается в вектор (эмбеддинг), вопрос — тоже,
и мы берём куски с наибольшей близостью. Это ровно то, что SpeakerRegistry
делает с голосами (`np.dot` по L2-нормированным векторам ECAPA), только
сравниваются не тембры, а смыслы.

Векторной СУБД (FAISS, sqlite-vec) здесь нет намеренно: на живых данных это
матрица в единицы мегабайт, полный перебор занимает миллисекунды, а лишнее
хранилище — это ещё одна зависимость и ещё один способ сломаться.
"""
import asyncio
import logging

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings
from .db.models import Chunk, Meeting, Segment
from .llm import prompts
from .llm.ollama_client import OllamaClient
from .llm.router import LlmRouter

log = logging.getLogger(__name__)

# Индексация идёт лениво, прямо в обработчике поиска, поэтому два одновременных
# запроса (двойной клик по «Найти», два открытых клиента) успевают увидеть одну
# и ту же неиндексированную встречу и посчитать её дважды — окно широкое, счёт
# эмбеддингов занимает секунды. Второй набор кусков лёг бы рядом с первым, и
# один разговор занял бы всю выдачу одинаковыми цитатами.
_замок = asyncio.Lock()


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


def _to_blob(vector: list[float]) -> bytes:
    """L2-нормированный float32 в BLOB: после нормировки близость — обычное
    скалярное произведение, без деления на длины при каждом поиске."""
    v = np.asarray(vector, dtype=np.float32)
    норма = float(np.linalg.norm(v))
    if норма:
        v = v / норма
    return v.tobytes()


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
    for кусок in куски:
        db.add(Chunk(
            meeting_id=meeting.id,
            model=model,
            vector=_to_blob(кусок["vector"]),
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
    q = np.frombuffer(_to_blob(вектор), dtype=np.float32)
    отбор = select(Chunk).where(Chunk.model == model)
    if owner_id is not None:
        отбор = отбор.join(Meeting, Chunk.meeting_id == Meeting.id).where(
            Meeting.owner_id == owner_id
        )
    куски = [к for к in db.scalars(отбор) if len(к.vector) == q.nbytes]
    if not куски:
        return []

    матрица = np.frombuffer(b"".join(к.vector for к in куски), dtype=np.float32)
    матрица = матрица.reshape(len(куски), -1)
    близости = матрица @ q
    лучшие = np.argsort(-близости)[:limit]
    return [_result(куски[i], float(близости[i])) for i in лучшие]


async def index_meeting(db: Session, cfg: Settings, meeting: Meeting) -> int:
    """Считает куски и векторы для одной встречи. Возвращает число кусков.

    Прежние куски этой встречи удаляются: встречу могли переиндексировать после
    смены модели, и смесь векторов от разных моделей давала бы бессмысленные
    близости — они лежат в разных пространствах.
    """
    for старый in db.scalars(select(Chunk).where(Chunk.meeting_id == meeting.id)):
        db.delete(старый)

    segments = list(db.scalars(
        select(Segment).where(Segment.meeting_id == meeting.id).order_by(Segment.start_s)
    ))
    куски = build_chunks(segments, cfg.search_chunk_chars)
    if not куски:
        return 0

    векторы = await OllamaClient(cfg).embed(cfg.search_embed_model, [к["text"] for к in куски])
    for кусок, вектор in zip(куски, векторы):
        db.add(Chunk(meeting_id=meeting.id, model=cfg.search_embed_model,
                     vector=_to_blob(вектор), **кусок))
    db.flush()
    return len(куски)


async def reindex_missing(db: Session, cfg: Settings, owner_id: int | None = None) -> int:
    """Индексирует завершённые встречи, у которых кусков нет или они от другой модели.

    Ленивая индексация вместо крючка в конвейере: встреча могла завершиться до
    того, как поиск появился, а модель — смениться в настройках. Проверка
    дешёвая (один запрос), пересчёт идёт только там, где его не хватает.

    Считаем только свои встречи: чужие всё равно не попадут в выдачу, а платить
    за них временем эмбеддера (и держать очередь к модели) незачем.
    """
    # Два запроса вместо запроса на каждую встречу: на архиве в тысячу встреч
    # цикл давал тысячу лишних обращений к базе перед КАЖДЫМ поиском.
    готовые = set(db.scalars(
        select(Chunk.meeting_id).where(Chunk.model == cfg.search_embed_model).distinct()
    ))
    запрос = select(Meeting).where(Meeting.status == "done")
    if owner_id is not None:
        запрос = запрос.where(Meeting.owner_id == owner_id)
    нужны = [m for m in db.scalars(запрос) if m.id not in готовые]
    if not нужны:
        return 0

    посчитано = 0
    async with _замок:
        for meeting in нужны:
            посчитано += await index_meeting(db, cfg, meeting)
    return посчитано


async def search(db: Session, cfg: Settings, query: str, limit: int | None = None,
                 owner_id: int | None = None) -> list[dict]:
    """Куски, ближайшие к вопросу. Пустой запрос — пустая выдача, без похода к модели.

    Владельца берём у встречи, а не у куска: отдельная колонка у Chunk была бы
    вторым источником правды о том же самом, и рассинхрон вылезал бы как чужая
    цитата в выдаче.
    """
    query = (query or "").strip()
    if not query:
        return []

    отбор = select(Chunk).where(Chunk.model == cfg.search_embed_model)
    if owner_id is not None:
        отбор = отбор.join(Meeting, Chunk.meeting_id == Meeting.id).where(
            Meeting.owner_id == owner_id
        )
    куски = list(db.scalars(отбор))
    if not куски:
        return []

    (вектор,) = await OllamaClient(cfg).embed(cfg.search_embed_model, [query])
    q = np.frombuffer(_to_blob(вектор), dtype=np.float32)

    # Отбираем куски нужной размерности, а не режем склеенный буфер: под одним
    # именем модели могут оказаться векторы разной длины (модель перекачали
    # новой версией, пересчёт прошёл наполовину). Склейка тогда не делится
    # нацело, и пользователь получил бы 500 вместо внятной пустой выдачи.
    подходят = [к for к in куски if len(к.vector) == q.nbytes]
    if len(подходят) != len(куски):
        log.warning("Кусков с чужой размерностью: %d из %d — нужен пересчёт",
                    len(куски) - len(подходят), len(куски))
    if not подходят:
        return []
    куски = подходят

    матрица = np.frombuffer(b"".join(к.vector for к in куски), dtype=np.float32)
    матрица = матрица.reshape(len(куски), -1)
    близости = матрица @ q
    лучшие = np.argsort(-близости)[: (limit or cfg.search_top_k)]
    return [_result(куски[i], float(близости[i])) for i in лучшие]


async def answer(db: Session, cfg: Settings, llm: LlmRouter, query: str,
                 limit: int | None = None, owner_id: int | None = None) -> dict:
    """Ответ модели по найденным фрагментам плюс сами фрагменты.

    Цитаты возвращаются вместе с ответом и всегда показываются рядом с ним:
    модель не помнит встреч, она пересказывает показанное, и проверить её можно
    только по тому, что видно на экране. Ответ без цитат про договорённости —
    это уверенное враньё, которое невозможно поймать.

    Какой моделью отвечать — настройка search_answer_model. По умолчанию
    моделью протокола: ответ по нескольким фрагментам ближе к резюме, чем к
    реплике на лету. Но на слабой машине разница в скорости решает (замер на
    M3: qwen3:4b — 36 с, qwen3:1.7b — 13 с), поэтому можно переключить на
    модель подсказок. Фрагментов немного и по другой причине — промпт должен
    помещаться в контекст локальной модели.
    """
    найденное = await search(db, cfg, query, limit, owner_id)
    if not найденное:
        return {"answer": "", "results": []}

    system, prompt = prompts.build_search_answer_prompt(query, найденное)
    если_подсказки = cfg.search_answer_model == "hints"
    текст = await (llm.hint(prompt, system=system) if если_подсказки
                   else llm.summarize(prompt, system=system))
    return {"answer": текст.strip(), "results": найденное}


def _result(chunk: Chunk, similarity: float) -> dict:
    return {
        "meeting_id": chunk.meeting_id,
        "meeting_title": chunk.meeting.title,
        "started_at": chunk.meeting.started_at.isoformat() if chunk.meeting.started_at else None,
        "start_s": chunk.start_s,
        "text": chunk.text,
        "similarity": round(similarity, 3),
    }
