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
from .llm.ollama_client import OllamaClient

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


async def reindex_missing(db: Session, cfg: Settings) -> int:
    """Индексирует завершённые встречи, у которых кусков нет или они от другой модели.

    Ленивая индексация вместо крючка в конвейере: встреча могла завершиться до
    того, как поиск появился, а модель — смениться в настройках. Проверка
    дешёвая (один запрос), пересчёт идёт только там, где его не хватает.
    """
    # Два запроса вместо запроса на каждую встречу: на архиве в тысячу встреч
    # цикл давал тысячу лишних обращений к базе перед КАЖДЫМ поиском.
    готовые = set(db.scalars(
        select(Chunk.meeting_id).where(Chunk.model == cfg.search_embed_model).distinct()
    ))
    нужны = [m for m in db.scalars(select(Meeting).where(Meeting.status == "done"))
             if m.id not in готовые]
    if not нужны:
        return 0

    посчитано = 0
    async with _замок:
        for meeting in нужны:
            посчитано += await index_meeting(db, cfg, meeting)
    return посчитано


async def search(db: Session, cfg: Settings, query: str, limit: int | None = None) -> list[dict]:
    """Куски, ближайшие к вопросу. Пустой запрос — пустая выдача, без похода к модели."""
    query = (query or "").strip()
    if not query:
        return []

    куски = list(db.scalars(select(Chunk).where(Chunk.model == cfg.search_embed_model)))
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


def _result(chunk: Chunk, similarity: float) -> dict:
    return {
        "meeting_id": chunk.meeting_id,
        "meeting_title": chunk.meeting.title,
        "started_at": chunk.meeting.started_at.isoformat() if chunk.meeting.started_at else None,
        "start_s": chunk.start_s,
        "text": chunk.text,
        "similarity": round(similarity, 3),
    }
