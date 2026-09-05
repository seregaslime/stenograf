"""Поиск по прошлым встречам: куски разговора, их векторы и подбор ближайших.

Зачем: договорённости растворяются. Через месяц никто не помнит, на какой
встрече решили перенести сроки, а протоколы приходится открывать по одному.

Как: каждый кусок разговора превращается в вектор (эмбеддинг), вопрос — тоже,
и мы берём куски с наибольшей близостью. Это ровно то, что SpeakerRegistry
делает с голосами (`np.dot` по L2-нормированным векторам ECAPA), только
сравниваются не тембры, а смыслы.

Векторы считает приложение: у каждого своя модель эмбеддингов и свой адрес.
Здесь осталось то, для чего модель не нужна, — нарезка разговора на куски и
сравнение готовых векторов.

Векторной СУБД (FAISS, sqlite-vec) здесь нет намеренно: на живых данных это
матрица в единицы мегабайт, полный перебор занимает миллисекунды, а лишнее
хранилище — это ещё одна зависимость и ещё один способ сломаться.
"""
import logging

import numpy as np
from sqlalchemy import select
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


def _result(chunk: Chunk, similarity: float) -> dict:
    return {
        "meeting_id": chunk.meeting_id,
        "meeting_title": chunk.meeting.title,
        "started_at": chunk.meeting.started_at.isoformat() if chunk.meeting.started_at else None,
        "start_s": chunk.start_s,
        "text": chunk.text,
        "similarity": round(similarity, 3),
    }
