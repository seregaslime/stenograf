"""Операции с БД. Все функции принимают открытую сессию — управление
транзакцией на вызывающей стороне (session_scope)."""
from datetime import datetime, timezone
from typing import Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from .models import Meeting, Segment, Speaker, VoicePrint

# --- Спикеры ---

def get_or_create_self_speaker(db: Session, owner_id: Optional[int] = None) -> Speaker:
    """Профиль «Вы» — владелец микрофона. У каждого человека свой: это разные
    люди, а на «Вы» висит скидка к порогу узнавания, и общий профиль сливал бы
    их голоса в один."""
    запрос = select(Speaker).where(Speaker.is_self.is_(True))
    if owner_id is not None:
        запрос = запрос.where(Speaker.owner_id == owner_id)
    else:
        запрос = запрос.where(Speaker.owner_id.is_(None))
    speaker = db.scalar(запрос)
    if speaker is None:
        speaker = Speaker(name="Вы", is_self=True, owner_id=owner_id)
        db.add(speaker)
        db.flush()
    return speaker


def create_speaker(db: Session, owner_id: Optional[int] = None) -> Speaker:
    speaker = Speaker(name="", owner_id=owner_id)
    db.add(speaker)
    db.flush()  # получаем id для имени по умолчанию
    speaker.name = f"Спикер {speaker.id}"
    return speaker


def speaker_for_owner(db: Session, speaker_id: int, owner_id: Optional[int]) -> Optional[Speaker]:
    """Профиль, если он существует и принадлежит этому человеку.

    Как и со встречами: чужой профиль неотличим от несуществующего, иначе
    перебором id узнаётся, сколько людей в чужой библиотеке голосов.
    """
    speaker = db.get(Speaker, speaker_id)
    if speaker is None:
        return None
    if owner_id is not None and speaker.owner_id != owner_id:
        return None
    return speaker


def list_speakers(db: Session, owner_id: Optional[int] = None) -> list[dict]:
    # скалярные подзапросы вместо join'ов — два outerjoin размножили бы строки
    meetings_sq = (
        select(func.count(func.distinct(Segment.meeting_id)))
        .where(Segment.speaker_id == Speaker.id)
        .scalar_subquery()
    )
    segments_sq = (
        select(func.count(Segment.id))
        .where(Segment.speaker_id == Speaker.id)
        .scalar_subquery()
    )
    prints_sq = (
        select(func.count(VoicePrint.id))
        .where(VoicePrint.speaker_id == Speaker.id)
        .scalar_subquery()
    )
    запрос = select(Speaker, meetings_sq, segments_sq, prints_sq)
    if owner_id is not None:
        запрос = запрос.where(Speaker.owner_id == owner_id)
    rows = db.execute(запрос.order_by(Speaker.is_self.desc(), Speaker.id)).all()
    result = []
    for speaker, meetings, segments, prints in rows:
        result.append(
            {
                "id": speaker.id,
                "name": speaker.name,
                "is_self": speaker.is_self,
                "meetings_count": meetings,
                "segments_count": segments,
                "voiceprints_count": prints,
                "created_at": speaker.created_at.isoformat() if speaker.created_at else None,
                "voiceprints": [
                    {
                        "id": p.id,
                        "count": p.embedding_count,
                        "audio_duration_s": round(p.audio_duration_s, 1)
                        if p.audio_duration_s is not None else None,
                    }
                    for p in sorted(speaker.voiceprints, key=lambda p: p.id)
                ],
            }
        )
    return result


def rename_speaker(db: Session, speaker_id: int, name: str) -> Optional[Speaker]:
    speaker = db.get(Speaker, speaker_id)
    if speaker is not None:
        speaker.name = name.strip() or speaker.name
    return speaker


def speaker_names(db: Session, speaker_ids: list[int]) -> dict[int, str]:
    """Имена по id — одним запросом.

    Нужны там, где имя нельзя запомнить заранее: его меняют посреди встречи, и
    сохранённая строка устаревает (см. LiveSession._participants_line).
    """
    if not speaker_ids:
        return {}
    rows = db.query(Speaker.id, Speaker.name).filter(Speaker.id.in_(speaker_ids)).all()
    return {row.id: row.name for row in rows}


def reassign_segments(db: Session, from_speaker_id: int, to_speaker_id: Optional[int]) -> int:
    """Переписывает speaker_id в сегментах ВСЕХ встреч (merge либо, с None,
    отвязка реплик при удалении спикера — текст остаётся как «Неизвестный»)."""
    segments = db.scalars(
        select(Segment).where(Segment.speaker_id == from_speaker_id)
    ).all()
    for segment in segments:
        segment.speaker_id = to_speaker_id
    return len(segments)


# --- Встречи ---

def create_meeting(
    db: Session, title: str, record_audio: bool, meeting_mode: str = "work",
    owner_id: Optional[int] = None,
) -> Meeting:
    # meeting_mode последним и с дефолтом — есть позиционные вызовы в тестах
    meeting = Meeting(
        title=title.strip() or "Встреча",
        record_audio=record_audio,
        meeting_mode=meeting_mode,
        owner_id=owner_id,
    )
    db.add(meeting)
    db.flush()
    return meeting


def meeting_for_owner(db: Session, meeting_id: int, owner_id: Optional[int]) -> Optional[Meeting]:
    """Встреча, если она существует и доступна этому человеку.

    Чужая встреча неотличима от несуществующей намеренно: иначе по разнице
    между 403 и 404 перебором узнаётся, сколько встреч у соседа и когда они шли.
    """
    meeting = db.get(Meeting, meeting_id)
    if meeting is None:
        return None
    if owner_id is not None and meeting.owner_id != owner_id:
        return None
    return meeting


def end_meeting(db: Session, meeting_id: int, status: str = "summarizing") -> Optional[Meeting]:
    meeting = db.get(Meeting, meeting_id)
    if meeting is not None and meeting.ended_at is None:
        meeting.ended_at = datetime.now(timezone.utc)
        meeting.status = status
    return meeting


def add_segment(
    db: Session,
    meeting_id: int,
    speaker_id: Optional[int],
    channel: str,
    start_s: float,
    end_s: float,
    text: str,
    similarity: Optional[float] = None,
    words: Optional[list] = None,
) -> Segment:
    segment = Segment(
        meeting_id=meeting_id,
        speaker_id=speaker_id,
        channel=channel,
        start_s=start_s,
        end_s=end_s,
        text=text,
        similarity=similarity,
        words=words,
    )
    db.add(segment)
    db.flush()
    return segment


def list_meetings(db: Session, owner_id: Optional[int] = None) -> list[dict]:
    """Встречи владельца. owner_id=None — личный сервер, показываем все.

    Разделение по людям включается вместе с токенами: пока их нет, фильтровать
    не по чему, и пустой список вместо архива был бы поломкой, а не защитой.
    """
    запрос = (
        select(Meeting, func.count(Segment.id))
        .outerjoin(Segment)
        .group_by(Meeting.id)
        .order_by(Meeting.started_at.desc())
    )
    if owner_id is not None:
        запрос = запрос.where(Meeting.owner_id == owner_id)
    rows = db.execute(запрос).all()
    return [
        {
            "id": m.id,
            "title": m.title,
            "status": m.status,
            "started_at": m.started_at.isoformat() if m.started_at else None,
            "ended_at": m.ended_at.isoformat() if m.ended_at else None,
            "segments_count": count,
            "has_summary": bool(m.summary),
        }
        for m, count in rows
    ]


def meeting_segments(db: Session, meeting_id: int) -> Sequence[Segment]:
    return db.scalars(
        select(Segment)
        .where(Segment.meeting_id == meeting_id)
        .options(joinedload(Segment.speaker))
        .order_by(Segment.start_s)
    ).all()


def word_parts(
    words: list, start_s: float, end_s: float, first: int, last: int,
) -> list[tuple[float, float, list, bool]]:
    """Куски реплики, если слова first..last отдать другому: [(начало, конец, слова, выделено)].

    Кусков до трёх — до выделения, выделение, после; пустые не появляются.
    Граница проходит посередине паузы между словами: конец слова и начало
    следующего модель ставит с точностью до кадра, и середина честнее любого
    из краёв. Крайние куски сохраняют края реплики — запас VAD по краям речи
    не должен теряться.
    """
    куски = [(words[:first], False), (words[first:last + 1], True), (words[last + 1:], False)]
    куски = [(слова, выделено) for слова, выделено in куски if слова]
    границы = [start_s]
    for (левые, _), (правые, _) in zip(куски, куски[1:]):
        середина = round((левые[-1][1] + правые[0][0]) / 2, 2)
        границы.append(min(max(середина, границы[-1]), end_s))
    границы.append(end_s)
    return [(границы[i], границы[i + 1], слова, выделено)
            for i, (слова, выделено) in enumerate(куски)]


def reassign_words(
    db: Session, segment: Segment, first: int, last: int, speaker_id: int,
) -> list[Segment]:
    """Отдаёт слова first..last реплики спикеру speaker_id, деля реплику по словам.

    Исходная строка становится первым куском, а не удаляется: на неё по id
    ссылаются куски поиска, и удаление каскадом унесло бы их из индекса.
    Близость голоса остаётся только у кусков прежнего спикера — выделенный
    кусок назначен рукой, мерить там нечего.
    """
    куски = word_parts(segment.words, segment.start_s, segment.end_s, first, last)
    # Прежние спикер и близость — до цикла: первый кусок и есть исходная строка,
    # и если выделение в начале реплики, она меняет спикера раньше, чем
    # остальные куски успеют его скопировать.
    прежний_спикер, прежняя_близость = segment.speaker_id, segment.similarity
    строки = []
    for номер, (начало, конец, слова, выделено) in enumerate(куски):
        строка = segment if номер == 0 else Segment(
            meeting_id=segment.meeting_id, channel=segment.channel,
            similarity=прежняя_близость, speaker_id=прежний_спикер,
        )
        строка.start_s, строка.end_s, строка.words = начало, конец, слова
        строка.text = " ".join(слово for _, _, слово in слова)
        if выделено:
            строка.speaker_id, строка.similarity = speaker_id, None
        if номер:
            db.add(строка)
        строки.append(строка)
    db.flush()
    for строка in строки:
        db.refresh(строка)  # speaker подтянется уже новый
    return строки


def segments_by_ids(db: Session, meeting_id: int, ids: Sequence[int]) -> Sequence[Segment]:
    """Реплики по идентификаторам — ТОЛЬКО из указанной встречи.

    Фильтр по meeting_id обязателен и не является перестраховкой: id приходят от
    клиента, и без него можно было бы вытянуть текст чужой встречи, передав
    произвольные числа.
    """
    if not ids:
        return []
    return db.scalars(
        select(Segment)
        .where(Segment.meeting_id == meeting_id, Segment.id.in_(list(ids)))
        .options(joinedload(Segment.speaker))
        .order_by(Segment.start_s)  # порядок разговора, а не порядок кликов
    ).all()


def segment_to_dict(segment: Segment) -> dict:
    speaker = segment.speaker
    return {
        "id": segment.id,
        "meeting_id": segment.meeting_id,
        "channel": segment.channel,
        "start_s": round(segment.start_s, 2),
        "end_s": round(segment.end_s, 2),
        "text": segment.text,
        "words": segment.words,
        "similarity": round(segment.similarity, 3) if segment.similarity is not None else None,
        "speaker": {
            "id": speaker.id,
            "name": speaker.name,
            "is_self": speaker.is_self,
        }
        if speaker
        else None,
    }
