"""Операции с БД. Все функции принимают открытую сессию — управление
транзакцией на вызывающей стороне (session_scope)."""
from datetime import datetime, timezone
from typing import Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from .models import Meeting, Segment, Speaker, SpeakerSample, VoicePrint


# --- Спикеры ---

def get_or_create_self_speaker(db: Session) -> Speaker:
    speaker = db.scalar(select(Speaker).where(Speaker.is_self.is_(True)))
    if speaker is None:
        speaker = Speaker(name="Вы", is_self=True)
        db.add(speaker)
        db.flush()
    return speaker


def create_speaker(db: Session) -> Speaker:
    speaker = Speaker(name="")
    db.add(speaker)
    db.flush()  # получаем id для имени по умолчанию
    speaker.name = f"Спикер {speaker.id}"
    return speaker


def list_speakers(db: Session) -> list[dict]:
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
    rows = db.execute(
        select(Speaker, meetings_sq, segments_sq, prints_sq)
        .order_by(Speaker.is_self.desc(), Speaker.id)
    ).all()
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
                "samples": [
                    {"id": s.id, "duration_s": round(s.duration_s, 1)} for s in speaker.samples
                ],
                "voiceprints": [
                    {"id": p.id, "count": p.embedding_count}
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

def create_meeting(db: Session, title: str, record_audio: bool) -> Meeting:
    meeting = Meeting(title=title.strip() or "Встреча", record_audio=record_audio)
    db.add(meeting)
    db.flush()
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
) -> Segment:
    segment = Segment(
        meeting_id=meeting_id,
        speaker_id=speaker_id,
        channel=channel,
        start_s=start_s,
        end_s=end_s,
        text=text,
        similarity=similarity,
    )
    db.add(segment)
    db.flush()
    return segment


def list_meetings(db: Session) -> list[dict]:
    rows = db.execute(
        select(Meeting, func.count(Segment.id))
        .outerjoin(Segment)
        .group_by(Meeting.id)
        .order_by(Meeting.started_at.desc())
    ).all()
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


def segment_to_dict(segment: Segment) -> dict:
    speaker = segment.speaker
    return {
        "id": segment.id,
        "meeting_id": segment.meeting_id,
        "channel": segment.channel,
        "start_s": round(segment.start_s, 2),
        "end_s": round(segment.end_s, 2),
        "text": segment.text,
        "similarity": round(segment.similarity, 3) if segment.similarity is not None else None,
        "speaker": {
            "id": speaker.id,
            "name": speaker.name,
            "is_self": speaker.is_self,
        }
        if speaker
        else None,
    }
