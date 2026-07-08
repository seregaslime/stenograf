from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Speaker(Base):
    """Глобальный профиль голоса — живёт между встречами."""

    __tablename__ = "speakers"
    # AUTOINCREMENT: id удалённых профилей не переиспользуются — иначе фоновые
    # задачи и файлы образцов «наследуются» новым профилем с тем же id
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    is_self: Mapped[bool] = mapped_column(default=False)  # владелец микрофона ("Вы")
    # Центроид ECAPA-эмбеддингов (float32 bytes) и сколько эмбеддингов в него вошло
    centroid: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    embedding_count: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    samples: Mapped[list["SpeakerSample"]] = relationship(
        back_populates="speaker", cascade="all, delete-orphan"
    )


class SpeakerSample(Base):
    """Короткий WAV-образец голоса для прослушивания на вкладке «Спикеры»."""

    __tablename__ = "speaker_samples"

    id: Mapped[int] = mapped_column(primary_key=True)
    speaker_id: Mapped[int] = mapped_column(ForeignKey("speakers.id", ondelete="CASCADE"))
    path: Mapped[str] = mapped_column(String(500))
    duration_s: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    speaker: Mapped[Speaker] = relationship(back_populates="samples")


class Meeting(Base):
    __tablename__ = "meetings"
    # AUTOINCREMENT: см. Speaker — новая встреча не должна получить id удалённой,
    # пока по удалённой ещё может дописывать резюме фоновая задача
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(300), default="Встреча")
    status: Mapped[str] = mapped_column(String(20), default="live")  # live | summarizing | done
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    record_audio: Mapped[bool] = mapped_column(default=False)
    audio_dir: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary_model: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    summary_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    segments: Mapped[list["Segment"]] = relationship(
        back_populates="meeting", cascade="all, delete-orphan"
    )


class Segment(Base):
    """Одна реплика транскрипта."""

    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[int] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"))
    speaker_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("speakers.id", ondelete="SET NULL"), nullable=True
    )
    channel: Mapped[str] = mapped_column(String(10))  # mic | system
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    # Отладочная метрика: косинусная близость к центроиду спикера (для "режима отладки" в UI)
    similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    meeting: Mapped[Meeting] = relationship(back_populates="segments")
    speaker: Mapped[Optional[Speaker]] = relationship()
