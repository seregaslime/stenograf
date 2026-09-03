from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    """Человек, работающий с сервером.

    Появился 02.09.2026. До этого сервер был однопользовательским: список встреч,
    библиотека голосов и профиль «Вы» — общие, и двое подключившихся видели одну
    кучу. Токен хранится хешем: восстанавливать его некому — при утере выдаётся
    новый, а украденная копия базы входа не даёт.
    """

    __tablename__ = "users"
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Speaker(Base):
    """Человек. Живёт между встречами; у него 1..N отпечатков голоса."""

    __tablename__ = "speakers"
    # AUTOINCREMENT: id удалённых профилей не переиспользуются — иначе фоновые
    # задачи и файлы образцов «наследуются» новым профилем с тем же id
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(primary_key=True)
    # Чей это голос в библиотеке. NULL — сервер личный (людей не заводили).
    # Библиотеки не пересекаются: один и тот же человек у двух пользователей —
    # два независимых профиля. Иначе «Вы» (владелец микрофона) был бы общим на
    # двоих, а на нём висит скидка к порогу — голоса разных людей слились бы в
    # один профиль. Отпечатки своей колонки не имеют: владелец у них через
    # спикера, второй источник правды тут разъехался бы.
    owner_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(120))
    is_self: Mapped[bool] = mapped_column(default=False)  # владелец микрофона ("Вы")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    voiceprints: Mapped[list["VoicePrint"]] = relationship(
        back_populates="speaker", cascade="all, delete-orphan"
    )


class VoicePrint(Base):
    """Одно «звучание» голоса: центроид ECAPA-эмбеддингов (float32 bytes)
    плюс аудио-фрагмент реплики, из которой отпечаток родился, — его можно
    прослушать на вкладке «Спикеры».

    У человека может быть несколько отпечатков: гарнитура, телефон и ноутбук
    звучат по-разному. При объединении профилей отпечатки не усредняются,
    а собираются под одним спикером — человек узнаётся в любом «звучании».
    """

    __tablename__ = "voiceprints"
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(primary_key=True)
    speaker_id: Mapped[int] = mapped_column(ForeignKey("speakers.id", ondelete="CASCADE"))
    centroid: Mapped[bytes] = mapped_column(LargeBinary)
    embedding_count: Mapped[int] = mapped_column(default=1)
    audio_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    audio_duration_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    speaker: Mapped[Speaker] = relationship(back_populates="voiceprints")


class Meeting(Base):
    __tablename__ = "meetings"
    # AUTOINCREMENT: см. Speaker — новая встреча не должна получить id удалённой,
    # пока по удалённой ещё может дописывать резюме фоновая задача
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(primary_key=True)
    # Чья встреча. NULL — сервер личный, людей на нём не заводили; как только
    # заводят первого, ничейные встречи достаются ему (см. auth.create_user).
    # SET NULL, а не CASCADE: отзыв доступа — это отзыв ключа, а не удаление
    # архива. Правило декоративное: внешние ключи в SQLite выключены, и на деле
    # встречи удалённого остаются с owner_id на несуществующего — то есть
    # невидимы всем и не достаются следующему заведённому. Так безопаснее, но
    # знать об этом надо: команда remove не стирает данные, а прячет их.
    owner_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(String(300), default="Встреча")
    status: Mapped[str] = mapped_column(String(20), default="live")  # live | summarizing | done
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    record_audio: Mapped[bool] = mapped_column(default=False)
    # Тип встречи (см. llm/prompts.py MODES): планёрка / собеседование / переговоры.
    # nullable — у баз до v0.5 колонки не было; нормализует prompts.normalize_mode
    meeting_mode: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, default="work")
    audio_dir: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary_model: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    summary_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    segments: Mapped[list["Segment"]] = relationship(
        back_populates="meeting", cascade="all, delete-orphan"
    )
    # Куски поиска умирают вместе со встречей: осиротевший вектор нашёлся бы в
    # поиске и указывал бы на удалённый разговор.
    chunks: Mapped[list["Chunk"]] = relationship(
        cascade="all, delete-orphan", overlaps="meeting"
    )


class Chunk(Base):
    """Кусок разговора для поиска: несколько подряд идущих реплик и их вектор.

    Ищем не по репликам: они короткие (на живых встречах в среднем 49 символов),
    и вектор от «Да-да, согласен» ничего не значит. Кусок набирается до
    search_chunk_chars символов, а в выдаче показываются реплики, попавшие в
    него, — их границы хранятся здесь же.

    Таблица создаётся сама (create_all добавляет ОТСУТСТВУЮЩИЕ таблицы), поэтому
    отдельной миграции не нужно — в отличие от новой колонки в существующей.
    """

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[int] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"))
    first_segment_id: Mapped[int] = mapped_column(ForeignKey("segments.id", ondelete="CASCADE"))
    last_segment_id: Mapped[int] = mapped_column(ForeignKey("segments.id", ondelete="CASCADE"))
    start_s: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    # Какой моделью посчитан вектор: сменили модель — старые куски надо
    # пересчитать, иначе в одном индексе окажутся несравнимые векторы.
    model: Mapped[str] = mapped_column(String(80))
    # float32, L2-нормированный. BLOB, а не JSON: 1024 числа текстом весят
    # вчетверо больше и разбираются на порядок дольше.
    vector: Mapped[bytes] = mapped_column(LargeBinary)

    meeting: Mapped[Meeting] = relationship()


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
