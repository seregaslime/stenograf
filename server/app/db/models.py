from datetime import datetime, timezone
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, CheckConstraint, DateTime, Float, ForeignKey, LargeBinary, String, Text
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

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Speaker(Base):
    """Человек. Живёт между встречами; у него 1..N отпечатков голоса."""

    __tablename__ = "speakers"
    # id удалённых профилей не переиспользуются: иначе фоновые задачи и файлы
    # образцов «наследуются» новым профилем с тем же id. В SQLite это надо было
    # просить отдельно (sqlite_autoincrement), PostgreSQL так делает сам —
    # последовательность назад не отматывается. Проверено тестом, потому что
    # держится это на поведении СУБД, а не на нашем коде.

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
    # См. Speaker: новая встреча не должна получить id удалённой, пока по
    # удалённой ещё может дописывать резюме фоновая задача.

    id: Mapped[int] = mapped_column(primary_key=True)
    # Чья встреча. NULL — сервер личный, людей на нём не заводили; как только
    # заводят первого, ничейные встречи достаются ему (см. auth.create_user).
    # SET NULL, а не CASCADE: отзыв доступа — это отзыв ключа, а не удаление
    # архива. На PostgreSQL правило наконец работает: встречи удалённого
    # становятся ничейными. Пока база была SQLite, ключи не проверялись, и они
    # оставались с owner_id на несуществующего — невидимые всем навсегда.
    #
    # Следствие, о котором надо знать: ничейные встречи достаются первому
    # заведённому человеку (см. auth.create_user). То есть «удалить всех и
    # завести одного» теперь отдаёт ему чужой архив, а не прячет его навсегда.
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


class Document(Base):
    """Свой файл человека в базе знаний: регламент, ТЗ, заметки (пункт 5а).

    Хранится извлечённый текст, а не сам файл: искать нужно по тексту, а
    оригинал у человека и так есть. Ищется тем же поиском, что и встречи.
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Как у встреч: SET NULL, а не CASCADE — отзыв доступа не удаляет данные;
    # ничейные документы достаются первому заведённому (auth.create_user).
    owner_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(String(300))
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Куски поиска умирают вместе с документом — как у встречи
    chunks: Mapped[list["Chunk"]] = relationship(cascade="all, delete-orphan")


class Chunk(Base):
    """Кусок для поиска и его вектор: из встречи или из документа.

    Ищем не по репликам: они короткие (на живых встречах в среднем 49 символов),
    и вектор от «Да-да, согласен» ничего не значит. Кусок набирается до
    search_chunk_chars символов. У куска встречи хранятся границы реплик и время;
    у куска документа их нет — есть только текст.

    Источник у куска ровно один, и это держит база, а не код: одна таблица
    кусков — это один индекс HNSW и один поиск на встречи и документы сразу.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        CheckConstraint("(meeting_id IS NULL) <> (document_id IS NULL)", name="chunks_one_source"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), nullable=True
    )
    document_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=True, index=True
    )
    first_segment_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("segments.id", ondelete="CASCADE"), nullable=True
    )
    last_segment_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("segments.id", ondelete="CASCADE"), nullable=True
    )
    # Секунда встречи, с которой начинается кусок; у документа времени нет
    start_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    text: Mapped[str] = mapped_column(Text)
    # Какой моделью посчитан вектор: сменили модель — старые куски надо
    # пересчитать, иначе в одном индексе окажутся несравнимые векторы.
    model: Mapped[str] = mapped_column(String(80))
    # L2-нормированный: близость — скалярное произведение, его и считает база.
    # Размерность не зафиксирована намеренно: у каждого человека своя модель
    # эмбеддингов (bge-m3 — 1024, другие — 768), и векторы разной длины лежат в
    # одной колонке. Цена — индекс HNSW требует фиксированной размерности; пока
    # кусков сотни, база перебирает их точно и быстро, индекс — после замера.
    vector: Mapped[list[float]] = mapped_column(Vector())

    meeting: Mapped[Optional[Meeting]] = relationship()


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
    # Время слов от начала встречи: [[начало, конец, слово], ...], склейка слов
    # равна text. По нему человек отдаёт часть реплики другому спикеру, и разрез
    # проходит между словами. NULL — старые реплики и whisper-движки: делить нельзя.
    words: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # Отладочная метрика: косинусная близость к центроиду спикера (для "режима отладки" в UI)
    similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    meeting: Mapped[Meeting] = relationship(back_populates="segments")
    speaker: Mapped[Optional[Speaker]] = relationship()
