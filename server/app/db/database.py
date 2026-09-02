from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..config import settings

settings.data_dir.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"sqlite:///{settings.db_path}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    from . import models  # noqa: F401 — регистрирует таблицы

    models.Base.metadata.create_all(engine)
    _migrate_voiceprints(models.Base.metadata)
    _migrate_meeting_mode()
    _migrate_meeting_owner()
    _migrate_autoincrement(models.Base.metadata)
    _migrate_samples_into_prints()


def _migrate_voiceprints(metadata) -> None:
    """Переносит центроиды из speakers в таблицу voiceprints (базы до v0.3).

    Раньше у спикера был один центроид прямо в строке; теперь отпечатков может
    быть несколько. Работает до _migrate_autoincrement: та перестройка берёт
    список колонок из актуальной модели и потеряла бы старую колонку centroid.
    """
    with engine.begin() as conn:
        columns = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info(speakers)")]
        if "centroid" not in columns:
            return
        conn.exec_driver_sql(
            "INSERT INTO voiceprints (speaker_id, centroid, embedding_count, created_at) "
            "SELECT id, centroid, embedding_count, created_at FROM speakers "
            "WHERE centroid IS NOT NULL"
        )
        keep = "id, name, is_self, created_at"
        conn.exec_driver_sql(f"CREATE TEMPORARY TABLE _migration AS SELECT {keep} FROM speakers")
        conn.exec_driver_sql("DROP TABLE speakers")
        metadata.tables["speakers"].create(conn)
        conn.exec_driver_sql(
            f"INSERT INTO speakers ({keep}) SELECT {keep} FROM _migration"
        )
        conn.exec_driver_sql("DROP TABLE _migration")


def _migrate_meeting_mode() -> None:
    """Добавляет тип встречи (базы до v0.5): планёрка / собеседование / переговоры.

    Обязана работать ДО _migrate_autoincrement: та перестройка берёт список
    колонок из актуальной модели и на старой таблице упала бы с
    «no such column: meeting_mode» — прямо при старте сервера.
    """
    with engine.begin() as conn:
        columns = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info(meetings)")]
        if "meeting_mode" in columns:
            return
        conn.exec_driver_sql("ALTER TABLE meetings ADD COLUMN meeting_mode VARCHAR(20)")
        conn.exec_driver_sql("UPDATE meetings SET meeting_mode = 'work'")


def _migrate_meeting_owner() -> None:
    """Добавляет владельца встречи (базы до v0.6).

    Как и _migrate_meeting_mode, обязана отработать ДО _migrate_autoincrement:
    та перестройка берёт список колонок из актуальной модели и на старой таблице
    упала бы с «no such column: owner_id» прямо при старте сервера.

    Существующие встречи остаются ничейными намеренно: на личном сервере людей
    нет вовсе, и приписывать записи некому. Первому заведённому человеку они
    достанутся целиком — иначе он обновил бы сервер и обнаружил пустой архив.
    """
    with engine.begin() as conn:
        columns = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info(meetings)")]
        if "owner_id" in columns:
            return
        conn.exec_driver_sql("ALTER TABLE meetings ADD COLUMN owner_id INTEGER")


def _migrate_autoincrement(metadata) -> None:
    """Перестраивает старые таблицы без AUTOINCREMENT (базы, созданные до v0.2).

    Без AUTOINCREMENT SQLite переиспользует id удалённых строк — новая встреча
    могла получить id только что удалённой и «поймать» её фоновое резюме.
    Данные сохраняются: таблица копируется во временную и обратно.
    """
    with engine.begin() as conn:
        for table_name in ("meetings", "speakers"):
            sql = conn.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).scalar()
            if sql is None or "AUTOINCREMENT" in sql.upper():
                continue
            columns = ", ".join(
                column.name for column in metadata.tables[table_name].columns
            )
            conn.exec_driver_sql(
                f"CREATE TEMPORARY TABLE _migration AS SELECT {columns} FROM {table_name}"
            )
            conn.exec_driver_sql(f"DROP TABLE {table_name}")
            metadata.tables[table_name].create(conn)
            conn.exec_driver_sql(
                f"INSERT INTO {table_name} ({columns}) SELECT {columns} FROM _migration"
            )
            conn.exec_driver_sql("DROP TABLE _migration")


def _migrate_samples_into_prints() -> None:
    """Сливает аудио-образцы с отпечатками (базы до v0.4).

    Раньше образец голоса и отпечаток были отдельными сущностями: образец
    можно слушать, отпечаток решает узнавание. Теперь это одно «звучание»:
    у отпечатка появляется аудио. Существующие образцы прикрепляются к
    отпечаткам того же спикера (старейший образец — старейшему отпечатку
    без аудио), лишние файлы удаляются, таблица speaker_samples исчезает.
    """
    with engine.begin() as conn:
        columns = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info(voiceprints)")]
        if "audio_path" not in columns:
            conn.exec_driver_sql("ALTER TABLE voiceprints ADD COLUMN audio_path VARCHAR(500)")
            conn.exec_driver_sql("ALTER TABLE voiceprints ADD COLUMN audio_duration_s FLOAT")
        has_samples = conn.exec_driver_sql(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='speaker_samples'"
        ).fetchone()
        if has_samples is None:
            return
        samples = conn.exec_driver_sql(
            "SELECT speaker_id, path, duration_s FROM speaker_samples ORDER BY speaker_id, id"
        ).fetchall()
        for speaker_id, path, duration in samples:
            target = conn.exec_driver_sql(
                "SELECT id FROM voiceprints WHERE speaker_id = ? AND audio_path IS NULL "
                "ORDER BY id LIMIT 1",
                (speaker_id,),
            ).fetchone()
            if target is not None and Path(path).exists():
                conn.exec_driver_sql(
                    "UPDATE voiceprints SET audio_path = ?, audio_duration_s = ? WHERE id = ?",
                    (path, duration, target[0]),
                )
            else:  # прикрепить некуда — файл больше не нужен
                Path(path).unlink(missing_ok=True)
        conn.exec_driver_sql("DROP TABLE speaker_samples")


@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
