from contextlib import contextmanager

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
    _migrate_autoincrement(models.Base.metadata)


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
