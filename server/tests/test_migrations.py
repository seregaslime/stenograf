"""Интеграционные тесты миграций схемы БД (db/database.py) на реальном файле
SQLite: старая схема → миграция → апгрейд без потери данных. Быстрые (без
моделей), поэтому идут в дефолтном прогоне."""
import pytest
from sqlalchemy import create_engine

from app.db import database, models


@pytest.fixture()
def temp_engine(tmp_path, monkeypatch):
    eng = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    monkeypatch.setattr(database, "engine", eng)  # миграции берут engine из модуля
    return eng


def _table_sql(engine, name):
    with engine.begin() as conn:
        return conn.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).scalar()


def _columns(engine, name):
    with engine.begin() as conn:
        return [row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({name})")]


def _tables(engine):
    with engine.begin() as conn:
        return [r[0] for r in conn.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type='table'")]


def test_migrate_autoincrement_rebuilds_and_preserves(temp_engine):
    """Старая таблица без AUTOINCREMENT перестраивается без потери данных (иначе id удалённой встречи мог достаться новой).
    """
    with temp_engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE meetings (id INTEGER PRIMARY KEY, title VARCHAR, status VARCHAR, "
            "started_at DATETIME, ended_at DATETIME, record_audio BOOLEAN, audio_dir VARCHAR, "
            "summary TEXT, summary_model VARCHAR, summary_error TEXT)"
        )
        conn.exec_driver_sql(
            "INSERT INTO meetings (id, title, status, started_at, record_audio) "
            "VALUES (5, 'Старая', 'done', '2026-01-01 00:00:00', 0)"
        )
    assert "AUTOINCREMENT" not in (_table_sql(temp_engine, "meetings") or "").upper()

    # Порядок как в init_db: сначала добавляются недостающие колонки, потом
    # перестройка (она берёт список колонок из актуальной модели)
    database._migrate_meeting_mode()
    database._migrate_autoincrement(models.Base.metadata)

    assert "AUTOINCREMENT" in _table_sql(temp_engine, "meetings").upper()
    with temp_engine.begin() as conn:
        assert conn.exec_driver_sql("SELECT id, title FROM meetings").fetchone() == (5, "Старая")


def test_migrate_meeting_mode_adds_column_and_backfills(temp_engine):
    """Колонка типа встречи добавляется к старой базе, существующие встречи помечаются как планёрки; повторный запуск безопасен.
    """
    with temp_engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE meetings (id INTEGER PRIMARY KEY, title VARCHAR, status VARCHAR, "
            "started_at DATETIME, ended_at DATETIME, record_audio BOOLEAN, audio_dir VARCHAR, "
            "summary TEXT, summary_model VARCHAR, summary_error TEXT)"
        )
        conn.exec_driver_sql(
            "INSERT INTO meetings (id, title, status, started_at, record_audio) "
            "VALUES (1, 'Старая', 'done', '2026-01-01 00:00:00', 0)"
        )
    assert "meeting_mode" not in _columns(temp_engine, "meetings")

    database._migrate_meeting_mode()

    assert "meeting_mode" in _columns(temp_engine, "meetings")
    with temp_engine.begin() as conn:
        assert conn.exec_driver_sql(
            "SELECT title, meeting_mode FROM meetings"
        ).fetchone() == ("Старая", "work")  # существующие встречи — планёрки

    database._migrate_meeting_mode()  # повторный вызов безопасен
    assert _columns(temp_engine, "meetings").count("meeting_mode") == 1


def test_meeting_mode_migration_runs_before_autoincrement(temp_engine):
    """Регрессия: _migrate_autoincrement строит SELECT по колонкам актуальной
    модели, поэтому на старой базе без meeting_mode он падает с «no such column»
    — прямо при старте сервера. Колонку обязана добавить предыдущая миграция."""
    with temp_engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE meetings (id INTEGER PRIMARY KEY, title VARCHAR, status VARCHAR, "
            "started_at DATETIME, ended_at DATETIME, record_audio BOOLEAN, audio_dir VARCHAR, "
            "summary TEXT, summary_model VARCHAR, summary_error TEXT)"
        )
        conn.exec_driver_sql(
            "INSERT INTO meetings (id, title, status, started_at, record_audio) "
            "VALUES (7, 'До обновления', 'done', '2026-01-01 00:00:00', 0)"
        )

    # обратный (неверный) порядок обязан падать — иначе регрессия незаметна
    with pytest.raises(Exception, match="meeting_mode"):
        database._migrate_autoincrement(models.Base.metadata)

    database._migrate_meeting_mode()
    database._migrate_autoincrement(models.Base.metadata)  # теперь проходит
    with temp_engine.begin() as conn:
        assert conn.exec_driver_sql(
            "SELECT id, meeting_mode FROM meetings"
        ).fetchone() == (7, "work")


def test_migrate_voiceprints_moves_centroid(temp_engine):
    """Центроид голоса переезжает из строки спикера в отдельную таблицу отпечатков."""
    models.Base.metadata.tables["voiceprints"].create(temp_engine)
    with temp_engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE speakers (id INTEGER PRIMARY KEY, name VARCHAR, is_self BOOLEAN, "
            "created_at DATETIME, centroid BLOB, embedding_count INTEGER)"
        )
        conn.exec_driver_sql(
            "INSERT INTO speakers (id, name, is_self, created_at, centroid, embedding_count) "
            "VALUES (1, 'A', 0, '2026-01-01 00:00:00', X'0102', 3)"
        )
    database._migrate_voiceprints(models.Base.metadata)

    assert "centroid" not in _columns(temp_engine, "speakers")  # колонка убрана из speakers
    with temp_engine.begin() as conn:
        vp = conn.exec_driver_sql(
            "SELECT speaker_id, centroid, embedding_count FROM voiceprints").fetchone()
    assert vp == (1, b"\x01\x02", 3)  # центроид переехал в voiceprints


def test_migrate_samples_adds_columns_and_merges(tmp_path, temp_engine):
    """Аудио-образцы сливаются с отпечатками в одно «звучание», лишние файлы удаляются."""
    sample_file = tmp_path / "sample.wav"
    sample_file.write_bytes(b"RIFF....")  # файл существует → образец прикрепится к отпечатку
    with temp_engine.begin() as conn:
        conn.exec_driver_sql(  # старая схема voiceprints — без audio_*
            "CREATE TABLE voiceprints (id INTEGER PRIMARY KEY, speaker_id INTEGER, "
            "centroid BLOB, embedding_count INTEGER, created_at DATETIME)"
        )
        conn.exec_driver_sql("INSERT INTO voiceprints (id, speaker_id, centroid) VALUES (1, 7, X'00')")
        conn.exec_driver_sql(
            "CREATE TABLE speaker_samples (id INTEGER PRIMARY KEY, speaker_id INTEGER, "
            "path VARCHAR, duration_s FLOAT)"
        )
        conn.exec_driver_sql(
            "INSERT INTO speaker_samples (speaker_id, path, duration_s) VALUES (7, ?, 2.5)",
            (str(sample_file),),
        )
    database._migrate_samples_into_prints()

    assert "audio_path" in _columns(temp_engine, "voiceprints")  # колонки добавлены
    assert "speaker_samples" not in _tables(temp_engine)         # старая таблица удалена
    with temp_engine.begin() as conn:
        vp = conn.exec_driver_sql(
            "SELECT audio_path, audio_duration_s FROM voiceprints WHERE id=1").fetchone()
    assert vp == (str(sample_file), 2.5)  # образец прикреплён к отпечатку
