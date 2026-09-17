"""Ревизии Alembic против models.py.

Один тест, но без него Alembic — украшение. Схему теперь создают ревизии, а
код работает с models.py, и это два разных описания одной таблицы. Разъезжаются
они молча: разработчик добавил поле в модель, забыл ревизию — локально всё
работает (его база уже с полем), а на сервере после обновления первый же запрос
падает с «column does not exist». Ровно та беда, из-за которой в проекте и
появились когда-то самописные миграции.

Проверка идёт на настоящей базе: сравнивать SQL-описание можно только с той
СУБД, которая его исполняет.
"""
import numpy as np
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import text

from app.db import models
from app.db.database import alembic_config, alembic_include_object, engine


def test_ревизии_создают_ровно_то_что_в_моделях():
    """Схема после `alembic upgrade head` совпадает с models.py.

    Сравнение делает сам Alembic — тем же кодом, которым он генерирует ревизии:
    если он не видит, что менять, значит база и модели описывают одно и то же.
    """
    models.Base.metadata.drop_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))

    command.upgrade(alembic_config(), "head")

    with engine.connect() as conn:
        расхождения = compare_metadata(
            MigrationContext.configure(conn), models.Base.metadata
        )
    # alembic_version — служебная таблица самого Alembic, в моделях её нет и
    # быть не должно; всё остальное расхождение — настоящее.
    расхождения = [d for d in расхождения if "alembic_version" not in str(d)]
    assert расхождения == [], (
        "Схема из ревизий разошлась с models.py. Скорее всего, поле добавили в "
        "модель и не сняли ревизию: .venv/bin/alembic revision --autogenerate "
        f"-m 'что поменялось'. Расхождения: {расхождения}"
    )


def test_старые_векторы_переезжают_в_pgvector_а_битые_уходят_на_пересчёт():
    """Ревизия 5d1c2e7a9b40 переносит байты в колонку vector, а не пересчитывает.

    Пересчитать векторы сервер не может — модель у приложения. Поэтому перенос
    обязан сохранить числа как были. Кусок с битыми байтами перенести не во
    что: удаляются все куски его встречи, чтобы она снова встала в очередь на
    индексацию, а не считалась проиндексированной с дырой.
    """
    models.Base.metadata.drop_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
    command.upgrade(alembic_config(), "69052433fe21")

    целый = np.array([0.6, 0.8, 0.0], dtype=np.float32)
    with engine.begin() as conn:
        for встреча in (1, 2, 3):
            conn.execute(text(
                "INSERT INTO meetings (id, title, status, started_at, record_audio) "
                "VALUES (:m, 'встреча', 'done', now(), false)"), {"m": встреча})
            conn.execute(text(
                "INSERT INTO segments (id, meeting_id, channel, start_s, end_s, text, created_at) "
                "VALUES (:m, :m, 'mic', 0, 1, 'реплика', now())"), {"m": встреча})

        def кусок(встреча: int, байты: bytes) -> None:
            conn.execute(text(
                "INSERT INTO chunks (meeting_id, first_segment_id, last_segment_id, start_s, "
                "text, model, vector) VALUES (:m, :m, :m, 0, 'кусок', 'bge-m3', :v)"),
                {"m": встреча, "v": байты})

        кусок(1, целый.tobytes())
        кусок(2, целый.tobytes())       # целый, но у его встречи есть битый сосед
        кусок(2, b"\x00\x01\x02")       # не делится на float32
        кусок(3, np.array([np.nan, 1.0], dtype=np.float32).tobytes())  # база NaN не примет

    command.upgrade(alembic_config(), "head")

    with engine.connect() as conn:
        строки = conn.execute(text(
            "SELECT meeting_id, vector::text FROM chunks ORDER BY meeting_id")).all()
    assert [встреча for встреча, _ in строки] == [1]
    перенесённый = np.asarray(строки[0][1].strip("[]").split(","), dtype=np.float32)
    assert np.array_equal(перенесённый, целый)


def test_индексы_поиска_не_считаются_расхождением():
    """Индексы HNSW сервер создаёт на лету, в models.py их нет. Autogenerate на
    живой базе, где они уже есть, не должен предлагать их удалить."""
    from app import search
    from app.db.database import SessionLocal

    db = SessionLocal()
    try:
        search.ensure_index(db, 1024)
        расхождения = compare_metadata(
            MigrationContext.configure(db.connection(),
                                       opts={"include_object": alembic_include_object}),
            models.Base.metadata,
        )
        assert [d for d in расхождения if "alembic_version" not in str(d)] == []
    finally:
        db.rollback()
        db.close()


def test_документы_не_ломают_куски_встреч_и_откат():
    """Ревизия c505c6365acb делает поля встречи у куска необязательными. Куски
    встреч, лежавшие до неё, должны остаться как были, а откат — пройти, даже
    если куски документов уже появились: иначе он упал бы на meeting_id NULL."""
    models.Base.metadata.drop_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
    command.upgrade(alembic_config(), "b41e9d05c3f8")
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO meetings (id, title, status, started_at, record_audio) "
            "VALUES (1, 'встреча', 'done', now(), false)"))
        conn.execute(text(
            "INSERT INTO segments (id, meeting_id, channel, start_s, end_s, text, created_at) "
            "VALUES (1, 1, 'mic', 0, 1, 'реплика', now())"))
        conn.execute(text(
            "INSERT INTO chunks (meeting_id, first_segment_id, last_segment_id, start_s, text, model, vector) "
            "VALUES (1, 1, 1, 0, 'кусок встречи', 'bge-m3', '[1,0,0]')"))

    command.upgrade(alembic_config(), "c505c6365acb")
    with engine.begin() as conn:
        assert conn.execute(text("SELECT meeting_id, document_id FROM chunks")).all() == [(1, None)]
        conn.execute(text("INSERT INTO documents (id, title, text, created_at) "
                          "VALUES (1, 'Регламент', 'текст', now())"))
        conn.execute(text("INSERT INTO chunks (document_id, text, model, vector) "
                          "VALUES (1, 'кусок документа', 'bge-m3', '[0,1,0]')"))

    command.downgrade(alembic_config(), "b41e9d05c3f8")
    with engine.connect() as conn:
        assert conn.execute(text("SELECT text FROM chunks")).scalars().all() == ["кусок встречи"]
    command.upgrade(alembic_config(), "head")
