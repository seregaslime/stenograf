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
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import text

from app.db import models
from app.db.database import alembic_config, engine


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
