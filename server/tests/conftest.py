"""Общие фикстуры тестов.

Тесты не трогают рабочую базу: адрес подменяется на stenograf_test ДО импорта
app.config, файлы уходят во временную папку pytest. Тяжёлые модели (ECAPA,
whisper) в юнит-тестах не загружаются — реальный звук проверяется
интеграционными тестами (-m integration) и скриптом scripts/eval_voices.py.

База настоящая, PostgreSQL: SQLite из проекта убран, и подменять его в тестах
на «почти такую же» базу значило бы проверять не то, что работает у людей.
Цена — тесты требуют запущенного PostgreSQL; как его поднять, написано в README.
"""
import os
import tempfile

# ДО импорта app.config: уводим базу и данные в сторону. Функциональные тесты
# через TestClient используют глобальные синглтоны app.main, и рабочая база не
# должна пострадать.
os.environ.setdefault(
    "STENOGRAF_DATABASE_URL",
    "postgresql+psycopg://stenograf:stenograf@127.0.0.1:5432/stenograf_test",
)
os.environ.setdefault("STENOGRAF_DATA_DIR", tempfile.mkdtemp(prefix="stenograf_test_"))
os.environ.setdefault("STENOGRAF_PRELOAD_ASR", "false")

# Раньше этой проверки не требовалось: база жила в памяти и погибала вместе с
# прогоном. Теперь фикстура схемы делает drop_all в настоящей базе, и адрес,
# оставшийся в окружении от рабочего запуска (или подхваченный внутри
# контейнера, где STENOGRAF_DATABASE_URL выставлен в docker-compose), стёр бы
# чужие встречи молча и без единого вопроса.
_адрес = os.environ["STENOGRAF_DATABASE_URL"]
if "test" not in _адрес.rsplit("/", 1)[-1]:
    raise SystemExit(
        f"Тесты сносят схему целиком, а STENOGRAF_DATABASE_URL указывает на «{_адрес}» — "
        "имя базы без «test». Уберите переменную из окружения или укажите тестовую базу."
    )

import pytest
from sqlalchemy import text

from app.config import Settings
from app.db import models
from app.db.database import SessionLocal, engine, init_db


@pytest.fixture()
def cfg(tmp_path) -> Settings:
    # _env_file=None — локальный server/.env не должен влиять на тесты
    return Settings(data_dir=tmp_path, _env_file=None)


@pytest.fixture(scope="session", autouse=True)
def схема():
    """Схема создаётся один раз на прогон, а не на каждый тест: на настоящей
    базе это сеть и десятки DDL-запросов, а не миллисекунды в памяти.

    Через init_db, то есть ревизиями Alembic, — тем же путём, каким схема
    появляется у людей. Через create_all было бы быстрее, но тогда тесты
    проверяли бы схему, которой ни у кого нет: разойдись ревизии с models.py, и
    прогон остался бы зелёным.
    """
    очистить_схему()
    init_db()
    yield
    очистить_схему()


def очистить_схему() -> None:
    models.Base.metadata.drop_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))


def очистить() -> None:
    """Пустые таблицы. RESTART IDENTITY — чтобы id начинались с единицы: часть
    тестов сверяет их напрямую."""
    таблицы = ", ".join(f'"{t.name}"' for t in models.Base.metadata.sorted_tables)
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {таблицы} RESTART IDENTITY CASCADE"))


@pytest.fixture(autouse=True)
def чистая_база(схема, request):
    """Чистая база перед каждым тестом.

    В памяти база умирала вместе с тестом, а тут она общая на прогон: без
    очистки встречи и спикеры перетекали бы из теста в тест, и порядок запуска
    начал бы влиять на результат.

    Медленные уровни — исключение: e2e и нагрузочный прогон ведут одну длинную
    историю, где база нарочно копится от теста к тесту (спикеры узнаются на
    следующей встрече — это и проверяется). Чистить между ними значило бы
    стирать то, ради чего они написаны.
    """
    if request.node.get_closest_marker("e2e") or request.node.get_closest_marker("load"):
        yield
        return
    очистить()
    yield


@pytest.fixture()
def db_session():
    session = SessionLocal()
    yield session
    session.close()
