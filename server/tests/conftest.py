"""Общие фикстуры тестов.

Тесты не трогают рабочую базу server/data: БД — SQLite в памяти, файлы — во
временной папке pytest. Тяжёлые модели (ECAPA, whisper) в юнит-тестах не
загружаются — реальный звук проверяется интеграционными тестами (-m integration)
и скриптом scripts/eval_voices.py.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db import models


@pytest.fixture()
def cfg(tmp_path) -> Settings:
    # _env_file=None — локальный server/.env не должен влиять на тесты
    return Settings(data_dir=tmp_path, _env_file=None)


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite://")
    models.Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    yield session
    session.close()
