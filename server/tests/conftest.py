"""Общие фикстуры тестов.

Тесты не трогают рабочую базу server/data: БД — SQLite в памяти, файлы — во
временной папке pytest. Тяжёлые модели (ECAPA, whisper) в юнит-тестах не
загружаются — реальный звук проверяется интеграционными тестами (-m integration)
и скриптом scripts/eval_voices.py.
"""
import os
import tempfile

# ДО импорта app.config: уводим data_dir во временную папку (функциональные тесты
# через TestClient используют глобальные синглтоны app.main — рабочая БД не должна
# пострадать) и не грузим ASR-модель при старте приложения.
os.environ.setdefault("STENOGRAF_DATA_DIR", tempfile.mkdtemp(prefix="stenograf_test_"))
os.environ.setdefault("STENOGRAF_PRELOAD_ASR", "false")

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
