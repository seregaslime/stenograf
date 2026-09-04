"""Юнит-тесты конфигурации (config.py): дефолты и персист выбора ASR-движка."""
import json

import pytest

from app import config
from app.config import Settings, load_asr_choice, save_asr_choice


def test_defaults():
    """Дефолты конфига соответствуют заявленным в README (движок ASR, провайдер LLM, пути к данным).
    """
    s = Settings(_env_file=None)
    assert s.asr_engine == "gigaam"
    assert s.llm_provider == "local"
    assert s.hints_min_gap_s == 15.0
    assert s.db_path.name == "stenograf.db"
    assert s.samples_dir == s.data_dir / "samples"


def test_save_and_load_asr_choice(tmp_path, monkeypatch):
    """Выбор движка и модели ASR переживает перезапуск: пишется в asr.json и читается обратно.
    """
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    monkeypatch.setattr(config.settings, "asr_engine", "gigaam")
    monkeypatch.setattr(config.settings, "asr_model", "v3_e2e_rnnt")

    save_asr_choice("faster_whisper", "small")
    assert json.loads((tmp_path / "asr.json").read_text()) == {
        "engine": "faster_whisper", "model": "small"}
    assert config.settings.asr_engine == "faster_whisper"

    # читаем обратно с диска после сброса
    monkeypatch.setattr(config.settings, "asr_engine", "gigaam")
    monkeypatch.setattr(config.settings, "asr_model", "v3_e2e_rnnt")
    load_asr_choice()
    assert config.settings.asr_engine == "faster_whisper"
    assert config.settings.asr_model == "small"


# --- Версия сервера ---
#
# Нужна ровно для одного: понять, отстала машина или нет. Прежняя «0.1.0» была
# захардкожена и одинакова у образа от 24 августа и у сегодняшнего — то есть
# как признак расхождения бесполезна.

def test_версия_по_умолчанию_dev():
    assert Settings(_env_file=None).version == "dev"


def test_версия_из_окружения(monkeypatch):
    monkeypatch.setenv("STENOGRAF_VERSION", "a488cfa")
    assert Settings(_env_file=None).version == "a488cfa"


@pytest.mark.parametrize("значение, ожидание", [
    ("a488cfa1b2c3d4e5f60718293a4b5c6d7e8f9012", "a488cfa"),  # полный sha режется
    ("dev", "dev"),                                            # локальный запуск
    ("0.2.0", "0.2.0"),                                        # обычная версия не трогается
    ("a488cfa", "a488cfa"),                                    # уже короткий
])
def test_короткая_версия_в_api(значение, ожидание, monkeypatch):
    """Сорок символов в строке состояния читать невозможно, семи хватает,
    чтобы найти коммит."""
    import app.main as main
    monkeypatch.setattr(main.settings, "version", значение)
    assert main._версия() == ожидание
