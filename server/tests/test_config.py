"""Юнит-тесты конфигурации (config.py): дефолты и персист выбора ASR-движка."""
import json

from app import config
from app.config import Settings, load_asr_choice, save_asr_choice


def test_defaults():
    s = Settings(_env_file=None)
    assert s.asr_engine == "gigaam"
    assert s.llm_provider == "local"
    assert s.hints_min_gap_s == 15.0
    assert s.db_path.name == "stenograf.db"
    assert s.samples_dir == s.data_dir / "samples"


def test_save_and_load_asr_choice(tmp_path, monkeypatch):
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
