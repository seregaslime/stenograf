"""Интеграционный тест: время слов на настоящем GigaAM.

Запуск (грузит модель распознавания из server/data/models):
    .venv/bin/python -m pytest -m integration

По номерам слов режется текст реплики, когда человек отдаёт её часть другому
спикеру. Поэтому проверяется не «время как-то есть», а то, на что опирается
разрез: слова складываются ровно в текст, идут по порядку и не вылезают за звук.
Записи — эталон regress.py: синтезированные фразы с известным текстом.
"""
import asyncio
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from app.asr.transcriber import GIGAAM_AVAILABLE, Transcriber
from app.config import SAMPLE_RATE, Settings

pytestmark = pytest.mark.integration

SERVER_DIR = Path(__file__).resolve().parent.parent
FIXTURES = SERVER_DIR / "tests" / "fixtures" / "regress"


def читать_wav(путь: Path) -> np.ndarray:
    with wave.open(str(путь)) as f:
        return np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768


@pytest.fixture(scope="module")
def распознавание() -> Transcriber:
    if not GIGAAM_AVAILABLE:
        pytest.skip("gigaam не установлен")
    if not (SERVER_DIR / "data" / "models" / "gigaam").exists():
        pytest.skip("нет кэша моделей server/data/models — сначала запустите сервер")
    cfg = Settings(data_dir=SERVER_DIR / "data", asr_engine="gigaam", _env_file=None)
    tr = Transcriber(cfg)
    tr.load()
    return tr


def test_слова_складываются_в_текст_и_лежат_внутри_звука(распознавание):
    фразы = json.loads((FIXTURES / "reference.json").read_text(encoding="utf-8"))["speech"]
    for фраза in фразы:
        звук = читать_wav(FIXTURES / фраза["file"])
        text, words = asyncio.run(распознавание.recognize(звук))
        assert words, f"нет времени слов: {фраза['file']}"
        assert " ".join(w for _, _, w in words) == text
        начала = [начало for начало, _, _ in words]
        assert начала == sorted(начала), f"слова не по порядку: {words}"
        assert all(0 <= начало <= конец <= len(звук) / SAMPLE_RATE + 0.1
                   for начало, конец, _ in words), words
