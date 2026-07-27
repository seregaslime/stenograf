"""Юнит-тесты фильтрации мусора ASR (asr/transcriber.py `_transcribe_sync`):
типовые галлюцинации whisper на тишине отбрасываются. Модель не грузим —
подменяем backend, load() видит его и не качает веса."""
import numpy as np

from app.asr.transcriber import Transcriber
from app.config import Settings

AUDIO = np.zeros(1600, dtype=np.float32)


class _FakeBackend:
    def __init__(self, parts):
        self.parts = parts

    def transcribe(self, audio, language):
        return self.parts


def _tr(tmp_path, parts):
    t = Transcriber(Settings(data_dir=tmp_path, _env_file=None))
    t._backend = _FakeBackend(parts)
    return t


def test_junk_exact_dropped(tmp_path):
    assert _tr(tmp_path, ["Продолжение следует..."])._transcribe_sync(AUDIO) == ""


def test_junk_case_and_punctuation_dropped(tmp_path):
    assert _tr(tmp_path, ["СПАСИБО ЗА ПРОСМОТР!"])._transcribe_sync(AUDIO) == ""


def test_real_text_kept(tmp_path):
    assert _tr(tmp_path, ["Привет, коллеги"])._transcribe_sync(AUDIO) == "Привет, коллеги"


def test_parts_joined_and_empties_dropped(tmp_path):
    assert _tr(tmp_path, ["привет", "", "мир"])._transcribe_sync(AUDIO) == "привет мир"


def test_no_parts_is_empty(tmp_path):
    assert _tr(tmp_path, [])._transcribe_sync(AUDIO) == ""
