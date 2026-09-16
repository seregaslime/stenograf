"""Юнит-тесты фильтрации мусора ASR (asr/transcriber.py `_recognize_sync`):
типовые галлюцинации whisper на тишине отбрасываются, время слов движка доходит
до конвейера. Модель не грузим — подменяем backend, load() видит его и не качает
веса."""
import numpy as np

from app.asr.transcriber import Recognized, Transcriber
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
    """Типичная галлюцинация ASR на тишине («Продолжение следует...») выбрасывается."""
    assert _tr(tmp_path, ["Продолжение следует..."])._transcribe_sync(AUDIO) == ""


def test_junk_case_and_punctuation_dropped(tmp_path):
    """Фильтр галлюцинаций не зависит от регистра и знаков препинания."""
    assert _tr(tmp_path, ["СПАСИБО ЗА ПРОСМОТР!"])._transcribe_sync(AUDIO) == ""


def test_real_text_kept(tmp_path):
    """Осмысленная речь фильтром не трогается."""
    assert _tr(tmp_path, ["Привет, коллеги"])._transcribe_sync(AUDIO) == "Привет, коллеги"


def test_parts_joined_and_empties_dropped(tmp_path):
    """Куски распознавания склеиваются через пробел, пустые пропускаются."""
    assert _tr(tmp_path, ["привет", "", "мир"])._transcribe_sync(AUDIO) == "привет мир"


def test_no_parts_is_empty(tmp_path):
    """Если распознавать нечего — результат пустая строка, а не ошибка."""
    assert _tr(tmp_path, [])._transcribe_sync(AUDIO) == ""


# --- время слов ---


class _FakeWordsBackend:
    """Движок со временем слов, как GigaAM: Transcriber должен брать его, а не
    голый текст."""

    def __init__(self, text, words):
        self.result = Recognized(text, words)

    def transcribe_words(self, audio):
        return self.result


def test_время_слов_доходит_до_конвейера(tmp_path):
    t = Transcriber(Settings(data_dir=tmp_path, _env_file=None))
    слова = [(0.1, 0.5, "Привет,"), (0.6, 1.0, "коллеги")]
    t._backend = _FakeWordsBackend("Привет, коллеги", слова)
    assert t._recognize_sync(AUDIO) == ("Привет, коллеги", слова)
    assert t._transcribe_sync(AUDIO) == "Привет, коллеги"  # замеры видят только текст


def test_мусор_отбрасывается_вместе_со_словами(tmp_path):
    """Отброшенная галлюцинация не оставляет слов, которые не к чему приложить."""
    t = Transcriber(Settings(data_dir=tmp_path, _env_file=None))
    t._backend = _FakeWordsBackend("Продолжение следует...", [(0.0, 1.0, "Продолжение"), (1.0, 2.0, "следует...")])
    assert t._recognize_sync(AUDIO) == ("", None)


def test_движок_без_времени_слов_делить_не_даёт(tmp_path):
    """whisper-движки времени слов не отдают — реплика хранится без него."""
    assert _tr(tmp_path, ["Привет, коллеги"])._recognize_sync(AUDIO) == ("Привет, коллеги", None)
