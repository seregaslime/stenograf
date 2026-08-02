"""Юнит-тесты буферной логики сегментатора (audio/vad.py): нарезка _cut и
подрезка памяти _trim_buffer. Silero-модель не грузим — конструируем через
__new__ и дёргаем чистые методы напрямую."""
import numpy as np

from app.audio.vad import WINDOW, SpeechSegmenter
from app.config import SAMPLE_RATE


def _segmenter(cfg, buffer, buffer_offset=0, processed=0, speech_start=None):
    s = object.__new__(SpeechSegmenter)  # без __init__ → без load_silero_vad()
    s._cfg = cfg
    s._buffer = buffer
    s._buffer_offset = buffer_offset
    s._processed = processed
    s._speech_start = speech_start
    return s


def test_cut_rejects_too_short(cfg):
    """Фрагмент короче порога речи не становится сегментом (щелчки и вздохи отбрасываются)."""
    n = int(cfg.vad_min_speech_ms * SAMPLE_RATE / 1000) - 1
    s = _segmenter(cfg, np.ones(SAMPLE_RATE, dtype=np.float32))
    assert s._cut(0, n) is None


def test_cut_returns_segment_with_timestamps(cfg):
    """У вырезанного сегмента корректные метки времени, а аудио — копия, а не вид на общий буфер.
    """
    buf = np.arange(SAMPLE_RATE, dtype=np.float32)  # 1 c, offset 0
    seg = _segmenter(cfg, buf, buffer_offset=0)._cut(0, SAMPLE_RATE)
    assert seg is not None
    assert seg.start_s == 0.0 and seg.end_s == 1.0
    assert len(seg.audio) == SAMPLE_RATE
    assert seg.audio is not buf  # копия, не вид на общий буфер


def test_cut_clamps_to_buffer(cfg):
    """Запрос за границу буфера обрезается по факту имеющихся данных, а не падает."""
    buf = np.ones(SAMPLE_RATE, dtype=np.float32)  # буфер = сэмплы [1..2) c
    seg = _segmenter(cfg, buf, buffer_offset=SAMPLE_RATE)._cut(SAMPLE_RATE, 2 * SAMPLE_RATE)
    assert seg is not None and len(seg.audio) == SAMPLE_RATE


def test_trim_buffer_during_speech_keeps_from_start(cfg):
    """Во время речи буфер подрезается от начала фразы с запасом паддинга — начало реплики не теряется.
    """
    total = 5 * SAMPLE_RATE
    speech_start = 3 * SAMPLE_RATE
    s = _segmenter(cfg, np.arange(total, dtype=np.float32),
                   buffer_offset=0, processed=total, speech_start=speech_start)
    s._trim_buffer()
    pad = int(cfg.vad_pad_ms * SAMPLE_RATE / 1000) + WINDOW
    assert s._buffer_offset == speech_start - pad
    assert len(s._buffer) == total - (speech_start - pad)


def test_trim_buffer_idle_keeps_recent(cfg):
    """В тишине буфер не растёт бесконечно: хранятся только последние секунды."""
    total = 10 * SAMPLE_RATE
    s = _segmenter(cfg, np.zeros(total, dtype=np.float32),
                   buffer_offset=0, processed=total, speech_start=None)
    s._trim_buffer()
    assert s._buffer_offset == total - 2 * SAMPLE_RATE  # держим последние ~2 c
