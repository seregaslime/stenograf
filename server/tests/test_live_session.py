"""Тесты правил LiveSession, влияющих на то, кому припишется реплика:
короткий сегмент («да», «угу») без эмбеддинга приписывается последнему
говорившему, но только в течение 4 секунд.
"""
import asyncio

import numpy as np
import pytest

from app.audio.vad import SpeechSegment
from app.diarization.registry import SpeakerRegistry
from app.ws import LiveSession


class _FakeEmbedder:
    """Возвращает заранее заданный вектор — реальная ECAPA тут не нужна."""

    def __init__(self, vector: np.ndarray):
        self.vector = vector
        self.calls = 0

    def embed(self, audio: np.ndarray) -> np.ndarray:
        self.calls += 1
        return self.vector


def _make_session(cfg, registry, embedder) -> LiveSession:
    return LiveSession(
        ws=None, cfg=cfg, transcriber=None, embedder=embedder,
        registry=registry, ollama=None, on_meeting_ended=lambda mid: None,
    )


def _segment(start_s: float, duration_s: float) -> SpeechSegment:
    samples = int(duration_s * 16_000)
    return SpeechSegment(np.zeros(samples, dtype=np.float32), start_s, start_s + duration_s)


@pytest.fixture()
def registry(cfg, db_session) -> SpeakerRegistry:
    reg = SpeakerRegistry(cfg)
    reg.load(db_session)
    return reg


def rand_unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(192).astype(np.float32)
    return v / np.linalg.norm(v)


def test_short_segment_reuses_last_speaker(cfg, db_session, registry):
    """Короткое «угу» сразу после реплики — тот же человек, эмбеддинг не считается."""
    embedder = _FakeEmbedder(rand_unit(1))
    session = _make_session(cfg, registry, embedder)

    long_seg = _segment(0.0, 2.0)
    first = asyncio.run(session._match_speaker(db_session, long_seg, "system"))
    session._last_match, session._last_match_end = first, long_seg.end_s

    short_seg = _segment(2.5, cfg.speaker_min_embed_s / 2)  # короче минимума
    match = asyncio.run(session._match_speaker(db_session, short_seg, "system"))

    assert match.speaker_id == first.speaker_id
    assert match.similarity is None  # эмбеддинг не считался
    assert embedder.calls == 1  # только для длинной реплики


def test_short_segment_after_long_pause_is_embedded(cfg, db_session, registry):
    """Пауза больше 4 с — «прилипание» к последнему спикеру не действует."""
    embedder = _FakeEmbedder(rand_unit(2))
    session = _make_session(cfg, registry, embedder)

    long_seg = _segment(0.0, 2.0)
    first = asyncio.run(session._match_speaker(db_session, long_seg, "system"))
    session._last_match, session._last_match_end = first, long_seg.end_s

    short_seg = _segment(10.0, cfg.speaker_min_embed_s / 2)
    asyncio.run(session._match_speaker(db_session, short_seg, "system"))

    assert embedder.calls == 2  # эмбеддинг посчитан заново


def test_short_segment_inherits_self(cfg, db_session, registry):
    """Короткая реплика после фразы владельца наследует и флаг «Вы»."""
    embedder = _FakeEmbedder(rand_unit(3))
    session = _make_session(cfg, registry, embedder)

    long_seg = _segment(0.0, 2.0)
    first = asyncio.run(session._match_speaker(db_session, long_seg, "mic"))
    assert first.is_self  # первый голос из микрофона — «Вы»
    session._last_match, session._last_match_end = first, long_seg.end_s

    short_seg = _segment(2.5, cfg.speaker_min_embed_s / 2)
    match = asyncio.run(session._match_speaker(db_session, short_seg, "mixed"))
    assert match.is_self
    assert match.speaker_id == first.speaker_id
