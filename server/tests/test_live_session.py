"""Тесты правил LiveSession, влияющих на то, кому припишется реплика:
короткий сегмент («да», «угу») без эмбеддинга приписывается последнему
говорившему, но только в течение 4 секунд и только из того же канала;
сегмент со сменой доминанты канала режется на части.
"""
import asyncio

import numpy as np
import pytest

from app.audio.mixer import ChannelMixer
from app.audio.vad import SpeechSegment
from app.config import SAMPLE_RATE
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
        registry=registry, llm=None, on_meeting_ended=lambda mid: None,
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
    session._last_by_channel["system"] = (first, long_seg.end_s)

    short_seg = _segment(2.5, cfg.speaker_min_embed_s / 2)  # короче минимума
    match = asyncio.run(session._match_speaker(db_session, short_seg, "system"))

    assert match.speaker_id == first.speaker_id
    assert match.similarity is None  # эмбеддинг не считался
    assert embedder.calls == 1  # только для длинной реплики


def test_short_segment_after_long_pause_stays_unattributed(cfg, db_session, registry):
    """Пауза больше 4 с — прилипания нет, но и опознавать нечего.

    Замечание куратора №13: раньше такой обрывок шёл в эмбеддер и заводил
    нового «Спикера N» на каждое «ага». На 0.2 секунды эмбеддер считает не
    голос, а что придётся, поэтому реплика остаётся ничьей.
    """
    embedder = _FakeEmbedder(rand_unit(2))
    session = _make_session(cfg, registry, embedder)

    long_seg = _segment(0.0, 2.0)
    first = asyncio.run(session._match_speaker(db_session, long_seg, "system"))
    session._last_by_channel["system"] = (first, long_seg.end_s)

    short_seg = _segment(10.0, cfg.speaker_min_embed_s / 2)
    match = asyncio.run(session._match_speaker(db_session, short_seg, "system"))

    assert match is None            # реплика без имени
    assert embedder.calls == 1      # эмбеддер к обрывку даже не звали


def test_short_segment_inherits_self(cfg, db_session, registry):
    """Короткая реплика после фразы владельца наследует и флаг «Вы»."""
    embedder = _FakeEmbedder(rand_unit(3))
    session = _make_session(cfg, registry, embedder)

    long_seg = _segment(0.0, 2.0)
    first = asyncio.run(session._match_speaker(db_session, long_seg, "mic"))
    assert first.is_self  # первый голос из микрофона — «Вы»
    session._last_by_channel["mic"] = (first, long_seg.end_s)

    short_seg = _segment(2.5, cfg.speaker_min_embed_s / 2)
    match = asyncio.run(session._match_speaker(db_session, short_seg, "mixed"))
    assert match.is_self
    assert match.speaker_id == first.speaker_id


def test_short_segment_from_other_channel_stays_unattributed(cfg, db_session, registry):
    """Быстрое «да» из звонка сразу после фразы владельца — другой человек.

    Наследовать нельзя (это не владелец), опознавать не по чему (0.2 секунды),
    поэтому реплика остаётся ничьей вместо выдуманного нового участника.
    """
    embedder = _FakeEmbedder(rand_unit(4))
    session = _make_session(cfg, registry, embedder)

    long_seg = _segment(0.0, 2.0)
    first = asyncio.run(session._match_speaker(db_session, long_seg, "mic"))
    session._last_by_channel["mic"] = (first, long_seg.end_s)

    short_seg = _segment(2.2, cfg.speaker_min_embed_s / 2)
    match = asyncio.run(session._match_speaker(db_session, short_seg, "system"))

    assert match is None
    assert embedder.calls == 1


def test_split_by_dominance_cuts_segment(cfg, registry):
    """Сегмент со сменой канала посередине режется на две части: аудио не
    теряется, граница — в точке смены доминанты."""
    session = _make_session(cfg, registry, _FakeEmbedder(rand_unit(5)))
    session._mixer = ChannelMixer(cfg)
    n = int(1.2 * SAMPLE_RATE)
    system = np.concatenate([np.full(n, 0.5, np.float32), np.full(n, 0.01, np.float32)])
    mic = np.concatenate([np.full(n, 0.01, np.float32), np.full(n, 0.5, np.float32)])
    for i in range(0, 2 * n, 1600):  # кадры по 100 мс, как с клиента
        session._mixer.feed("system", system[i:i + 1600])
        session._mixer.feed("mic", mic[i:i + 1600])

    segment = SpeechSegment(np.arange(2 * n, dtype=np.float32), 0.0, 2.4)
    parts = session._split_by_dominance(segment)

    assert len(parts) == 2
    assert np.array_equal(np.concatenate([p.audio for p in parts]), segment.audio)
    assert parts[0].end_s == pytest.approx(1.2)
    assert parts[1].start_s == pytest.approx(1.2)


def test_first_replica_of_a_meeting_is_not_a_new_speaker(cfg, db_session, registry):
    """Первая же реплика встречи — короткая: раньше она заводила спикера.

    Донора ещё нет по определению, и «алло» на старте становилось «Спикером 1»
    с отпечатком голоса, собранным по обрывку. Такой отпечаток потом путал
    диаризацию всей встречи.
    """
    embedder = _FakeEmbedder(rand_unit(7))
    session = _make_session(cfg, registry, embedder)

    short_seg = _segment(0.0, cfg.speaker_min_embed_s / 2)
    match = asyncio.run(session._match_speaker(db_session, short_seg, "mic"))

    assert match is None
    assert embedder.calls == 0


def test_unattributed_replica_leaves_no_voiceprint(cfg, db_session, registry):
    """Ничья реплика не попадает в базу голосов — эмбеддер к ней не звали."""
    embedder = _FakeEmbedder(rand_unit(8))
    session = _make_session(cfg, registry, embedder)
    before = sum(len(prints) for prints in registry._prints.values())

    short_seg = _segment(0.0, cfg.speaker_min_embed_s / 2)
    asyncio.run(session._match_speaker(db_session, short_seg, "system"))

    assert sum(len(prints) for prints in registry._prints.values()) == before


def test_unattributed_replica_does_not_become_a_donor(cfg, db_session, registry):
    """Ничьим обрывком следующие короткие реплики не приписываются.

    Иначе одна неопознанная реплика тянула бы за собой цепочку таких же —
    приписанных неизвестно кому.
    """
    embedder = _FakeEmbedder(rand_unit(9))
    session = _make_session(cfg, registry, embedder)

    short_seg = _segment(0.0, cfg.speaker_min_embed_s / 2)
    assert asyncio.run(session._match_speaker(db_session, short_seg, "mic")) is None

    следующая = _segment(0.5, cfg.speaker_min_embed_s / 2)
    assert asyncio.run(session._match_speaker(db_session, следующая, "mic")) is None
    assert session._short_segment_donor("mic", 0.5) is None
