"""Тесты правил LiveSession, влияющих на то, кому припишется реплика:
короткий сегмент («да», «угу») без эмбеддинга приписывается последнему
говорившему, но только в течение 4 секунд и только из того же канала;
сегмент со сменой доминанты канала режется на части.
"""
import asyncio

import numpy as np
import pytest

import app.ws as ws_module
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
        ws=None, cfg=cfg, transcriber=None, embedder=embedder, registry=registry,
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

    long_seg = _segment(0.0, 3.0)
    first = asyncio.run(session._match_speaker(db_session, long_seg, "system"))
    session._last_by_channel["system"] = (first, long_seg.end_s)

    short_seg = _segment(3.5, cfg.speaker_min_embed_s / 2)  # короче минимума
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

    long_seg = _segment(0.0, 3.0)
    first = asyncio.run(session._match_speaker(db_session, long_seg, "system"))
    session._last_by_channel["system"] = (first, long_seg.end_s)

    short_seg = _segment(11.0, cfg.speaker_min_embed_s / 2)
    match = asyncio.run(session._match_speaker(db_session, short_seg, "system"))

    assert match is None            # реплика без имени
    assert embedder.calls == 1      # эмбеддер к обрывку даже не звали


def test_short_segment_inherits_self(cfg, db_session, registry):
    """Короткая реплика после фразы владельца наследует и флаг «Вы»."""
    embedder = _FakeEmbedder(rand_unit(3))
    session = _make_session(cfg, registry, embedder)

    long_seg = _segment(0.0, 3.0)
    first = asyncio.run(session._match_speaker(db_session, long_seg, "mic"))
    assert first.is_self  # первый голос из микрофона — «Вы»
    session._last_by_channel["mic"] = (first, long_seg.end_s)

    short_seg = _segment(3.5, cfg.speaker_min_embed_s / 2)
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

    long_seg = _segment(0.0, 3.0)
    first = asyncio.run(session._match_speaker(db_session, long_seg, "mic"))
    session._last_by_channel["mic"] = (first, long_seg.end_s)

    short_seg = _segment(3.2, cfg.speaker_min_embed_s / 2)
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


def _речь(start_s: float, speech_s: float, cfg) -> SpeechSegment:
    """Сегмент с заданной длиной самой речи: VAD добавляет запас по краям."""
    return _segment(start_s, speech_s + 2 * cfg.vad_pad_ms / 1000)


def test_короткая_незнакомая_реплика_не_заводит_спикера(cfg, db_session, registry):
    """«Ха-ха» и «Ну зачем?» знакомого человека на секунде звука не узнаются.

    Раньше каждая такая реплика заводила новый профиль: на записи подкаста
    с двумя голосами их набралось пять лишних. Теперь она остаётся ничьей.
    Длина на полсекунды короче порога — чтобы тест не зависел от того, как
    округлится граница.
    """
    embedder = _FakeEmbedder(rand_unit(10))
    session = _make_session(cfg, registry, embedder)

    короткая = _речь(0.0, cfg.speaker_new_min_s - 0.5, cfg)
    assert asyncio.run(session._match_speaker(db_session, короткая, "system")) is None
    assert embedder.calls == 1  # голос сравнили — просто не с кем


def test_длинная_незнакомая_реплика_заводит_спикера(cfg, db_session, registry):
    """Страж с другой стороны: новый человек, сказавший фразу подлиннее, получает
    профиль сразу, а не копится в «Неизвестных»."""
    session = _make_session(cfg, registry, _FakeEmbedder(rand_unit(11)))

    длинная = _речь(0.0, cfg.speaker_new_min_s + 0.5, cfg)
    match = asyncio.run(session._match_speaker(db_session, длинная, "system"))
    assert match is not None and match.is_new


def test_короткая_реплика_знакомого_голоса_узнаётся(cfg, db_session, registry):
    """Порог только про новых: короткая реплика уже известного голоса приписывается ему."""
    session = _make_session(cfg, registry, _FakeEmbedder(rand_unit(12)))
    знакомый = asyncio.run(session._match_speaker(
        db_session, _речь(0.0, cfg.speaker_new_min_s + 0.5, cfg), "system"))

    короткая = _речь(10.0, cfg.speaker_new_min_s - 0.5, cfg)
    match = asyncio.run(session._match_speaker(db_session, короткая, "system"))
    assert match.speaker_id == знакомый.speaker_id


# --- разрез реплики по смене голоса ---
#
# Двое говорят встык в одном канале: канал не меняется, пауза короче, чем видит
# VAD. Режется по голосу. «Голос» здесь зашит в уровень сигнала, а поддельный
# эмбеддер узнаёт его по среднему значению окна; окно на стыке двух голосов
# даёт третье направление — как смешанное окно в живом звуке.

class _VoiceByLevelEmbedder:
    """Отпечаток по уровню сигнала: 0.1 — один голос, 0.2 — другой, иначе смесь."""

    def __init__(self):
        rng = np.random.default_rng(11)
        self._голоса = {0.1: rand_unit(rng.integers(1_000_000)),
                        0.2: rand_unit(rng.integers(1_000_000))}
        self._смесь = rand_unit(rng.integers(1_000_000))
        self.calls = 0

    def embed(self, audio: np.ndarray) -> np.ndarray:
        self.calls += 1
        return self._голоса.get(round(float(audio.mean()), 2), self._смесь)


def _реплика(start_s: float, *куски: tuple[float, float]) -> SpeechSegment:
    """Куски вида (уровень, секунды) подряд, без пауз между ними."""
    звук = np.concatenate([np.full(int(сек * SAMPLE_RATE), уровень, dtype=np.float32)
                           for уровень, сек in куски])
    return SpeechSegment(звук, start_s, start_s + len(звук) / SAMPLE_RATE)


def test_двое_встык_в_одном_канале_режутся_на_две_реплики(cfg, registry):
    """Главный случай: второй подхватывает сразу за первым, без паузы.

    Разрез по каналу тут бессилен, VAD паузы не видит — остаётся голос.
    """
    session = _make_session(cfg, registry, _VoiceByLevelEmbedder())
    части = asyncio.run(session._split_by_voice(_реплика(10.0, (0.1, 3.0), (0.2, 3.0))))

    assert len(части) == 2
    assert части[0].start_s == 10.0 and части[1].end_s == 16.0
    # Разрез около настоящей смены (13.0): точнее полсекунды по отпечаткам окон
    # не сказать — где внутри окна сменился голос, они не знают.
    assert abs(части[0].end_s - 13.0) <= 0.5
    assert части[0].end_s == части[1].start_s  # ни звука не потеряно на стыке
    assert sum(len(ч.audio) for ч in части) == 6 * SAMPLE_RATE


def test_монолог_одним_голосом_не_режется(cfg, registry):
    session = _make_session(cfg, registry, _VoiceByLevelEmbedder())
    реплика = _реплика(0.0, (0.1, 6.0))
    assert asyncio.run(session._split_by_voice(реплика)) == [реплика]


def test_короткую_реплику_модель_даже_не_смотрит(cfg, registry):
    """Короче двух пар окон сравнивать нечего — и отпечатки не считаются вовсе.

    На живой встрече такие реплики — большинство («да», «угу», короткие
    фразы), и гонять по ним модель впустую значило бы замедлять распознавание.
    """
    эмбеддер = _VoiceByLevelEmbedder()
    session = _make_session(cfg, registry, эмбеддер)
    реплика = _реплика(0.0, (0.1, 1.0), (0.2, 1.0))
    assert asyncio.run(session._split_by_voice(реплика)) == [реплика]
    assert эмбеддер.calls == 0


def test_очередь_обработки_отдаёт_дальше_части_а_не_склейку(cfg, registry):
    """Разрез встроен в потребителя очереди, а не просто существует рядом.

    Тесты выше зовут _split_by_voice напрямую и не заметили бы, если вызов
    уберут из _consume: функция жива, а на встрече реплики снова склеиваются.
    """
    session = _make_session(cfg, registry, _VoiceByLevelEmbedder())
    обработано: list[SpeechSegment] = []

    async def запомнить(meeting_id, segment):
        обработано.append(segment)

    session._process_segment = запомнить

    async def прогон():
        session._queue.put_nowait((1, _реплика(0.0, (0.1, 3.0), (0.2, 3.0))))
        session._queue.put_nowait(ws_module._STOP)
        await session._consume()

    asyncio.run(прогон())
    assert len(обработано) == 2


def test_сбой_одной_части_не_уносит_остальные(cfg, registry):
    """Ревью нашло: try стоял вокруг всего цикла по частям, и падение на первой
    части молча теряло вторую — чужой голос, ради которого разрез и делался."""
    session = _make_session(cfg, registry, _VoiceByLevelEmbedder())
    обработано: list[SpeechSegment] = []

    async def первая_падает(meeting_id, segment):
        if not обработано and segment.start_s == 0.0:
            обработано.append(None)  # отметка, что первая была
            raise RuntimeError("сбой распознавания")
        обработано.append(segment)

    session._process_segment = первая_падает

    async def прогон():
        session._queue.put_nowait((1, _реплика(0.0, (0.1, 3.0), (0.2, 3.0))))
        session._queue.put_nowait(ws_module._STOP)
        await session._consume()

    asyncio.run(прогон())
    assert len([s for s in обработано if s is not None]) == 1  # вторая часть дошла


def test_сбой_разреза_не_теряет_реплику(cfg, registry):
    """Разрез — улучшение, а не условие: упал он — реплика идёт целиком."""
    session = _make_session(cfg, registry, _VoiceByLevelEmbedder())
    обработано: list[SpeechSegment] = []

    async def запомнить(meeting_id, segment):
        обработано.append(segment)

    async def разрез_падает(segment):
        raise RuntimeError("сбой модели")

    session._process_segment = запомнить
    session._split_by_voice = разрез_падает
    реплика = _реплика(0.0, (0.1, 3.0), (0.2, 3.0))

    async def прогон():
        session._queue.put_nowait((1, реплика))
        session._queue.put_nowait(ws_module._STOP)
        await session._consume()

    asyncio.run(прогон())
    assert обработано == [реплика]
