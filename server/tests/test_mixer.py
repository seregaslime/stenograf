"""Тесты микшера каналов: выравнивание потоков, паддинг отстающего канала,
RMS-доминанта для диаризации."""
import numpy as np
import pytest

from app.audio.mixer import ChannelMixer
from app.config import SAMPLE_RATE


def const_chunk(value: float, n: int) -> np.ndarray:
    return np.full(n, value, dtype=np.float32)


def drain_all(chunks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


@pytest.fixture()
def mixer(cfg) -> ChannelMixer:
    return ChannelMixer(cfg)


def test_two_channels_are_summed_aligned(mixer):
    """Каналы приходят разными чанками — на выходе поэлементная сумма."""
    out = []
    out += mixer.feed("mic", const_chunk(0.2, 300))
    out += mixer.feed("system", const_chunk(0.3, 200))  # готово только 200
    mixed = drain_all(out)
    assert len(mixed) == 200
    assert np.allclose(mixed, 0.5)
    out = mixer.feed("system", const_chunk(0.3, 100))  # догнал mic
    mixed = drain_all(out)
    assert len(mixed) == 100
    assert np.allclose(mixed, 0.5)


def test_single_channel_passthrough(cfg):
    """Системный источник выключен — после стартовой паузы (ждём второй канал
    не дольше max_lag) микрофон идёт как есть."""
    mixer = ChannelMixer(cfg)
    lag = int(cfg.mixer_max_lag_ms / 1000 * SAMPLE_RATE)
    assert drain_all(mixer.feed("mic", const_chunk(0.4, lag // 2))).size == 0  # ещё ждём
    mixed = drain_all(mixer.feed("mic", const_chunk(0.4, lag)))
    assert len(mixed) == lag + lag // 2  # дождались max_lag — всё накопленное вышло
    assert np.allclose(mixed, 0.4)
    # дальше — без задержек
    mixed = drain_all(mixer.feed("mic", const_chunk(0.4, 1000)))
    assert len(mixed) == 1000


def test_lagging_channel_padded_and_late_samples_dropped(cfg):
    """Канал отстал больше max_lag — доигрываем с тишиной, а опоздавшие
    семплы отбрасываем, чтобы шкала времени не разъехалась."""
    mixer = ChannelMixer(cfg)
    lag = int(cfg.mixer_max_lag_ms / 1000 * SAMPLE_RATE)
    mixer.feed("system", const_chunk(0.3, 10))  # system активен, но замолчал
    out = mixer.feed("mic", const_chunk(0.2, lag + 500))
    mixed = drain_all(out)
    assert len(mixed) == 500  # выдано ровно то, что старше max_lag
    assert np.allclose(mixed[:10], 0.5)   # где system был — сумма
    assert np.allclose(mixed[10:], 0.2)   # дальше — тишина вместо system
    # опоздавший system-чанк выбрасывается в счёт «долга»
    out = mixer.feed("system", const_chunk(0.9, 490))
    assert drain_all(out).size == 0


def test_flush_drains_remainder(mixer):
    mixer.feed("mic", const_chunk(0.1, 700))
    mixer.feed("system", const_chunk(0.1, 300))
    drained_before = drain_all(mixer.flush())
    assert len(drained_before) == 400  # хвост mic, system дополнен тишиной
    assert np.allclose(drained_before, 0.1)


def test_clipping_protection(mixer):
    mixer.feed("mic", const_chunk(0.8, 100))
    out = mixer.feed("system", const_chunk(0.8, 100))
    assert float(drain_all(out).max()) <= 1.0


def test_dominance_mic_system_mixed(cfg):
    mixer = ChannelMixer(cfg)
    second = SAMPLE_RATE
    # 1-я секунда: громкий mic, тихий system; 2-я: наоборот; 3-я: поровну
    mic = np.concatenate([const_chunk(0.5, second), const_chunk(0.01, second), const_chunk(0.3, second)])
    system = np.concatenate([const_chunk(0.01, second), const_chunk(0.5, second), const_chunk(0.3, second)])
    chunk = 1600  # каналы приходят вперемешку кадрами по 100 мс, как с клиента
    for i in range(0, len(mic), chunk):
        mixer.feed("mic", mic[i:i + chunk])
        mixer.feed("system", system[i:i + chunk])
    assert mixer.dominance(0.0, 1.0) == "mic"
    assert mixer.dominance(1.0, 2.0) == "system"
    assert mixer.dominance(2.0, 3.0) == "mixed"


def test_dominance_single_active_channel(cfg):
    mixer = ChannelMixer(cfg)
    mixer.feed("mic", const_chunk(0.2, 1000))
    assert mixer.dominance(0.0, 0.05) == "mic"


def _feed_interleaved(mixer, mic: np.ndarray, system: np.ndarray) -> None:
    chunk = 1600  # каналы приходят вперемешку кадрами по 100 мс, как с клиента
    for i in range(0, len(mic), chunk):
        mixer.feed("mic", mic[i:i + chunk])
        mixer.feed("system", system[i:i + chunk])


def test_dominance_spans_split_on_channel_flip(cfg):
    """Ответ без паузы: 1.2 с из звонка, потом 1.2 с в микрофон — сегмент
    делится на два участка ровно в точке смены канала."""
    mixer = ChannelMixer(cfg)
    n = int(1.2 * SAMPLE_RATE)
    system = np.concatenate([const_chunk(0.5, n), const_chunk(0.01, n)])
    mic = np.concatenate([const_chunk(0.01, n), const_chunk(0.5, n)])
    _feed_interleaved(mixer, mic, system)

    spans = mixer.dominance_spans(0.0, 2.4, 0.3, 0.6)
    assert len(spans) == 2
    assert spans[0][0] == 0.0 and spans[1][1] == 2.4
    assert spans[0][1] == pytest.approx(1.2)
    assert spans[1][0] == pytest.approx(1.2)


def test_dominance_spans_ignores_short_blip(cfg):
    """Кашель в микрофон (одно окно) посреди чужой фразы — не смена говорящего."""
    mixer = ChannelMixer(cfg)
    n = int(0.3 * SAMPLE_RATE)
    mic = np.concatenate([const_chunk(0.01, 4 * n), const_chunk(0.6, n), const_chunk(0.01, 3 * n)])
    system = const_chunk(0.25, 8 * n)
    _feed_interleaved(mixer, mic, system)

    assert mixer.dominance_spans(0.0, 2.4, 0.3, 0.6) == [(0.0, 2.4)]


def test_dominance_spans_single_channel(cfg):
    """Второго канала нет — резать не по чему."""
    mixer = ChannelMixer(cfg)
    mixer.feed("mic", const_chunk(0.2, SAMPLE_RATE))
    assert mixer.dominance_spans(0.0, 1.0, 0.3, 0.6) == [(0.0, 1.0)]
