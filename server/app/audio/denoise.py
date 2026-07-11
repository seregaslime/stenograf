"""Этап чистки шума в конвейере: заготовка со сменной реализацией.

Место в конвейере: после микшера, до VAD. Сейчас есть только NoopDenoiser —
современные ASR (GigaAM обучен на колл-центрах и шумной речи) от агрессивного
денойза чаще теряют, чем выигрывают, поэтому включать его стоит только по
результатам замеров. Реальная реализация (RNNoise / спектральный гейтинг)
подключается здесь же новым классом и значением STENOGRAF_DENOISE.
"""
import numpy as np

from ..config import Settings


class NoopDenoiser:
    """Пропускает звук как есть."""

    def process(self, chunk: np.ndarray) -> np.ndarray:
        return chunk


def create_denoiser(cfg: Settings):
    if cfg.denoise == "off":
        return NoopDenoiser()
    raise ValueError(f"Неизвестный денойзер: {cfg.denoise!r} (поддерживается: off)")
