"""Потоковая нарезка аудио на фразы с помощью silero-vad.

На вход подаются куски float32 16 кГц произвольной длины, на выход —
законченные речевые сегменты (аудио + абсолютные таймстемпы от начала потока).
Каждому каналу (mic / system) нужен свой экземпляр SpeechSegmenter.
"""
from dataclasses import dataclass

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

from ..config import SAMPLE_RATE, Settings

WINDOW = 512  # silero-vad работает окнами по 512 сэмплов при 16 кГц


@dataclass
class SpeechSegment:
    audio: np.ndarray  # float32 16 кГц
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class SpeechSegmenter:
    def __init__(self, cfg: Settings):
        self._cfg = cfg
        self._model = load_silero_vad()
        self._iterator = VADIterator(
            self._model,
            threshold=cfg.vad_threshold,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=cfg.vad_min_silence_ms,
            speech_pad_ms=cfg.vad_pad_ms,
        )
        self._tail = np.zeros(0, dtype=np.float32)  # необработанный хвост < WINDOW
        self._buffer = np.zeros(0, dtype=np.float32)  # аудио с начала текущей фразы
        self._buffer_offset = 0  # абсолютный индекс первого сэмпла _buffer
        self._processed = 0  # сколько сэмплов скормлено VAD (абсолютный счётчик)
        self._speech_start: int | None = None

    def feed(self, chunk: np.ndarray) -> list[SpeechSegment]:
        """Принимает очередной кусок аудио, возвращает завершённые фразы."""
        segments: list[SpeechSegment] = []
        data = np.concatenate([self._tail, chunk.astype(np.float32, copy=False)])
        n_windows = len(data) // WINDOW
        self._tail = data[n_windows * WINDOW:]

        self._buffer = np.concatenate([self._buffer, data[: n_windows * WINDOW]])

        for i in range(n_windows):
            window = data[i * WINDOW: (i + 1) * WINDOW]
            result = self._iterator(torch.from_numpy(window))
            self._processed += WINDOW

            if result and "start" in result:
                self._speech_start = int(result["start"])
            elif result and "end" in result and self._speech_start is not None:
                segment = self._cut(self._speech_start, int(result["end"]))
                if segment is not None:
                    segments.append(segment)
                self._speech_start = None

            # Принудительная нарезка слишком длинной непрерывной речи
            if (
                self._speech_start is not None
                and self._processed - self._speech_start
                >= self._cfg.vad_max_segment_s * SAMPLE_RATE
            ):
                segment = self._cut(self._speech_start, self._processed)
                if segment is not None:
                    segments.append(segment)
                self._speech_start = self._processed

        self._trim_buffer()
        return segments

    def flush(self) -> list[SpeechSegment]:
        """Завершение потока: отдаёт недоговорённую фразу, сбрасывает состояние."""
        segments = []
        if self._speech_start is not None:
            segment = self._cut(self._speech_start, self._processed)
            if segment is not None:
                segments.append(segment)
        self._iterator.reset_states()
        self._tail = np.zeros(0, dtype=np.float32)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_offset = self._processed
        self._speech_start = None
        return segments

    def _cut(self, start: int, end: int) -> SpeechSegment | None:
        if (end - start) < self._cfg.vad_min_speech_ms * SAMPLE_RATE / 1000:
            return None
        lo = max(start - self._buffer_offset, 0)
        hi = min(end - self._buffer_offset, len(self._buffer))
        if hi <= lo:
            return None
        return SpeechSegment(
            audio=self._buffer[lo:hi].copy(),
            start_s=start / SAMPLE_RATE,
            end_s=end / SAMPLE_RATE,
        )

    def _trim_buffer(self) -> None:
        """Держим в памяти только аудио от начала текущей фразы (плюс запас на pad)."""
        pad = int(self._cfg.vad_pad_ms * SAMPLE_RATE / 1000) + WINDOW
        keep_from = (self._speech_start - pad) if self._speech_start is not None else (
            self._processed - 2 * SAMPLE_RATE  # 2 секунды на случай ретро-старта фразы
        )
        cut = max(keep_from - self._buffer_offset, 0)
        if cut > 0:
            self._buffer = self._buffer[cut:]
            self._buffer_offset += cut
