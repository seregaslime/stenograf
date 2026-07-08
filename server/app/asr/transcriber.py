"""Обёртка над faster-whisper. Модель грузится лениво и живёт одна на процесс.

CPU-инференс запускается через asyncio.to_thread под глобальным asyncio.Lock —
одновременно транскрибируется один сегмент, очередь сегментов ждёт (на M-серии
whisper small int8 работает быстрее реального времени, очередь не растёт).
"""
import asyncio
import logging
import threading

import numpy as np
from faster_whisper import WhisperModel

from ..config import Settings

log = logging.getLogger(__name__)

# Типовые галлюцинации whisper на тишине/шуме (русский)
_JUNK = {
    "продолжение следует...",
    "субтитры сделал dimatorzok",
    "субтитры подогнал «симон»",
    "редактор субтитров а.семкин корректор а.егорова",
    "спасибо за просмотр!",
    "спасибо за просмотр",
}


class Transcriber:
    def __init__(self, cfg: Settings):
        self._cfg = cfg
        self._model: WhisperModel | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._load_lock:
            if self._model is None:
                log.info("Загрузка whisper '%s' (%s, %s)...",
                         self._cfg.asr_model, self._cfg.asr_device, self._cfg.asr_compute_type)
                self._model = WhisperModel(
                    self._cfg.asr_model,
                    device=self._cfg.asr_device,
                    compute_type=self._cfg.asr_compute_type,
                    download_root=str(self._cfg.models_dir),
                )
                log.info("Whisper загружен")

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        self.load()
        language = None if self._cfg.asr_language == "auto" else self._cfg.asr_language
        segments, _info = self._model.transcribe(
            audio,
            language=language,
            beam_size=self._cfg.asr_beam_size,
            condition_on_previous_text=False,
            vad_filter=False,  # нарезка уже сделана нашим VAD
        )
        parts = [s.text.strip() for s in segments if s.no_speech_prob < 0.85]
        text = " ".join(p for p in parts if p).strip()
        if text.lower().strip(" .!") in _JUNK or text.lower() in _JUNK:
            return ""
        return text

    async def transcribe(self, audio: np.ndarray) -> str:
        async with self._infer_lock:
            return await asyncio.to_thread(self._transcribe_sync, audio)
