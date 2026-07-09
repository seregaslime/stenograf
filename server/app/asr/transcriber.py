"""Обёртка над ASR-движками: faster-whisper (CPU) и mlx-whisper (GPU Metal).

Модель грузится лениво и живёт одна на процесс. Инференс запускается через
asyncio.to_thread под глобальным asyncio.Lock — одновременно транскрибируется
один сегмент, очередь сегментов ждёт (на M-серии whisper small работает быстрее
реального времени, очередь не растёт).

Движок и размер модели можно менять на лету через reconfigure() — этим
пользуется POST /api/asr, когда пользователь переключает модель в настройках.
"""
import asyncio
import gc
import logging
import threading

import numpy as np
from faster_whisper import WhisperModel

from ..config import Settings

log = logging.getLogger(__name__)

try:
    import mlx_whisper  # доступен только на Apple Silicon

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Квантованные q4-варианты: в ~3 раза меньше памяти, чем fp16, качество почти то же
_MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx-q4",
    "base": "mlx-community/whisper-base-mlx-q4",
    "small": "mlx-community/whisper-small-mlx-q4",
}

# Типовые галлюцинации whisper на тишине/шуме (русский)
_JUNK = {
    "продолжение следует...",
    "субтитры сделал dimatorzok",
    "субтитры подогнал «симон»",
    "редактор субтитров а.семкин корректор а.егорова",
    "спасибо за просмотр!",
    "спасибо за просмотр",
}

_NO_SPEECH_MAX = 0.85


class _FasterWhisperBackend:
    def __init__(self, model_name: str, cfg: Settings):
        self._model_name = model_name
        self._cfg = cfg
        self._model: WhisperModel | None = None

    def load(self) -> None:
        self._model = WhisperModel(
            self._model_name,
            device=self._cfg.asr_device,
            compute_type=self._cfg.asr_compute_type,
            download_root=str(self._cfg.models_dir),
        )

    def transcribe(self, audio: np.ndarray, language: str | None) -> list[str]:
        segments, _info = self._model.transcribe(
            audio,
            language=language,
            beam_size=self._cfg.asr_beam_size,
            condition_on_previous_text=False,
            vad_filter=False,  # нарезка уже сделана нашим VAD
        )
        return [s.text.strip() for s in segments if s.no_speech_prob < _NO_SPEECH_MAX]


class _MlxBackend:
    def __init__(self, model_name: str):
        self._repo = _MLX_REPOS[model_name]

    def load(self) -> None:
        # mlx_whisper держит одну модель в кэше процесса и грузит её при первом
        # transcribe — прогреваем секундой тишины, чтобы первая фраза не ждала
        mlx_whisper.transcribe(np.zeros(16_000, dtype=np.float32), path_or_hf_repo=self._repo)

    def transcribe(self, audio: np.ndarray, language: str | None) -> list[str]:
        # beam search в mlx-whisper нет — всегда greedy (как и наш beam_size=1)
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._repo,
            language=language,
            condition_on_previous_text=False,
        )
        return [
            s["text"].strip()
            for s in result["segments"]
            if s.get("no_speech_prob", 0.0) < _NO_SPEECH_MAX
        ]

    @staticmethod
    def release() -> None:
        """Выкинуть модель из внутреннего кэша mlx_whisper (при уходе с mlx)."""
        try:
            from mlx_whisper.transcribe import ModelHolder

            ModelHolder.model = None
            ModelHolder.model_path = None
        except Exception:
            pass


class Transcriber:
    def __init__(self, cfg: Settings):
        self._cfg = cfg
        self._engine = cfg.asr_engine
        self._model_name = cfg.asr_model
        self._backend: _FasterWhisperBackend | _MlxBackend | None = None
        self._loading = False
        self._load_error: str | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = asyncio.Lock()

    @property
    def engine(self) -> str:
        return self._engine

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def loaded(self) -> bool:
        return self._backend is not None

    @property
    def loading(self) -> bool:
        return self._loading

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> None:
        with self._load_lock:
            if self._backend is not None:
                return
            log.info("Загрузка ASR: движок '%s', модель '%s'...", self._engine, self._model_name)
            self._loading = True
            try:
                if self._engine == "mlx":
                    backend = _MlxBackend(self._model_name)
                else:
                    backend = _FasterWhisperBackend(self._model_name, self._cfg)
                backend.load()
                self._backend = backend
                self._load_error = None
                log.info("ASR загружен")
            except Exception as exc:
                self._load_error = str(exc)
                log.exception("Не удалось загрузить ASR")
                raise
            finally:
                self._loading = False

    def _reconfigure_sync(self, engine: str, model: str) -> None:
        with self._load_lock:
            if engine == self._engine and model == self._model_name:
                return
            was_mlx = self._engine == "mlx"
            self._engine = engine
            self._model_name = model
            self._backend = None
            self._load_error = None
            if was_mlx and engine != "mlx":
                _MlxBackend.release()
            gc.collect()
        log.info("ASR переключён: движок '%s', модель '%s'", engine, model)

    async def reconfigure(self, engine: str, model: str) -> None:
        """Сменить движок/модель. Ждёт конца текущего сегмента, старую модель
        отпускает сразу; новая загрузится лениво (или фоновым прогревом).
        Локи берём в to_thread: идущая загрузка может держать _load_lock долго
        (скачивание модели), event loop блокировать нельзя."""
        async with self._infer_lock:
            await asyncio.to_thread(self._reconfigure_sync, engine, model)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        self.load()
        language = None if self._cfg.asr_language == "auto" else self._cfg.asr_language
        parts = self._backend.transcribe(audio, language)
        text = " ".join(p for p in parts if p).strip()
        if text.lower().strip(" .!") in _JUNK or text.lower() in _JUNK:
            return ""
        return text

    async def transcribe(self, audio: np.ndarray) -> str:
        async with self._infer_lock:
            return await asyncio.to_thread(self._transcribe_sync, audio)
