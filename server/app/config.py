"""Конфигурация сервера. Все параметры переопределяются переменными окружения
с префиксом STENOGRAF_ (например STENOGRAF_ASR_MODEL=base) или файлом .env."""
import json
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

SAMPLE_RATE = 16_000  # весь пайплайн работает на 16 кГц mono

ASR_ENGINES = ("faster_whisper", "mlx")
ASR_MODELS = ("tiny", "base", "small")  # medium на 8 ГБ рядом с Ollama не помещается

_SERVER_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STENOGRAF_", env_file=_SERVER_DIR / ".env")

    data_dir: Path = _SERVER_DIR / "data"

    # --- ASR ---
    asr_engine: str = "faster_whisper"  # faster_whisper (CPU) | mlx (GPU Metal, только Apple Silicon)
    asr_model: str = "small"        # tiny | base | small
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_language: str = "ru"        # "auto" — автоопределение языка
    asr_beam_size: int = 1          # 1 = greedy, быстрее на CPU
    preload_asr: bool = True        # грузить модель при старте, а не при первой фразе

    # --- VAD / сегментация речи ---
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250    # короче — отбрасываем (щелчки, вздохи)
    vad_min_silence_ms: int = 500   # пауза, после которой фраза считается законченной
    vad_pad_ms: int = 150           # запас аудио по краям фразы
    vad_max_segment_s: float = 15.0  # принудительная нарезка длинного монолога

    # --- Диаризация (глобальная база голосов) ---
    speaker_match_threshold: float = 0.35  # косинусная близость ECAPA для "тот же человек"
    speaker_min_embed_s: float = 0.4       # короче — не считаем эмбеддинг, берём последнего спикера
    speaker_max_samples: int = 3           # аудио-образцов голоса на профиль
    speaker_sample_min_s: float = 0.8
    speaker_sample_max_s: float = 8.0
    speaker_centroid_max_count: int = 200  # ограничение веса центроида (чтобы профиль мог "дрейфовать")

    # --- LLM (Ollama) ---
    ollama_url: str = "http://127.0.0.1:11434"
    summary_model: str = "qwen3:4b"
    # Для подсказок в реальном времени — модель поменьше: быстрее отвечает и
    # спокойно живёт в памяти рядом с ASR на 8 ГБ RAM
    hints_model: str = "qwen3:1.7b"
    llm_keep_alive: str = "2m"      # сколько Ollama держит модель в RAM после запроса
    hints_interval_s: float = 25.0  # период подсказок в демо-режиме
    hints_window_chars: int = 2500  # сколько последних символов транскрипта видит LLM

    @property
    def db_path(self) -> Path:
        return self.data_dir / "stenograf.db"

    @property
    def samples_dir(self) -> Path:
        return self.data_dir / "samples"

    @property
    def recordings_dir(self) -> Path:
        return self.data_dir / "recordings"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"


settings = Settings()


def _asr_choice_path() -> Path:
    return settings.data_dir / "asr.json"


def load_asr_choice() -> None:
    """Выбор движка/модели, сделанный из приложения, важнее env-дефолтов."""
    try:
        data = json.loads(_asr_choice_path().read_text())
    except (OSError, ValueError):
        return
    if data.get("engine") in ASR_ENGINES:
        settings.asr_engine = data["engine"]
    if data.get("model") in ASR_MODELS:
        settings.asr_model = data["model"]


def save_asr_choice(engine: str, model: str) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    _asr_choice_path().write_text(json.dumps({"engine": engine, "model": model}))
    settings.asr_engine = engine
    settings.asr_model = model


load_asr_choice()

# mlx-whisper качает модели через huggingface_hub — держим кэш рядом с остальными
# моделями в data/models, а не в ~/.cache
os.environ.setdefault("HF_HUB_CACHE", str(settings.models_dir / "hf"))
