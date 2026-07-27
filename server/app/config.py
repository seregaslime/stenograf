"""Конфигурация сервера. Все параметры переопределяются переменными окружения
с префиксом STENOGRAF_ (например STENOGRAF_ASR_MODEL=base) или файлом .env."""
import json
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

SAMPLE_RATE = 16_000  # весь пайплайн работает на 16 кГц mono

ASR_ENGINES = ("faster_whisper", "mlx", "gigaam")
# Допустимые модели каждого движка (medium whisper на 8 ГБ рядом с Ollama не помещается)
ASR_MODELS = {
    "faster_whisper": ("tiny", "base", "small"),
    "mlx": ("tiny", "base", "small"),
    "gigaam": ("v3_e2e_rnnt", "v3_e2e_ctc"),  # русский SOTA, сразу с пунктуацией
}

_SERVER_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STENOGRAF_", env_file=_SERVER_DIR / ".env")

    data_dir: Path = _SERVER_DIR / "data"

    # --- ASR ---
    # Целевой движок — GigaAM (Сбер): для русского на порядок точнее whisper
    # (замер: WER 1.2% против 9.8% у whisper small) и быстрее реального времени
    # на CPU. Если пакет gigaam не установлен, main.py откатывается на whisper.
    asr_engine: str = "gigaam"      # gigaam | faster_whisper (CPU) | mlx (GPU, Apple Silicon)
    asr_model: str = "v3_e2e_rnnt"  # см. ASR_MODELS
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_language: str = "ru"        # "auto" — автоопределение языка
    asr_beam_size: int = 1          # 1 = greedy, быстрее на CPU
    preload_asr: bool = True        # грузить модель при старте, а не при первой фразе

    # --- Конвейер звука (микшер каналов, денойз) ---
    mixer_max_lag_ms: int = 500     # насколько канал может отставать, прежде чем дополним тишиной
    speaker_channel_dominance: float = 2.0  # во сколько раз RMS канала должен быть громче для доминанты
    # Диалог без паузы VAD не разрежет, но если голос перескочил между каналами
    # (владелец ↔ звонок), говорящий сменился — сегмент режется по смене доминанты:
    segment_split_window_ms: int = 300   # окно оценки доминанты внутри сегмента
    segment_split_min_run_ms: int = 600  # серия короче этого — не смена говорящего (кашель, вздох)
    denoise: str = "off"            # этап чистки шума: off | (реализации подключаются в audio/denoise.py)

    # --- VAD / сегментация речи ---
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250    # короче — отбрасываем (щелчки, вздохи)
    vad_min_silence_ms: int = 300   # пауза, после которой фраза считается законченной;
                                    # при смене говорящего пауза обычно 200–400 мс
    vad_pad_ms: int = 150           # запас аудио по краям фразы
    vad_max_segment_s: float = 15.0  # принудительная нарезка длинного монолога

    # --- Диаризация (глобальная база голосов) ---
    speaker_match_threshold: float = 0.35  # косинусная близость ECAPA для "тот же человек"
    # Живой голос гуляет сильнее порога, поэтому у порога есть скидки-приоры:
    speaker_self_bonus: float = 0.12       # голос из микрофона — скорее всего владелец
    speaker_recent_bonus: float = 0.10     # кто говорил недавно, вероятно говорит и сейчас
    speaker_recent_window_s: float = 30.0  # сколько секунд спикер считается «недавним»
    speaker_max_prints: int = 5            # отпечатков на человека (разные «звучания» голоса)
    speaker_min_embed_s: float = 0.4       # короче — не считаем эмбеддинг, берём последнего спикера
    speaker_print_min_s: float = 2.0       # новый отпечаток «со скидкой» — только из реплики
                                           # такой длины: пограничные коротыши следа не оставляют
    speaker_print_audio_max_s: float = 8.0  # сколько секунд аудио отпечатка хранить для прослушивания
    speaker_centroid_max_count: int = 200  # ограничение веса центроида (чтобы профиль мог "дрейфовать")

    # --- LLM (Ollama) ---
    ollama_url: str = "http://127.0.0.1:11434"
    summary_model: str = "qwen3:4b"
    # Для подсказок в реальном времени — модель поменьше: быстрее отвечает и
    # спокойно живёт в памяти рядом с ASR на 8 ГБ RAM
    hints_model: str = "qwen3:1.7b"
    llm_keep_alive: str = "2m"      # сколько Ollama держит модель в RAM после запроса

    # --- Подсказки во время встречи ---
    # Подсказка выдаётся не по таймеру, а когда накопилось достаточно нового
    # разговора и прошёл минимальный интервал (быстрый API это позволяет).
    hints_window_chars: int = 2500  # сколько последних символов транскрипта видит LLM
    hints_poll_s: float = 3.0       # как часто цикл проверяет, не пора ли подсказать
    hints_min_gap_s: float = 15.0   # минимум между подсказками (чтобы не частить)
    hints_min_new_chars: int = 200  # сколько нового текста накопить перед подсказкой
    hints_memory: int = 3           # сколько последних подсказок помнить (против повторов)
    hints_dup_ratio: float = 0.8    # похожесть [0..1], при которой подсказка — дубль
    hints_max_fails: int = 5        # столько ошибок подряд — подсказки выключаются
    hints_max_backoff_s: float = 120.0  # потолок паузы между повторами после ошибок

    # --- LLM: провайдер (локальная модель ↔ внешний API) ---
    # local — локальная Ollama (по умолчанию; данные не покидают контур).
    # api — OpenAI-совместимый сервер (внутренний сервер организации или внешний
    # сервис). Адрес/ключ/модели берутся ТОЛЬКО из env/.env и на клиент не уходят;
    # переключается тумблером в настройках (см. load_llm_choice/save_llm_choice).
    llm_provider: str = "local"      # local | api
    llm_api_base_url: str = ""       # напр. https://api.openai.com/v1 или http://ai.corp.local:8000/v1
    llm_api_key: str = ""
    llm_api_summary_model: str = ""  # модель API для резюме (напр. gpt-4o-mini)
    llm_api_hints_model: str = ""    # модель API для подсказок (можно ту же)

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
    if data.get("model") in ASR_MODELS.get(settings.asr_engine, ()):
        settings.asr_model = data["model"]


def save_asr_choice(engine: str, model: str) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    _asr_choice_path().write_text(json.dumps({"engine": engine, "model": model}))
    settings.asr_engine = engine
    settings.asr_model = model


LLM_PROVIDERS = ("local", "api")


def _llm_choice_path() -> Path:
    return settings.data_dir / "llm.json"


def load_llm_choice() -> None:
    """Выбор провайдера из приложения важнее env-дефолта (адрес/ключ — из env).
    Если сохранён 'api', но конфигурация подключения пропала, остаёмся на 'local'."""
    try:
        data = json.loads(_llm_choice_path().read_text())
    except (OSError, ValueError):
        return
    provider = data.get("provider")
    if provider == "api" and not (settings.llm_api_base_url and settings.llm_api_key):
        return
    if provider in LLM_PROVIDERS:
        settings.llm_provider = provider


def save_llm_choice(provider: str) -> None:
    if provider not in LLM_PROVIDERS:
        raise ValueError(f"Неизвестный провайдер LLM: {provider}")
    if provider == "api" and not (settings.llm_api_base_url and settings.llm_api_key):
        raise ValueError(
            "API не настроен: задайте STENOGRAF_LLM_API_BASE_URL и "
            "STENOGRAF_LLM_API_KEY в server/.env."
        )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    _llm_choice_path().write_text(json.dumps({"provider": provider}))
    settings.llm_provider = provider


load_asr_choice()
load_llm_choice()

# mlx-whisper качает модели через huggingface_hub — держим кэш рядом с остальными
# моделями в data/models, а не в ~/.cache
os.environ.setdefault("HF_HUB_CACHE", str(settings.models_dir / "hf"))
