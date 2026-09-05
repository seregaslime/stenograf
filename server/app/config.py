"""Конфигурация сервера. Все параметры переопределяются переменными окружения
с префиксом STENOGRAF_ (например STENOGRAF_ASR_MODEL=base) или файлом .env."""
import json
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

SAMPLE_RATE = 16_000  # весь пайплайн работает на 16 кГц mono

ASR_ENGINES = ("faster_whisper", "mlx", "gigaam")
# Допустимые модели каждого движка (medium whisper не помещается в 8 ГБ вместе с диаризацией)
ASR_MODELS = {
    "faster_whisper": ("tiny", "base", "small"),
    "mlx": ("tiny", "base", "small"),
    "gigaam": ("v3_e2e_rnnt", "v3_e2e_ctc"),  # русский SOTA, сразу с пунктуацией
}

_SERVER_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # extra="ignore" — про переезд моделей на клиент: у Сергея и у куратора в
    # .env и в docker-compose остались STENOGRAF_LLM_* и STENOGRAF_OLLAMA_URL от
    # тех времён, когда генерацию звал сервер. По умолчанию pydantic на
    # незнакомую переменную с нашим префиксом падает при импорте — сервер не
    # поднялся бы вообще, и по сообщению «Extra inputs are not permitted» никто
    # бы не догадался просто вычистить старые строки.
    model_config = SettingsConfigDict(
        env_prefix="STENOGRAF_", env_file=_SERVER_DIR / ".env", extra="ignore",
    )

    data_dir: Path = _SERVER_DIR / "data"

    # Версия кода. Задаётся при сборке образа (STENOGRAF_VERSION=<git sha>),
    # локально остаётся "dev". Нужна ровно для одного: понять, отстала машина
    # или нет. Прежняя захардкоженная «0.1.0» на этот вопрос не отвечала — она
    # одинакова у образа от 24 августа и у сегодняшнего.
    version: str = "dev"

    # --- Доступ ---
    # Кому браузер разрешит читать ответы сервера, вдобавок к локальным адресам
    # (те разрешены выражением в main.py — vite берёт свободный порт, и
    # перечислить их нельзя). Иллюзий насчёт CORS быть не должно: он защищает
    # пользователя браузера от чужих сайтов, а сервер от curl — никак; сервер
    # закрывает токен.
    # «null» обязателен: собранное приложение грузится с file://, и Chromium
    # шлёт для него именно такой Origin. Без него сборка у куратора перестанет
    # ходить на сервер, а в dev-режиме всё будет работать — и сразу не поймёшь.
    cors_origins: str = "null"


    # --- ASR ---
    # Целевой движок — GigaAM (Сбер): для русского на порядок точнее whisper
    # (замер: WER 1.2% против 9.8% у whisper small) и быстрее реального времени
    # на CPU. Если пакет gigaam не установлен, main.py откатывается на whisper.
    asr_engine: str = "gigaam"      # gigaam | faster_whisper (CPU) | mlx (GPU, Apple Silicon)
    asr_model: str = "v3_e2e_rnnt"  # см. ASR_MODELS
    # auto — выбрать самим: cuda → mps → cpu (см. app/device.py). Раньше здесь
    # стоял «cpu», и он никогда не менялся: у куратора на Windows всё считалось
    # процессором при живой видеокарте. Значения cuda/mps/cpu задают устройство
    # руками; недоступное молча откатывается на процессор.
    asr_device: str = "auto"
    asr_compute_type: str = "auto"  # auto: float16 на видеокарте, int8 на процессоре
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

    # --- Поиск по прошлым встречам ---
    # Векторы считает приложение своей моделью — сервер их только хранит и
    # перебирает, поэтому имени модели эмбеддингов здесь больше нет: у каждого
    # человека она своя, и приходит она в запросе.
    # Реплики короткие (в среднем 49 символов на живых встречах) — вектор от
    # «Да-да, согласен» это шум. Ищем по кускам разговора: склеиваем подряд
    # идущие реплики, пока не наберётся столько символов.
    search_chunk_chars: int = 600
    search_top_k: int = 5           # сколько кусков отдавать на один запрос

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


def cors_origin_list() -> list[str]:
    """Разрешённые Origin списком. Пусто — не разрешаем никому: приложение
    ходит на сервер и без CORS (Electron), а браузерная версия должна быть
    вписана явно."""
    return [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
