"""Конфигурация сервера. Все параметры переопределяются переменными окружения
с префиксом STENOGRAF_ (например STENOGRAF_ASR_MODEL=base) или файлом .env."""
import json
import os
from pathlib import Path

from pydantic import Field
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
    hints_poll_s: float = 3.0       # как часто цикл проверяет, не пора ли подсказать
    hints_min_gap_s: float = 15.0   # минимум между подсказками (чтобы не частить)
    # Сколько нового текста накопить, прежде чем спросить у модели. Порог тут не
    # вместо суждения модели (она сама решает через SKIP), а чтобы не дёргать её
    # на пустом месте: с тем же текстом ответ будет тот же.
    # Было 200 — это ~30 слов, и короткий вопрос («а что такое SLA?» = 16
    # символов) не набирал порог никогда: человек спрашивал и замолкал, ожидая
    # подсказку, а счётчик стоял. Самый ценный повод подсказать оказывался и
    # самым недостижимым.
    # Частоту запросов этот порог не ограничивает — это делают hints_min_gap_s
    # (не чаще раза в 15 c) и бэкофф после SKIP (8 → 16 → … → 45 c). Поэтому
    # порог держим низким: пропускаем короткие вопросы, отсекаем «угу» и «ага».
    hints_min_new_chars: int = 15
    hints_memory: int = 3           # сколько последних подсказок помнить (против повторов)
    hints_dup_ratio: float = 0.85   # похожесть [0..1], при которой подсказка — дубль
    hints_max_fails: int = 5        # столько ошибок подряд — подсказки выключаются
    hints_max_backoff_s: float = 120.0  # потолок паузы между повторами после ошибок
    hints_temperature: float = 0.4  # ниже прежних 0.5 — лучше слушается правило SKIP
    hints_min_context_chars: int = 80   # меньше разговора — рано подсказывать
    # История реплик для подсказок. Держим большой всегда: окно режется срезом по
    # бюджету провайдера, и маленькая дека обнулила бы большое API-окно.
    hints_recent_maxlen: int = 400

    # Право промолчать: модель отвечает SKIP, если полезного нет (см. llm/prompts.py).
    # После SKIP ждём меньше обычного — разговор в любой момент может дойти до
    # важного; но серия SKIP подряд линейно растит паузу, чтобы не жечь CPU.
    hints_skip_gap_s: float = 8.0
    hints_skip_max_gap_s: float = 45.0
    hints_min_len_chars: int = 12   # короче — считаем, что модель промолчала

    # --- Бюджет контекста (зависит от провайдера) ---
    # local: qwen3 с 8k контекста на 8 ГБ RAM — экономим жёстко.
    # api: контекст не жалеем, качество важнее (0 = без ограничения).
    hints_window_chars: int = 2500       # local: сколько символов транскрипта видит LLM
    hints_window_chars_api: int = 40_000  # api: ~16k токенов хвоста разговора
    summary_max_chars_local: int = 12_000
    # Не 0 (без лимита): даже при контексте 131k очень длинная встреча выйдет за
    # предел. 200k символов ≈ 80k токенов — с запасом влезает и покрывает
    # встречу часов на пять.
    summary_max_chars_api: int = 200_000

    # Русский текст в современных токенизаторах — примерно 2.5 символа на токен.
    # Нужно, чтобы прикинуть, влезут ли наши промпты в контекст модели.
    chars_per_token: float = 2.5

    # --- LLM: провайдер (локальная модель ↔ внешний API) ---
    # local — локальная Ollama (по умолчанию; данные не покидают контур).
    # api — OpenAI-совместимый сервер (внутренний сервер организации или внешний
    # сервис). Адрес/ключ/модели берутся ТОЛЬКО из env/.env и на клиент не уходят;
    # переключается тумблером в настройках (см. load_llm_choice/save_llm_choice).
    llm_provider: str = "local"      # local | api
    llm_api_base_url: str = ""       # см. LLM_API_ALLOWED_HOSTS
    llm_api_key: str = ""
    llm_api_summary_model: str = ""  # модель API для резюме
    llm_api_hints_model: str = ""    # модель API для подсказок (можно ту же)
    # Модель с маленьким контекстом молча сломается на первой же встрече: окно
    # подсказок само по себе ~16k токенов. Поэтому в списке выбора показываем
    # только модели, у которых контекст не меньше — размер берём из ответа
    # провайдера, а не из зашитых знаний о моделях.
    llm_api_min_context_tokens: int = 32_768

    # Токены в минуту — то ограничение, в которое реально упираются подсказки
    # (в отличие от размера контекста). В списке моделей его нет, провайдер
    # сообщает его заголовком ответа; меряем при выборе модели и храним здесь
    # по имени модели, у разных моделей лимиты разные.
    llm_api_tpm_limits: dict[str, int] = Field(default_factory=dict)
    # Когда измерить не удалось (упал VPN, провайдер не прислал заголовок,
    # внутренний сервер организации таких заголовков не шлёт). Единственное
    # зашитое здесь число, и оно честно означает «узнать не смогли»: берём
    # нижнюю из встречающихся цифр бесплатного тарифа, чтобы не упереться.
    llm_api_tpm_fallback: int = 6_000
    # Какую ДОЛЮ минутного лимита оставить модели на ответ. Лимит покрывает вход
    # и ответ вместе, поэтому отдать весь его под вход нельзя: либо получим 413,
    # либо пройдём впритык и модель оборвётся, не дописав.
    # Доля, а не фиксированное число токенов: у другого провайдера и лимит
    # другой, а соотношение «сколько просим — столько и ответят» сохраняется.
    # Была половина — с тех пор, когда мысли рассуждающих моделей шли из этого же
    # лимита и стоили столько же, сколько ответ (832 токена мыслей на 822 текста).
    # После того как мысли прижали (reasoning_effort), замер 09.08 показал, что
    # списывается промпт плюс ФАКТИЧЕСКИЙ ответ: без max_tokens за ответ в 118
    # токенов сняли 118, а не резерв (с max_tokens=4000 сняли бы все 4000 —
    # поэтому его и не задаём). Ответ протокола занимает 600–2200 токенов, то
    # есть 8–27% лимита в 8000. Четверть покрывает это с запасом, а бюджету
    # входа достаётся 6000 вместо 4000 — в полтора раза меньше фрагментов и
    # минутных пауз на длинной встрече.
    llm_api_output_share: float = 0.25
    # Насколько подробно модели думать перед ответом. Мысли считаются в тот же
    # минутный лимит, что и ответ, а пользователю не показываются: замер на
    # gpt-oss-120b — 168 токенов мыслей из 270 токенов ответа, то есть 62%
    # лимита уходило впустую. С «low» мыслей осталось 12 из 118.
    # Наши задачи (выписать факты, составить протокол) в рассуждениях не
    # нуждаются — это не олимпиадные задачи.
    # Пустая строка — не передавать параметр вовсе. Провайдер, который его не
    # знает, отвечает 400: тогда клиент повторяет запрос без него (см.
    # OpenAIClient.generate) и больше в этой сессии не присылает.
    llm_api_reasoning_effort: str = "low"

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

# Пока поддерживается только Groq — и это не про «любимого вендора», а про то,
# что мы обязаны знать размер контекста модели. Стандарт OpenAI на /v1/models
# отдаёт только id/object/created/owned_by; context_window — расширение Groq.
# Без него нельзя ни отсеять негодные модели, ни посчитать бюджет промпта, и
# пользователь узнает о проблеме только когда встреча уже идёт.
LLM_API_ALLOWED_HOSTS = ("api.groq.com",)

# Единственный источник правды для адреса: сервер отдаёт его клиенту в
# GET /api/llm, чтобы поле в настройках было заполнено сразу и не разъехалось
# со списком разрешённых хостов.
LLM_API_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


def api_host_supported(base_url: str) -> bool:
    from urllib.parse import urlparse
    return urlparse(base_url).hostname in LLM_API_ALLOWED_HOSTS


API_HOST_HINT = (
    f"Пока поддерживается только Groq ({LLM_API_DEFAULT_BASE_URL}): только он "
    "сообщает размер контекста модели, без которого нельзя проверить, что "
    "выбранная модель потянет наши промпты."
)


def _llm_choice_path() -> Path:
    return settings.data_dir / "llm.json"


def save_tpm_limits(limits: dict[str, int]) -> None:
    """Дописывает измеренные лимиты к уже сохранённому выбору LLM.

    Отдельной функцией, а не параметром save_llm_choice: лимиты выясняются
    запросом к провайдеру уже ПОСЛЕ сохранения выбора, и неудачная проба
    (упал VPN) не должна мешать настройкам сохраниться.
    """
    if not limits:
        return
    settings.llm_api_tpm_limits = {**settings.llm_api_tpm_limits, **limits}
    path = _llm_choice_path()
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return  # выбор ещё не сохранён — записывать лимиты некуда
    data["tpm_limits"] = settings.llm_api_tpm_limits
    path.write_text(json.dumps(data))


def load_llm_choice() -> None:
    """Выбор из приложения важнее env-дефолтов. В llm.json теперь лежит не только
    провайдер, но и адрес/ключ/модели API (их вводят в настройках приложения).
    Если сохранён 'api', но подключение неполно (нет адреса или ключа), остаёмся
    на 'local'."""
    try:
        data = json.loads(_llm_choice_path().read_text())
    except (OSError, ValueError):
        return
    # Креды/модели из файла перекрывают env-дефолты (как у ASR-выбора)
    if data.get("api_base_url"):
        settings.llm_api_base_url = data["api_base_url"]
    if data.get("api_key"):
        settings.llm_api_key = data["api_key"]
    if data.get("summary_model"):
        settings.llm_api_summary_model = data["summary_model"]
    if data.get("hints_model"):
        settings.llm_api_hints_model = data["hints_model"]
    if isinstance(data.get("tpm_limits"), dict):
        settings.llm_api_tpm_limits = {
            model: int(limit) for model, limit in data["tpm_limits"].items()
        }
    provider = data.get("provider")
    if provider == "api" and not (settings.llm_api_base_url and settings.llm_api_key):
        return
    if provider in LLM_PROVIDERS:
        settings.llm_provider = provider


def save_llm_choice(
    provider: str,
    *,
    api_base_url: str | None = None,
    api_key: str | None = None,
    summary_model: str | None = None,
    hints_model: str | None = None,
) -> None:
    """Сохраняет провайдера и (для 'api') креды/модели, введённые в приложении.
    None-поля не трогаются; пустой api_key НЕ затирает сохранённый (клиент
    присылает ключ только когда его меняют — прежний обратно не отдаётся)."""
    if provider not in LLM_PROVIDERS:
        raise ValueError(f"Неизвестный провайдер LLM: {provider}")
    if api_base_url is not None:
        settings.llm_api_base_url = api_base_url.strip()
    if api_key:  # пустой/None — оставляем прежний ключ
        settings.llm_api_key = api_key
    if summary_model is not None:
        settings.llm_api_summary_model = summary_model.strip()
    if hints_model is not None:
        settings.llm_api_hints_model = hints_model.strip()
    if provider == "api" and not (settings.llm_api_base_url and settings.llm_api_key):
        raise ValueError(
            "API не настроен: укажите адрес и ключ API в настройках приложения "
            "(или STENOGRAF_LLM_API_BASE_URL / STENOGRAF_LLM_API_KEY в server/.env)."
        )
    if provider == "api" and not api_host_supported(settings.llm_api_base_url):
        raise ValueError(API_HOST_HINT)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    _llm_choice_path().write_text(json.dumps({
        "provider": provider,
        "api_base_url": settings.llm_api_base_url,
        "api_key": settings.llm_api_key,
        "summary_model": settings.llm_api_summary_model,
        "hints_model": settings.llm_api_hints_model,
        "tpm_limits": settings.llm_api_tpm_limits,
    }))
    settings.llm_provider = provider


load_asr_choice()
load_llm_choice()

# mlx-whisper качает модели через huggingface_hub — держим кэш рядом с остальными
# моделями в data/models, а не в ~/.cache
os.environ.setdefault("HF_HUB_CACHE", str(settings.models_dir / "hf"))
