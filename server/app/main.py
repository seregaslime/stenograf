"""Стенограф — сервер транскрипции встреч.

Запуск для разработки:  uvicorn app.main:app --host 0.0.0.0 --port 8765
Все данные (БД, модели, образцы голосов, записи) лежат в server/data/.
"""
import asyncio
import logging
import shutil
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from . import search
from .asr.transcriber import GIGAAM_AVAILABLE, MLX_AVAILABLE, Transcriber
from .config import (
    API_HOST_HINT,
    ASR_ENGINES,
    ASR_MODELS,
    LLM_API_DEFAULT_BASE_URL,
    OLLAMA_URL_HINT,
    Settings,
    api_host_supported,
    ollama_url_valid,
    save_asr_choice,
    save_llm_choice,
    save_tpm_limits,
    settings,
)
from .db import crud
from .db.database import init_db, session_scope
from .db.models import Meeting, Speaker, VoicePrint
from .diarization.embedder import VoiceEmbedder
from .diarization.registry import SpeakerRegistry
from .llm.base import LlmError
from .llm.ollama_client import OllamaClient
from .llm.openai_client import OpenAIClient
from .llm.router import LlmRouter
from .llm.summary import build_transcript, generate_summary
from .ws import LiveSession, notify_speakers_merged

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("stenograf")

# Выбранный движок может быть не установлен (gigaam ставится с GitHub,
# mlx — только Apple Silicon) — откатываемся на whisper, который есть всегда
if (settings.asr_engine == "gigaam" and not GIGAAM_AVAILABLE) or (
    settings.asr_engine == "mlx" and not MLX_AVAILABLE
):
    log.warning("Движок '%s' недоступен — используем faster-whisper small", settings.asr_engine)
    settings.asr_engine, settings.asr_model = "faster_whisper", "small"

transcriber = Transcriber(settings)
embedder = VoiceEmbedder(settings)
registry = SpeakerRegistry(settings)
llm = LlmRouter(settings)

# Задачи суммаризации по id встречи — чтобы отменять их при удалении встречи
# (иначе задача удалённой встречи допишет резюме в новую встречу с тем же id)
_summary_tasks: dict[int, asyncio.Task] = {}


# Прогресс длинного резюме по id встречи: «фрагмент 2 из 3». Держим в памяти, а
# не в БД: это состояние живой задачи, оно бессмысленно без неё, и ради него не
# стоит менять схему. Сервер перезапустили — задача умерла, и уборка при старте
# закрывает встречу с внятной ошибкой.
_summary_progress: dict[int, tuple[int, int]] = {}


def _schedule_summary(meeting_id: int) -> None:
    previous = _summary_tasks.pop(meeting_id, None)
    if previous is not None:
        previous.cancel()

    def _progress(step: int, total: int, mid: int = meeting_id) -> None:
        _summary_progress[mid] = (step, total)

    task = asyncio.create_task(generate_summary(llm, meeting_id, on_progress=_progress))
    _summary_tasks[meeting_id] = task

    def _cleanup(done: asyncio.Task, mid: int = meeting_id) -> None:
        _summary_progress.pop(mid, None)
        if _summary_tasks.get(mid) is done:
            del _summary_tasks[mid]

    task.add_done_callback(_cleanup)


def _cancel_summary(meeting_id: int) -> None:
    task = _summary_tasks.pop(meeting_id, None)
    if task is not None:
        task.cancel()


def _warm_models() -> None:
    """Прогрев тяжёлых моделей в фоне, чтобы первая фраза не ждала загрузки."""
    try:
        transcriber.load()
        embedder.load()
    except Exception:
        log.exception("Не удалось прогреть модели")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    for directory in (settings.data_dir, settings.samples_dir,
                      settings.recordings_dir, settings.models_dir):
        directory.mkdir(parents=True, exist_ok=True)
    init_db()
    with session_scope() as db:
        # Чистим зависшие встречи. Два разных способа зависнуть: клиент оборвался
        # и встреча осталась live; сервер убили посреди составления резюме и она
        # осталась summarizing — такую клиент опрашивал бы вечно, потому что
        # довести её до конца больше некому.
        stale = db.scalars(
            select(Meeting).where(Meeting.status.in_(("live", "summarizing")))
        ).all()
        for meeting in stale:
            if meeting.status == "summarizing":
                meeting.summary_error = (
                    "Сервер перезапустился, пока составлялось резюме. "
                    "Нажмите «Пересоздать резюме»."
                )
            else:  # ended_at у live ещё не проставлен, у summarizing — уже
                meeting.ended_at = datetime.now(timezone.utc)
            meeting.status = "done"
            log.info("Зависшая встреча #%d «%s» принудительно завершена", meeting.id, meeting.title)
        registry.load(db)
    if settings.preload_asr:
        threading.Thread(target=_warm_models, daemon=True).start()
    yield


app = FastAPI(title="Стенограф API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # клиент — локальное Electron-приложение
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- здоровье

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": app.version,
        "asr": {
            "engine": transcriber.engine,
            "model": transcriber.model_name,
            "loaded": transcriber.loaded,
            # Куратор жаловался на тормоза, потому что всё считалось процессором
            # при живой видеокарте. Теперь это видно, не заглядывая в журнал.
            "device": transcriber.device,
        },
        "diarization": {"loaded": embedder.loaded, "device": embedder.device},
        "ollama": await llm.local_status(),
        "llm": {
            "provider": llm.provider,
            "api_configured": bool(settings.llm_api_base_url and settings.llm_api_key),
        },
        "summary_model": llm.summary_model_name,
        "hints_model": llm.hints_model_name,
    }


# ---------------------------------------------------------------- ASR

class AsrBody(BaseModel):
    engine: str
    model: str


_ENGINE_AVAILABLE = {
    "faster_whisper": True,
    "mlx": MLX_AVAILABLE,
    "gigaam": GIGAAM_AVAILABLE,
}


def _asr_state() -> dict:
    return {
        "engine": transcriber.engine,
        "model": transcriber.model_name,
        "loaded": transcriber.loaded,
        "loading": transcriber.loading,
        "error": transcriber.load_error,
        "engines": _ENGINE_AVAILABLE,
        "models_by_engine": {engine: list(models) for engine, models in ASR_MODELS.items()},
    }


@app.get("/api/asr")
def get_asr():
    return _asr_state()


@app.post("/api/asr")
async def set_asr(body: AsrBody):
    if body.engine not in ASR_ENGINES or body.model not in ASR_MODELS[body.engine]:
        raise HTTPException(400, "Неизвестный движок или модель")
    if not _ENGINE_AVAILABLE[body.engine]:
        raise HTTPException(400, f"Движок «{body.engine}» не установлен на этом сервере")
    with session_scope() as db:
        live = db.scalars(select(Meeting).where(Meeting.status == "live")).first()
        if live is not None:
            raise HTTPException(409, "Идёт встреча — модель можно сменить после её завершения")
    await transcriber.reconfigure(body.engine, body.model)
    save_asr_choice(body.engine, body.model)
    threading.Thread(target=_warm_models, daemon=True).start()
    return _asr_state()


# ---------------------------------------------------------------- LLM-провайдер

# Потолок на число цитат в одном ответе — общий для обоих эндпоинтов поиска.
SEARCH_LIMIT_MAX = 20


class LlmBody(BaseModel):
    provider: str
    # Для provider="api" — вводятся в настройках приложения. None-поля не меняются;
    # пустой api_key не затирает сохранённый (см. save_llm_choice).
    api_base_url: str | None = None
    api_key: str | None = None
    summary_model: str | None = None
    hints_model: str | None = None
    # Для provider="local" — то же самое для Ollama: адрес и две модели.
    ollama_url: str | None = None
    local_summary_model: str | None = None
    local_hints_model: str | None = None
    # Роль модели для ответов по прошлым встречам: summary | hints
    search_answer_model: str | None = None


class ModelsProbeBody(BaseModel):
    """Проверка ещё не сохранённых кредов: запрос списка моделей у API."""
    api_base_url: str
    api_key: str | None = None


class OllamaProbeBody(BaseModel):
    """То же для Ollama по ещё не сохранённому адресу."""
    ollama_url: str


async def _llm_state() -> dict:
    status = await llm.status()
    return {
        "provider": llm.provider,
        "api_configured": bool(settings.llm_api_base_url and settings.llm_api_key),
        "api_base_url": settings.llm_api_base_url,  # адрес не секрет; ключ не отдаём
        # Токены в минуту по моделям — то ограничение, в которое упираются
        # подсказки. Размер контекста рядом с моделью уже показан, но упираются
        # не в него, и без этой цифры настройки вводят в заблуждение.
        "api_tpm_limits": settings.llm_api_tpm_limits,
        # чем заполнить поле адреса, если ничего не сохранено
        "api_base_url_default": LLM_API_DEFAULT_BASE_URL,
        "reachable": status.get("reachable", False),
        "models": status.get("models", []),
        # только для api: размер контекста у каждой модели и сколько отсеяли
        "models_info": status.get("models_info", []),
        "models_rejected": status.get("models_rejected", 0),
        # модели активного провайдера (для строки статуса)
        "summary_model": llm.summary_model_name,
        "hints_model": llm.hints_model_name,
        # модели API отдельно: форма настроек показывает их и когда активен local
        "api_summary_model": settings.llm_api_summary_model,
        "api_hints_model": settings.llm_api_hints_model,
        # и симметрично — настройки Ollama, чтобы форма показывала их при
        # активном api и не выдавала чужие значения за локальные
        "ollama_url": settings.ollama_url,
        "local_summary_model": settings.summary_model,
        "local_hints_model": settings.hints_model,
        "search_answer_model": settings.search_answer_model,
    }


@app.get("/api/llm")
async def get_llm():
    return await _llm_state()


@app.post("/api/llm")
async def set_llm(body: LlmBody):
    # Провайдера можно менять и во время встречи: подсказки читают выбор на лету,
    # перезагрузка модели (в отличие от ASR) не нужна.
    try:
        save_llm_choice(
            body.provider,
            api_base_url=body.api_base_url,
            api_key=body.api_key,
            summary_model=body.summary_model,
            hints_model=body.hints_model,
            ollama_url=body.ollama_url,
            local_summary_model=body.local_summary_model,
            local_hints_model=body.local_hints_model,
            search_answer_model=body.search_answer_model,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if body.provider == "api":
        await _measure_tpm_limits()
    return await _llm_state()


async def _measure_tpm_limits() -> None:
    """Выясняет лимит токенов в минуту у выбранных моделей и запоминает его.

    Делается здесь, при выборе модели, а не по ходу встречи: иначе первая же
    встреча шла бы с бюджетом наугад и выясняла лимит, упираясь в него. Модели
    для резюме и подсказок могут быть разными, и лимиты у них разные.

    Проба необязательна: не ответила — работаем на запасном значении. Ронять
    из-за неё сохранение настроек нельзя, сеть у пользователя нестабильна.
    """
    client = OpenAIClient(settings)
    measured = {}
    for model in {settings.llm_api_summary_model, settings.llm_api_hints_model}:
        if not model:
            continue
        limit = await client.token_limit(model)
        if limit:
            measured[model] = limit
            log.info("Лимит модели «%s»: %d токенов/мин", model, limit)
    save_tpm_limits(measured)


@app.post("/api/llm/ollama/models")
async def probe_ollama_models(body: OllamaProbeBody):
    """Какие модели скачаны у Ollama по ещё не сохранённому адресу.

    Отдельно от /api/llm/models: там проверка хоста по белому списку и отсев
    моделей по размеру контекста — у Ollama ни того, ни другого не нужно, а
    ветвление внутри одного эндпоинта пришлось бы читать в обе стороны.
    """
    if not ollama_url_valid(body.ollama_url):
        raise HTTPException(400, OLLAMA_URL_HINT)
    проба = Settings(_env_file=None)
    проба.ollama_url = body.ollama_url.strip().rstrip("/")
    return await OllamaClient(проба).status()


@app.post("/api/llm/models")
async def probe_llm_models(body: ModelsProbeBody):
    """Список ПРИГОДНЫХ моделей по введённым (ещё не сохранённым) кредам — для
    выпадающего списка в настройках. Пустой ключ → берём уже сохранённый.

    Негодные модели (не текст→текст, слишком маленький контекст) отсеиваются
    в OpenAIClient.status() по данным самого провайдера."""
    base_url = body.api_base_url.strip()
    if not api_host_supported(base_url):
        raise HTTPException(400, API_HOST_HINT)
    cfg = settings.model_copy(update={
        "llm_api_base_url": base_url,
        "llm_api_key": body.api_key or settings.llm_api_key,
    })
    return await OpenAIClient(cfg).status()


# ---------------------------------------------------------------- встречи

@app.get("/api/meetings")
def get_meetings():
    with session_scope() as db:
        return crud.list_meetings(db)


@app.get("/api/meetings/{meeting_id}")
def get_meeting(meeting_id: int):
    with session_scope() as db:
        meeting = db.get(Meeting, meeting_id)
        if meeting is None:
            raise HTTPException(404, "Встреча не найдена")
        segments = crud.meeting_segments(db, meeting_id)
        return {
            "id": meeting.id,
            "title": meeting.title,
            "status": meeting.status,
            "started_at": meeting.started_at.isoformat() if meeting.started_at else None,
            "ended_at": meeting.ended_at.isoformat() if meeting.ended_at else None,
            "record_audio": meeting.record_audio,
            "meeting_mode": meeting.meeting_mode or "work",
            "summary": meeting.summary,
            "summary_model": meeting.summary_model,
            "summary_error": meeting.summary_error,
            # Прогресс длинного резюме: [шаг, всего] или null. Без него минуты
            # ожидания выглядят как то самое зависание, которое мы чинили.
            "summary_progress": list(_summary_progress[meeting_id])
            if meeting_id in _summary_progress else None,
            "segments": [crud.segment_to_dict(s) for s in segments],
        }


@app.delete("/api/meetings/{meeting_id}")
async def delete_meeting(meeting_id: int):
    # async: отмена задачи резюме должна происходить в том же event loop
    _cancel_summary(meeting_id)
    with session_scope() as db:
        meeting = db.get(Meeting, meeting_id)
        if meeting is None:
            raise HTTPException(404, "Встреча не найдена")
        if meeting.status == "live":
            # Встреча могла зависнуть после обрыва клиента — принудительно завершаем
            meeting.status = "done"
            meeting.ended_at = datetime.now(timezone.utc)
        if meeting.audio_dir:
            shutil.rmtree(meeting.audio_dir, ignore_errors=True)
        db.delete(meeting)
    return {"deleted": meeting_id}


@app.post("/api/meetings/{meeting_id}/summarize")
async def summarize_meeting(meeting_id: int):
    with session_scope() as db:
        meeting = db.get(Meeting, meeting_id)
        if meeting is None:
            raise HTTPException(404, "Встреча не найдена")
        if meeting.status == "live":
            raise HTTPException(409, "Встреча ещё идёт")
        meeting.status = "summarizing"
        meeting.summary_error = None
    _schedule_summary(meeting_id)
    return {"status": "summarizing"}


@app.get("/api/meetings/{meeting_id}/export")
def export_meeting(meeting_id: int, fmt: str = "md"):
    with session_scope() as db:
        meeting = db.get(Meeting, meeting_id)
        if meeting is None:
            raise HTTPException(404, "Встреча не найдена")
        segments = crud.meeting_segments(db, meeting_id)
        # max_chars=0 — в выгрузке нужен полный транскрипт: усечение «головы с
        # хвостом» нужно только чтобы уместить встречу в контекст LLM
        transcript, participants = build_transcript(segments, 0)
        date = meeting.started_at.strftime("%d.%m.%Y %H:%M") if meeting.started_at else ""
        if fmt == "md":
            parts = [f"# {meeting.title}", f"*{date}*", f"**Участники:** {participants}", ""]
            if meeting.summary:
                parts += [meeting.summary, "", "---", ""]
            parts += ["## Транскрипт", "", transcript, ""]
            content, media, ext = "\n".join(parts), "text/markdown", "md"
        else:
            parts = [meeting.title, date, f"Участники: {participants}", ""]
            if meeting.summary:
                parts += [meeting.summary, "", "-" * 40, ""]
            parts += [transcript, ""]
            content, media, ext = "\n".join(parts), "text/plain", "txt"
    filename = f"meeting_{meeting_id}.{ext}"
    return Response(
        content,
        media_type=f"{media}; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------- спикеры

class RenameBody(BaseModel):
    name: str


class MergeBody(BaseModel):
    speaker_ids: list[int]  # ровно два id; сервер сам выбирает целевой профиль


@app.get("/api/search")
async def search_meetings(q: str, limit: int | None = Query(None, ge=1, le=SEARCH_LIMIT_MAX)):
    """Куски прошлых встреч, ближайшие по смыслу к вопросу.

    Индексация ленивая, прямо здесь: встречи могли пройти до появления поиска,
    а модель эмбеддингов — смениться в настройках. Крючок в конвейере обошёлся
    бы дороже и молчал бы про старые встречи.
    """
    try:
        with session_scope() as db:
            посчитано = await search.reindex_missing(db, settings)
            if посчитано:
                log.info("Проиндексировано кусков: %d", посчитано)
            return {"results": await search.search(db, settings, q, limit)}
    except LlmError as exc:
        # Модель эмбеддингов не скачана или Ollama не запущена — это чинится
        # одной командой, и текст ошибки должен эту команду называть.
        raise HTTPException(503, str(exc))


class SearchAnswerBody(BaseModel):
    q: str
    # Границы те же, что у GET /api/search: один и тот же по смыслу параметр
    # не должен вести себя по-разному у двух эндпоинтов одного поиска.
    limit: int | None = Field(None, ge=1, le=SEARCH_LIMIT_MAX)


@app.post("/api/search/answer")
async def answer_from_meetings(body: SearchAnswerBody):
    """Ответ модели по найденным фрагментам прошлых встреч.

    Отдельно от /api/search, а не флагом к нему: поиск отвечает за доли
    секунды, ответ модели — за секунды. Клиент сначала показывает цитаты, и
    человек уже читает их, пока думает модель.
    """
    try:
        with session_scope() as db:
            await search.reindex_missing(db, settings)
            return await search.answer(db, settings, llm, body.q, body.limit)
    except LlmError as exc:
        raise HTTPException(503, str(exc))


@app.get("/api/speakers")
def get_speakers():
    with session_scope() as db:
        return crud.list_speakers(db)


@app.patch("/api/speakers/{speaker_id}")
def patch_speaker(speaker_id: int, body: RenameBody):
    with session_scope() as db:
        speaker = crud.rename_speaker(db, speaker_id, body.name)
        if speaker is None:
            raise HTTPException(404, "Спикер не найден")
        return {"id": speaker.id, "name": speaker.name}


@app.delete("/api/speakers/{speaker_id}")
def delete_speaker(speaker_id: int):
    with session_scope() as db:
        speaker = db.get(Speaker, speaker_id)
        if speaker is None:
            raise HTTPException(404, "Спикер не найден")
        if speaker.is_self:
            raise HTTPException(400, "Нельзя удалить собственный профиль «Вы»")
        unassigned = crud.reassign_segments(db, speaker_id, None)
        db.delete(speaker)  # отпечатки и образцы каскадом
    registry.forget(speaker_id)
    shutil.rmtree(settings.samples_dir / f"spk_{speaker_id}", ignore_errors=True)
    return {"deleted": speaker_id, "unassigned_segments": unassigned}


@app.delete("/api/speakers/{speaker_id}/voiceprints/{print_id}")
def delete_voiceprint(speaker_id: int, print_id: int):
    """Удаляет одно «звучание» голоса (отпечаток и его аудио) — например,
    «испорченное» чужим звуком. Профиль и его реплики остаются."""
    with session_scope() as db:
        if not registry.remove_print(db, speaker_id, print_id):
            raise HTTPException(404, "Отпечаток не найден")
    return {"deleted": print_id, "speaker_id": speaker_id}


@app.get("/api/speakers/{speaker_id}/voiceprints/{print_id}/audio")
def get_voiceprint_audio(speaker_id: int, print_id: int):
    """Аудио-фрагмент реплики, из которой родился отпечаток."""
    with session_scope() as db:
        row = db.get(VoicePrint, print_id)
        if (
            row is None or row.speaker_id != speaker_id
            or not row.audio_path or not Path(row.audio_path).exists()
        ):
            raise HTTPException(404, "Аудио отпечатка не найдено")
        return FileResponse(row.audio_path, media_type="audio/wav")


@app.post("/api/speakers/merge")
def merge_speakers(body: MergeBody):
    if len(body.speaker_ids) != 2:
        raise HTTPException(400, "Нужно ровно два спикера")
    with session_scope() as db:
        try:
            result = registry.merge(db, body.speaker_ids[0], body.speaker_ids[1])
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    # Только после коммита: идущая встреча по этому сигналу перепишет своё
    # состояние, а профиля-источника к тому моменту уже не должно быть в базе.
    notify_speakers_merged(
        result["source_id"], result["target_id"], result["name"], result["was_named"],
    )
    return result


# ---------------------------------------------------------------- live

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    session = LiveSession(
        ws, settings, transcriber, embedder, registry, llm,
        on_meeting_ended=_schedule_summary,
    )
    await session.run()
