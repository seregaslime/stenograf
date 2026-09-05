"""Стенограф — сервер транскрипции встреч.

Запуск для разработки:  uvicorn app.main:app --host 0.0.0.0 --port 8765
Все данные (БД, модели, образцы голосов, записи) лежат в server/data/.
"""
import logging
import shutil
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from . import auth, search
from .asr.transcriber import GIGAAM_AVAILABLE, MLX_AVAILABLE, Transcriber
from .config import (
    ASR_ENGINES,
    ASR_MODELS,
    cors_origin_list,
    save_asr_choice,
    settings,
)
from .db import crud
from .db.database import init_db, session_scope
from .db.models import Meeting, VoicePrint
from .diarization.embedder import VoiceEmbedder
from .diarization.registry import SpeakerRegistry
from .transcript import build_transcript
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


def _версия() -> str:
    """Версия для API: полный git-sha режем до семи знаков.

    Сорок шестнадцатеричных символов в строке состояния читать невозможно, а
    семи хватает, чтобы найти коммит. Всё, что не похоже на sha (например
    «dev» при локальном запуске), отдаём как есть.
    """
    v = settings.version.strip()
    return v[:7] if len(v) == 40 and all(c in "0123456789abcdef" for c in v.lower()) else v


app = FastAPI(title="Стенограф API", version=_версия(), lifespan=lifespan)

# Что отвечает без токена. Только состояние сервера — по нему видно, жив ли он,
# ещё до того как человек ввёл токен, и это единственная причина исключения.
# Данных встреч здоровье не отдаёт (см. health): без токена оно урезано.
ОТКРЫТЫЕ_ПУТИ = ("/api/health",)


@app.middleware("http")
async def проверить_доступ(request: Request, call_next):
    """Токен спрашиваем в одном месте, а не зависимостью на каждом эндпоинте.

    Причина в том, чем ошибиться дороже: забытая зависимость у нового эндпоинта
    открывает его молча, и заметить это можно только специально глядя. Здесь
    новый путь закрыт по умолчанию, а открывать его надо руками — ошибка
    становится видимой.
    """
    # WebSocket сюда не попадает: у него нет заголовков (ограничение браузерного
    # API), токен придёт первым кадром — это следующий коммит серии.
    # Закрыто ВСЁ, кроме явно открытого, а не только /api: под /api не попадают
    # автодокументация FastAPI (/docs, /redoc, /openapi.json), и по ней сервер,
    # который только что закрыли токеном, выдал бы посторонним полную карту API.
    # Слэш на конце срезаем: путь «/api/health/» — это тот же health, и отвечать
    # на него отказом значит показывать мониторингу упавший сервер вместо живого.
    путь = request.url.path.rstrip("/") or "/"
    if request.method == "OPTIONS" or путь in ОТКРЫТЫЕ_ПУТИ:
        return await call_next(request)

    with session_scope() as db:
        if auth.auth_required(db):
            user = auth.user_by_token(
                db, auth.token_from_header(request.headers.get("authorization"))
            )
            if user is None:
                return JSONResponse(
                    {"detail": "Нужен токен доступа. Настройки → Токен доступа."},
                    status_code=401,
                )
            # Кладём id и имя, а не объект: сессия закроется на выходе из блока,
            # и отсоединённый объект развалится при первом же обращении к полю.
            request.state.user_id = user.id
            request.state.user_name = user.name
    return await call_next(request)


# Добавляется ПОСЛЕ проверки доступа, поэтому оказывается снаружи неё: иначе
# ответ 401 уходил бы без заголовков CORS, и браузер показывал бы вместо
# внятного «нужен токен» невнятную сетевую ошибку.
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origin_list(),
    # Локальные адреса — выражением, а не списком: vite берёт свободный порт,
    # если 5173 занят, и перечислить их заранее невозможно. Проверено живьём —
    # со списком браузерная превью на порту 62953 получала отказ CORS.
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


def владелец(request: Request) -> int | None:
    """Кто спрашивает. None — сервер личный, людей на нём не заводили, и
    фильтровать данные не по чему (проверку ставит middleware выше)."""
    return getattr(request.state, "user_id", None)


# ---------------------------------------------------------------- здоровье

@app.get("/api/health")
def health(request: Request):
    """Открыт без токена намеренно: иначе не понять, жив ли сервер, до того как
    человек ввёл токен. Поэтому чужому отдаём только «жив» — какие модели
    настроены и какой провайдер выбран, посторонним знать незачем."""
    with session_scope() as db:
        свой = not auth.auth_required(db) or auth.user_by_token(
            db, auth.token_from_header(request.headers.get("authorization"))
        ) is not None
    if not свой:
        # Поля не выбрасываем молча: клиент читает health.asr.model, и ответ без
        # asr ронял бы ему экран настроек. Отдаём признак, по которому видно, что
        # подробностей не будет, и клиент показывает «нужен токен».
        return {"status": "ok", "version": app.version, "authorized": False}
    return {
        "status": "ok",
        "version": app.version,
        "authorized": True,
        "asr": {
            "engine": transcriber.engine,
            "model": transcriber.model_name,
            "loaded": transcriber.loaded,
            # Куратор жаловался на тормоза, потому что всё считалось процессором
            # при живой видеокарте. Теперь это видно, не заглядывая в журнал.
            "device": transcriber.device,
        },
        "diarization": {"loaded": embedder.loaded, "device": embedder.device},
        # Про модели языка сервер больше ничего не знает: их адрес, ключ и выбор
        # живут в приложении. Здесь остались только те модели, которые сервер
        # действительно держит у себя, — распознавание и голоса.
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


# ---------------------------------------------------------------- встречи

@app.get("/api/meetings")
def get_meetings(request: Request):
    with session_scope() as db:
        return crud.list_meetings(db, владелец(request))


@app.get("/api/meetings/{meeting_id}")
def get_meeting(meeting_id: int, request: Request):
    with session_scope() as db:
        meeting = crud.meeting_for_owner(db, meeting_id, владелец(request))
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
            # Прогресс длинного протокола показывает приложение: считает его оно.
            "summary_progress": None,
            "segments": [crud.segment_to_dict(s) for s in segments],
        }


@app.delete("/api/meetings/{meeting_id}")
async def delete_meeting(meeting_id: int, request: Request):
    with session_scope() as db:
        meeting = crud.meeting_for_owner(db, meeting_id, владелец(request))
        if meeting is None:
            raise HTTPException(404, "Встреча не найдена")
    with session_scope() as db:
        meeting = db.get(Meeting, meeting_id)
        if meeting is None:
            # Между проверкой прав и удалением встречу успели удалить (двойной
            # клик, два открытых клиента). Это не ошибка сервера: результат тот
            # же, которого просили.
            return {"deleted": meeting_id}
        if meeting.status == "live":
            # Встреча могла зависнуть после обрыва клиента — принудительно завершаем
            meeting.status = "done"
            meeting.ended_at = datetime.now(timezone.utc)
        if meeting.audio_dir:
            shutil.rmtree(meeting.audio_dir, ignore_errors=True)
        db.delete(meeting)
    return {"deleted": meeting_id}


class SummaryBody(BaseModel):
    """Готовый протокол от клиента: он теперь считает его сам.

    Ошибку принимаем тем же эндпоинтом, а не молчанием: если у человека не
    ответила модель, встреча должна показывать причину, а не вечное
    «составляется» — раньше это писал сервер, потому что считал он же.
    """
    text: str | None = None
    error: str | None = None
    model: str | None = None


@app.post("/api/meetings/{meeting_id}/summary")
def save_summary(meeting_id: int, body: SummaryBody, request: Request):
    """Принимает протокол, составленный клиентом.

    Проверка владельца здесь не формальность: без неё любой, у кого есть токен,
    подписал бы чужой встрече любой текст — а протокол читают те, кого на
    встрече не было, и проверить его они не смогут.
    """
    with session_scope() as db:
        meeting = crud.meeting_for_owner(db, meeting_id, владелец(request))
        if meeting is None:
            raise HTTPException(404, "Встреча не найдена")
        if meeting.status == "live":
            raise HTTPException(409, "Встреча ещё идёт")
        текст = (body.text or "").strip()
        if not текст and not body.error:
            raise HTTPException(400, "Нужен текст протокола или причина неудачи")
        # Успех затирает прошлую ошибку, неудача не затирает прошлый протокол:
        # неудачная попытка пересоздать не должна стирать то, что уже есть.
        if текст:
            meeting.summary = текст
            meeting.summary_model = (body.model or "").strip() or None
            meeting.summary_error = None
        else:
            meeting.summary_error = body.error
        meeting.status = "done"
        return {"status": meeting.status, "has_summary": bool(meeting.summary)}


@app.get("/api/meetings/{meeting_id}/export")
def export_meeting(meeting_id: int, request: Request, fmt: str = "md"):
    with session_scope() as db:
        meeting = crud.meeting_for_owner(db, meeting_id, владелец(request))
        if meeting is None:
            raise HTTPException(404, "Встреча не найдена")
        segments = crud.meeting_segments(db, meeting_id)
        transcript, participants = build_transcript(segments)
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


# Потолок на число цитат в одном ответе поиска.
SEARCH_LIMIT_MAX = 20


class IndexChunk(BaseModel):
    """Кусок разговора с уже посчитанным вектором."""
    first_segment_id: int
    last_segment_id: int
    start_s: float
    text: str
    vector: list[float]


class IndexBody(BaseModel):
    model: str
    meeting_id: int
    chunks: list[IndexChunk]


class QueryBody(BaseModel):
    model: str
    vector: list[float]
    limit: int | None = Field(None, ge=1, le=SEARCH_LIMIT_MAX)


@app.get("/api/search/pending")
def search_pending(request: Request, model: str):
    """Что осталось проиндексировать ЭТОЙ моделью: встречи и куски разговора.

    Имя модели обязательно и приходит от приложения: векторы считает оно, у
    каждого своя модель, и сервер про этот выбор больше ничего не знает.
    Нарезка осталась здесь — она про содержимое встречи, а не про модель.
    """
    with session_scope() as db:
        return {"meetings": search.pending_chunks(db, settings, model, владелец(request))}


@app.post("/api/search/index")
def search_index(body: IndexBody, request: Request):
    """Принимает посчитанные векторы. Чужую встречу проиндексировать нельзя."""
    with session_scope() as db:
        meeting = crud.meeting_for_owner(db, body.meeting_id, владелец(request))
        if meeting is None:
            raise HTTPException(404, "Встреча не найдена")
        сохранено = search.store_vectors(
            db, body.model, meeting, [к.model_dump() for к in body.chunks]
        )
    return {"meeting_id": body.meeting_id, "chunks": сохранено}


@app.post("/api/search/query")
def search_query(body: QueryBody, request: Request):
    """Поиск по уже посчитанному вектору вопроса.

    Сравнение векторов модели не требует, поэтому сервер ищет сам: по сети едет
    вопрос в килобайтах, а не вся матрица в мегабайтах.
    """
    with session_scope() as db:
        return {"results": search.search_by_vector(
            db, body.model, body.vector,
            body.limit or settings.search_top_k, владелец(request),
        )}


@app.get("/api/speakers")
def get_speakers(request: Request):
    with session_scope() as db:
        return crud.list_speakers(db, владелец(request))


@app.patch("/api/speakers/{speaker_id}")
def patch_speaker(speaker_id: int, body: RenameBody, request: Request):
    with session_scope() as db:
        if crud.speaker_for_owner(db, speaker_id, владелец(request)) is None:
            raise HTTPException(404, "Спикер не найден")
        # Переименование остаётся в crud: правило «пустое имя не затирает
        # прежнее» должно жить в одном месте, а не повторяться в эндпоинте.
        speaker = crud.rename_speaker(db, speaker_id, body.name)
        return {"id": speaker.id, "name": speaker.name}


@app.delete("/api/speakers/{speaker_id}")
def delete_speaker(speaker_id: int, request: Request):
    кто = владелец(request)
    with session_scope() as db:
        speaker = crud.speaker_for_owner(db, speaker_id, кто)
        if speaker is None:
            raise HTTPException(404, "Спикер не найден")
        if speaker.is_self:
            raise HTTPException(400, "Нельзя удалить собственный профиль «Вы»")
        unassigned = crud.reassign_segments(db, speaker_id, None)
        db.delete(speaker)  # отпечатки и образцы каскадом
    registry.forget(speaker_id, кто)
    shutil.rmtree(settings.samples_dir / f"spk_{speaker_id}", ignore_errors=True)
    return {"deleted": speaker_id, "unassigned_segments": unassigned}


@app.delete("/api/speakers/{speaker_id}/voiceprints/{print_id}")
def delete_voiceprint(speaker_id: int, print_id: int, request: Request):
    """Удаляет одно «звучание» голоса (отпечаток и его аудио) — например,
    «испорченное» чужим звуком. Профиль и его реплики остаются."""
    with session_scope() as db:
        if not registry.remove_print(db, speaker_id, print_id, владелец(request)):
            raise HTTPException(404, "Отпечаток не найден")
    return {"deleted": print_id, "speaker_id": speaker_id}


@app.get("/api/speakers/{speaker_id}/voiceprints/{print_id}/audio")
def get_voiceprint_audio(speaker_id: int, print_id: int, request: Request):
    """Аудио-фрагмент реплики, из которой родился отпечаток.

    Владельца проверяем обязательно: это запись чужого голоса на диске, и без
    проверки её забирал бы любой, кто подставит номер профиля.
    """
    with session_scope() as db:
        row = db.get(VoicePrint, print_id)
        if crud.speaker_for_owner(db, speaker_id, владелец(request)) is None:
            raise HTTPException(404, "Аудио отпечатка не найдено")
        if (
            row is None or row.speaker_id != speaker_id
            or not row.audio_path or not Path(row.audio_path).exists()
        ):
            raise HTTPException(404, "Аудио отпечатка не найдено")
        return FileResponse(row.audio_path, media_type="audio/wav")


@app.post("/api/speakers/merge")
def merge_speakers(body: MergeBody, request: Request):
    if len(body.speaker_ids) != 2:
        raise HTTPException(400, "Нужно ровно два спикера")
    with session_scope() as db:
        try:
            result = registry.merge(db, body.speaker_ids[0], body.speaker_ids[1],
                                    владелец(request))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    # Только после коммита: идущая встреча по этому сигналу перепишет своё
    # состояние, а профиля-источника к тому моменту уже не должно быть в базе.
    notify_speakers_merged(
        result["source_id"], result["target_id"], result["name"], result["was_named"],
        владелец(request),
    )
    return result


# ---------------------------------------------------------------- live

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    session = LiveSession(ws, settings, transcriber, embedder, registry)
    await session.run()
