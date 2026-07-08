"""Стенограф — сервер транскрипции встреч.

Запуск для разработки:  uvicorn app.main:app --host 0.0.0.0 --port 8765
Все данные (БД, модели, образцы голосов, записи) лежат в server/data/.
"""
import asyncio
import logging
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from .asr.transcriber import Transcriber
from .config import settings
from .db import crud
from .db.database import init_db, session_scope
from .db.models import Meeting, SpeakerSample
from .diarization.embedder import VoiceEmbedder
from .diarization.registry import SpeakerRegistry
from .llm.ollama_client import OllamaClient
from .llm.summary import build_transcript, generate_summary
from .ws import LiveSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("stenograf")

transcriber = Transcriber(settings)
embedder = VoiceEmbedder(settings)
registry = SpeakerRegistry(settings)
ollama = OllamaClient(settings)

# Задачи суммаризации по id встречи — чтобы отменять их при удалении встречи
# (иначе задача удалённой встречи допишет резюме в новую встречу с тем же id)
_summary_tasks: dict[int, asyncio.Task] = {}


def _schedule_summary(meeting_id: int) -> None:
    previous = _summary_tasks.pop(meeting_id, None)
    if previous is not None:
        previous.cancel()
    task = asyncio.create_task(generate_summary(settings, ollama, meeting_id))
    _summary_tasks[meeting_id] = task

    def _cleanup(done: asyncio.Task, mid: int = meeting_id) -> None:
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
        "asr": {"model": settings.asr_model, "loaded": transcriber.loaded},
        "diarization": {"loaded": embedder.loaded},
        "ollama": await ollama.status(),
        "summary_model": settings.summary_model,
        "hints_model": settings.hints_model,
    }


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
            "summary": meeting.summary,
            "summary_model": meeting.summary_model,
            "summary_error": meeting.summary_error,
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
            raise HTTPException(409, "Нельзя удалить идущую встречу")
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
    source_id: int
    target_id: int


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


@app.post("/api/speakers/merge")
def merge_speakers(body: MergeBody):
    with session_scope() as db:
        try:
            result = registry.merge(db, body.source_id, body.target_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    return result


@app.get("/api/samples/{sample_id}")
def get_sample(sample_id: int):
    with session_scope() as db:
        sample = db.get(SpeakerSample, sample_id)
        if sample is None or not Path(sample.path).exists():
            raise HTTPException(404, "Образец не найден")
        return FileResponse(sample.path, media_type="audio/wav")


# ---------------------------------------------------------------- live

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    session = LiveSession(
        ws, settings, transcriber, embedder, registry, ollama,
        on_meeting_ended=_schedule_summary,
    )
    await session.run()
