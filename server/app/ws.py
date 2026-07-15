"""Живая сессия встречи поверх WebSocket.

Клиент шлёт бинарные кадры: [1 байт канала (0=mic, 1=system)] + PCM16LE 16 кГц mono,
и JSON-команды (start / stop / hints). Сервер отвечает JSON-событиями:
ready, segment, speaker_new, hint, stopped, error.

Конвейер из обособленных этапов (каждый заменяем независимо):
    микшер каналов → денойз → VAD-сегментация → разрез по смене канала →
    → очередь → ASR → эмбеддинг → диаризация → БД → событие клиенту.

Каналы склеиваются в один поток и распознаются единым проходом; кто говорит —
решает диаризация по всем спикерам (включая «Вы»), опираясь на подсказку
микшера о том, в каком канале голос был громче. Эхо из колонок дублей не даёт:
копия голоса в миксе совпадает по времени с оригиналом.

Очередь с одним потребителем сохраняет порядок реплик и не даёт CPU-задачам
выполняться параллельно (важно для 8 ГБ RAM).
"""
import asyncio
import json
import logging
import wave
from collections import deque
from typing import Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .asr.transcriber import Transcriber
from .audio.denoise import create_denoiser
from .audio.mixer import ChannelMixer
from .audio.vad import SpeechSegment, SpeechSegmenter
from .config import SAMPLE_RATE, Settings
from .db import crud
from .db.database import session_scope
from .diarization.embedder import VoiceEmbedder
from .diarization.registry import MatchResult, SpeakerRegistry
from .llm import prompts
from .llm.ollama_client import OllamaClient, OllamaError

log = logging.getLogger(__name__)

CHANNELS = {0: "mic", 1: "system"}
_STOP = object()  # сигнал потребителю очереди


class LiveSession:
    def __init__(
        self,
        ws: WebSocket,
        cfg: Settings,
        transcriber: Transcriber,
        embedder: VoiceEmbedder,
        registry: SpeakerRegistry,
        ollama: OllamaClient,
        on_meeting_ended,  # callback(meeting_id) — запускает суммаризацию
    ):
        self._ws = ws
        self._cfg = cfg
        self._transcriber = transcriber
        self._embedder = embedder
        self._registry = registry
        self._ollama = ollama
        self._on_meeting_ended = on_meeting_ended

        self._meeting_id: Optional[int] = None
        # Этапы конвейера (создаются на старте встречи)
        self._mixer: Optional[ChannelMixer] = None
        self._denoiser = None
        self._segmenter: Optional[SpeechSegmenter] = None
        self._recorders: dict[str, wave.Wave_write] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._consumer: Optional[asyncio.Task] = None
        self._hints_task: Optional[asyncio.Task] = None
        self._hints_enabled = False
        self._recent: deque[str] = deque(maxlen=60)  # последние реплики для подсказок
        self._new_text_since_hint = False
        # Короткие сегменты без эмбеддинга приписываем последнему говорившему,
        # но только из того же канала: быстрое «да» из звонка сразу после фразы
        # владельца — другой человек. Доминанта → (кто, когда закончил)
        self._last_by_channel: dict[str, tuple[MatchResult, float]] = {}
        # Кто говорил недавно — приор для диаризации (speaker_id → конец реплики)
        self._recent_speakers: dict[int, float] = {}

    # ------------------------------------------------------------------ основной цикл

    async def run(self) -> None:
        await self._ws.accept()
        try:
            while True:
                message = await self._ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes"):
                    self._on_audio(message["bytes"])
                elif message.get("text"):
                    command = json.loads(message["text"])
                    if command.get("type") == "stop":
                        await self._finalize()
                        break
                    await self._on_command(command)
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("Ошибка live-сессии")
        finally:
            await self._finalize()  # повторный вызов безопасен

    async def _on_command(self, command: dict) -> None:
        kind = command.get("type")
        if kind == "start":
            await self._start(command)
        elif kind == "hints":
            self._hints_enabled = bool(command.get("enabled"))
        else:
            await self._send({"type": "error", "message": f"Неизвестная команда: {kind}"})

    async def _start(self, command: dict) -> None:
        if self._meeting_id is not None:
            await self._send({"type": "error", "message": "Встреча уже идёт"})
            return
        with session_scope() as db:
            meeting = crud.create_meeting(
                db,
                title=command.get("title", "Встреча"),
                record_audio=bool(command.get("record_audio")),
            )
            self._meeting_id = meeting.id
            if meeting.record_audio:
                directory = self._cfg.recordings_dir / f"meeting_{meeting.id}"
                directory.mkdir(parents=True, exist_ok=True)
                meeting.audio_dir = str(directory)
                for channel in ("mic", "system"):
                    writer = wave.open(str(directory / f"{channel}.wav"), "wb")
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(SAMPLE_RATE)
                    self._recorders[channel] = writer
            title = meeting.title
        self._mixer = ChannelMixer(self._cfg)
        self._denoiser = create_denoiser(self._cfg)
        self._segmenter = SpeechSegmenter(self._cfg)
        self._hints_enabled = bool(command.get("hints"))
        self._consumer = asyncio.create_task(self._consume())
        self._hints_task = asyncio.create_task(self._hints_loop())
        log.info("Встреча #%d «%s» началась", self._meeting_id, title)
        await self._send({"type": "ready", "meeting_id": self._meeting_id, "title": title})

    # ------------------------------------------------------------------ аудио

    def _on_audio(self, frame: bytes) -> None:
        if self._meeting_id is None or self._segmenter is None or len(frame) < 2:
            return
        channel = CHANNELS.get(frame[0])
        if channel is None:
            return
        pcm = frame[1:]
        recorder = self._recorders.get(channel)
        if recorder is not None:
            recorder.writeframes(pcm)  # в запись идут сырые каналы, до микса
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        # Микшер и VAD лёгкие (~1 мс на кадр 100 мс) — работаем прямо в event loop
        for mixed in self._mixer.feed(channel, audio):
            self._enqueue(self._meeting_id, self._segmenter.feed(self._denoiser.process(mixed)))

    def _enqueue(self, meeting_id: int, segments: list[SpeechSegment]) -> None:
        for segment in segments:
            for part in self._split_by_dominance(segment):
                self._queue.put_nowait((meeting_id, part))

    def _split_by_dominance(self, segment: SpeechSegment) -> list[SpeechSegment]:
        """Диалог без паузы: VAD не видит тишины и склеивает реплики двух людей
        в один сегмент. Но если голос перескочил между каналами (владелец ↔
        звонок), говорящий сменился — режем сегмент в точках смены доминанты."""
        spans = self._mixer.dominance_spans(
            segment.start_s, segment.end_s,
            self._cfg.segment_split_window_ms / 1000,
            self._cfg.segment_split_min_run_ms / 1000,
        )
        if len(spans) < 2:
            return [segment]
        parts = []
        for span_start, span_end in spans:
            lo = int((span_start - segment.start_s) * SAMPLE_RATE)
            hi = min(int((span_end - segment.start_s) * SAMPLE_RATE), len(segment.audio))
            if hi > lo:
                parts.append(SpeechSegment(segment.audio[lo:hi], span_start, span_end))
        log.info("Сегмент %.1f–%.1f с разрезан по смене канала на %d части",
                 segment.start_s, segment.end_s, len(parts))
        return parts or [segment]

    # ------------------------------------------------------------------ обработка сегментов

    async def _consume(self) -> None:
        while True:
            item = await self._queue.get()
            if item is _STOP:
                return
            meeting_id, segment = item
            try:
                await self._process_segment(meeting_id, segment)
            except Exception:
                log.exception("Ошибка обработки сегмента")

    async def _process_segment(self, meeting_id: int, segment: SpeechSegment) -> None:
        text = await self._transcriber.transcribe(segment.audio)
        if not text:
            return

        # Подсказка микшера: в каком канале голос сегмента был громче
        dominance = self._mixer.dominance(segment.start_s, segment.end_s)

        with session_scope() as db:
            match = await self._match_speaker(db, segment, dominance)
            self._last_by_channel[dominance] = (match, segment.end_s)
            self._recent_speakers[match.speaker_id] = segment.end_s
            self._registry.maybe_save_sample(db, match.speaker_id, segment.audio)
            row = crud.add_segment(
                db, meeting_id, match.speaker_id, dominance,
                segment.start_s, segment.end_s, text, match.similarity,
            )
            segment_id = row.id

        self._recent.append(f"{match.name}: {text}")
        self._new_text_since_hint = True

        if match.is_new:
            await self._send({
                "type": "speaker_new",
                "speaker": {"id": match.speaker_id, "name": match.name},
            })
        await self._send({
            "type": "segment",
            "segment": {
                "id": segment_id,
                "meeting_id": meeting_id,
                "channel": dominance,
                "start_s": round(segment.start_s, 2),
                "end_s": round(segment.end_s, 2),
                "text": text,
                "similarity": match.similarity,
                "speaker": {
                    "id": match.speaker_id,
                    "name": match.name,
                    "is_self": match.is_self,
                },
            },
        })

    async def _match_speaker(self, db, segment: SpeechSegment, dominance: str) -> MatchResult:
        # duration_s включает паддинг VAD по краям — вычитаем его, чтобы порог
        # сравнивался с длительностью самой речи (иначе правило не срабатывает)
        speech_s = segment.duration_s - 2 * self._cfg.vad_pad_ms / 1000
        if speech_s < self._cfg.speaker_min_embed_s:
            donor = self._short_segment_donor(dominance, segment.start_s)
            if donor is not None:
                return MatchResult(donor.speaker_id, donor.name, donor.is_self, None, False)
        embedding = await asyncio.to_thread(self._embedder.embed, segment.audio)
        recent = frozenset(
            speaker_id for speaker_id, end_s in self._recent_speakers.items()
            if segment.start_s - end_s <= self._cfg.speaker_recent_window_s
        )
        return self._registry.match_all(
            db, embedding, mic_dominant=dominance == "mic", recent_ids=recent
        )

    def _short_segment_donor(self, dominance: str, start_s: float) -> Optional[MatchResult]:
        """Кому отдать сегмент, слишком короткий для эмбеддинга: последнему
        говорившему из того же канала не дольше 4 секунд назад
        ('mixed' совместим с обоими каналами)."""
        fresh = [
            (end_s, match)
            for channel, (match, end_s) in self._last_by_channel.items()
            if start_s - end_s < 4.0
            and (channel == dominance or "mixed" in (channel, dominance))
        ]
        if not fresh:
            return None
        return max(fresh, key=lambda item: item[0])[1]

    # ------------------------------------------------------------------ подсказки (демо)

    async def _hints_loop(self) -> None:
        while self._meeting_id is not None:
            await asyncio.sleep(self._cfg.hints_interval_s)
            if not self._hints_enabled or not self._new_text_since_hint:
                continue
            window = "\n".join(self._recent)[-self._cfg.hints_window_chars:]
            if len(window) < 80:  # слишком мало контекста — рано подсказывать
                continue
            self._new_text_since_hint = False
            try:
                hint = await self._ollama.generate(
                    self._cfg.hints_model,
                    prompts.HINTS_TEMPLATE.format(transcript=window),
                    system=prompts.HINTS_SYSTEM,
                    temperature=0.5,
                )
                if hint:
                    await self._send({"type": "hint", "text": hint})
            except OllamaError as exc:
                self._hints_enabled = False
                await self._send({"type": "hint_error", "message": str(exc)})

    # ------------------------------------------------------------------ завершение

    async def _finalize(self) -> None:
        if self._meeting_id is None:
            return
        meeting_id, self._meeting_id = self._meeting_id, None

        try:
            if self._hints_task:
                self._hints_task.cancel()
            # Дожимаем недоговорённые фразы и ждём, пока ASR обработает очередь
            if self._segmenter is not None:
                for mixed in self._mixer.flush():
                    self._enqueue(meeting_id, self._segmenter.feed(self._denoiser.process(mixed)))
                self._enqueue(meeting_id, self._segmenter.flush())
            self._queue.put_nowait(_STOP)
            if self._consumer:
                try:
                    await asyncio.wait_for(self._consumer, timeout=120)
                except asyncio.TimeoutError:
                    self._consumer.cancel()

            for recorder in self._recorders.values():
                recorder.close()
            self._recorders.clear()
        except Exception:
            log.exception("Ошибка при финализации встречи #%d", meeting_id)
        finally:
            # ВАЖНО: встреча завершается в БД даже если consumer упал
            with session_scope() as db:
                crud.end_meeting(db, meeting_id)
            log.info("Встреча #%d завершена", meeting_id)
            await self._send({"type": "stopped", "meeting_id": meeting_id})
            self._on_meeting_ended(meeting_id)

    async def _send(self, payload: dict) -> None:
        try:
            await self._ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass  # клиент мог отключиться — события уходят в БД в любом случае
