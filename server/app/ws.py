"""Живая сессия встречи поверх WebSocket.

Клиент шлёт бинарные кадры: [1 байт канала (0=mic, 1=system)] + PCM16LE 16 кГц mono,
и JSON-команды (auth / start / stop). Сервер отвечает JSON-событиями:
ready, segment, speaker_new, stopped, error.

Подсказок и вопросов здесь больше нет: их ведёт приложение, у которого свой
адрес модели и свой ключ. Сессия осталась конвейером звука.

Если на сервере заведены люди, первым кадром обязан прийти auth с токеном:
заголовок сюда не поставить — браузерный WebSocket их задавать не умеет, а в
адресе токен оказался бы в журналах сервера.

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
from collections import Counter
from dataclasses import replace
from typing import Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from . import auth
from .asr.transcriber import Transcriber
from .audio.denoise import create_denoiser
from .audio.mixer import ChannelMixer
from .audio.vad import SpeechSegment, SpeechSegmenter
from .config import SAMPLE_RATE, Settings
from .db import crud
from .db.database import session_scope
from .diarization.embedder import VoiceEmbedder
from .diarization.registry import MatchResult, SpeakerRegistry
from .modes import DEFAULT_MODE, normalize_mode

log = logging.getLogger(__name__)

CHANNELS = {0: "mic", 1: "system"}
_STOP = object()  # сигнал потребителю очереди


_LIVE_SESSIONS: set["LiveSession"] = set()


def notify_speakers_merged(source_id: int, target_id: int, name: str,
                           was_named: list[str], owner_id: Optional[int] = None) -> None:
    """Сообщает идущим встречам ЭТОГО человека, что двух спикеров объединили.

    Слияние приходит по REST (страница «Спикеры» открыта окном поверх встречи),
    а живая сессия держит id спикеров в памяти и про удаление профиля сама не
    узнает. Без этого следующая короткая реплика уехала бы на удалённого:
    внешние ключи в SQLite выключены, поэтому она не упала бы с ошибкой, а молча
    записалась ссылкой в никуда — и не нашлась бы потом ни у одного участника.

    Чужие сессии пропускаем: окно разговора правится по ИМЕНИ спикера, и
    объединение своего «Ивана» переименовывало бы строки в чужой идущей встрече,
    где просто есть однофамилец. Эти строки уходят в промпт подсказок.
    """
    for session in list(_LIVE_SESSIONS):
        if session._user_id != owner_id:
            continue
        session.on_speakers_merged(source_id, target_id, name, was_named)


class LiveSession:
    def __init__(
        self,
        ws: WebSocket,
        cfg: Settings,
        transcriber: Transcriber,
        embedder: VoiceEmbedder,
        registry: SpeakerRegistry,
    ):
        self._ws = ws
        self._cfg = cfg
        self._transcriber = transcriber
        self._embedder = embedder
        self._registry = registry

        self._meeting_id: Optional[int] = None
        # Этапы конвейера (создаются на старте встречи)
        self._mixer: Optional[ChannelMixer] = None
        self._denoiser = None
        self._segmenter: Optional[SpeechSegmenter] = None
        self._recorders: dict[str, wave.Wave_write] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._consumer: Optional[asyncio.Task] = None
        self._summarize = True  # составлять ли резюме по завершении (выбор при старте)
        self._meeting_title = ""          # уходит в промпт: модель должна видеть тему
        self._mode = DEFAULT_MODE  # тип встречи (планёрка/собеседование/…)
        # speaker_id → число реплик, для промпта. Именно id, а не имя: имя меняют
        # прямо во время встречи, и счётчик по строке разъезжался бы на двух
        # участников — «Спикер 3 (12 реплик)» и «Иван (4 реплики)» вместо одного.
        self._participants: Counter[int] = Counter()
        # Историю держим большой всегда: окно режется срезом по бюджету провайдера,
        # маленькая дека обнулила бы большое API-окно.
        # Короткие сегменты без эмбеддинга приписываем последнему говорившему,
        # но только из того же канала: быстрое «да» из звонка сразу после фразы
        # владельца — другой человек. Доминанта → (кто, когда закончил)
        self._last_by_channel: dict[str, tuple[MatchResult, float]] = {}
        # Кто говорил недавно — приор для диаризации (speaker_id → конец реплики)
        self._recent_speakers: dict[int, float] = {}
        # Чья это сессия. Заполняется при проверке токена; None — сервер личный,
        # людей на нём не заведено. Понадобится, когда у данных появится владелец.
        self._user_id: Optional[int] = None

    # ------------------------------------------------------------------ основной цикл

    # Сколько ждём кадр auth. Без потолка молчащее соединение висит вечно, и
    # достаточно открыть их пачку, чтобы занять сервер, не зная токена.
    AUTH_TIMEOUT_S = 10.0

    async def _authenticate(self) -> bool:
        """Первый кадр — auth с токеном. Возвращает False, если пускать нельзя.

        Пока людей на сервере нет, соединение принимается как раньше: сервер
        личный, спрашивать не у кого (см. app/auth.py).
        """
        with session_scope() as db:
            if not auth.auth_required(db):
                return True
        try:
            message = await asyncio.wait_for(self._ws.receive(), self.AUTH_TIMEOUT_S)
        except (asyncio.TimeoutError, WebSocketDisconnect):
            await self._close_unauthorized("Токен не пришёл")
            return False
        if message["type"] == "websocket.disconnect":
            return False

        токен = None
        if message.get("text"):
            try:
                команда = json.loads(message["text"])
            except json.JSONDecodeError:
                команда = {}
            if команда.get("type") == "auth":
                токен = str(команда.get("token") or "")
        with session_scope() as db:
            пользователь = auth.user_by_token(db, токен)
            if пользователь is None:
                await self._close_unauthorized("Нужен токен доступа")
                return False
            self._user_id = пользователь.id
        return True

    async def _close_unauthorized(self, причина: str) -> None:
        """Сначала событие, потом закрытие: код 1008 клиент видит числом, а
        человеку нужно сказать словами, что делать."""
        try:
            await self._send({"type": "error", "message": f"{причина}. Настройки → Токен доступа."})
            await self._ws.close(code=1008, reason=причина)
        except Exception:  # соединение уже могло отвалиться — это не ошибка
            pass

    async def run(self) -> None:
        await self._ws.accept()
        if not await self._authenticate():
            return
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
        elif kind == "auth":
            pass  # уже проверен до начала сессии; повтор игнорируем
        else:
            await self._send({"type": "error", "message": f"Неизвестная команда: {kind}"})

    async def _start(self, command: dict) -> None:
        if self._meeting_id is not None:
            await self._send({"type": "error", "message": "Встреча уже идёт"})
            return
        # Старый клиент поля не шлёт — normalize_mode вернёт режим по умолчанию
        self._mode = normalize_mode(command.get("meeting_mode"))
        _LIVE_SESSIONS.add(self)  # чтобы узнать о слиянии спикеров по ходу встречи
        with session_scope() as db:
            meeting = crud.create_meeting(
                db,
                title=command.get("title", "Встреча"),
                record_audio=bool(command.get("record_audio")),
                meeting_mode=self._mode,
                owner_id=self._user_id,  # чья это встреча; None — сервер личный
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
        self._meeting_title = title  # уходит в промпт подсказок — модель видит тему
        self._mixer = ChannelMixer(self._cfg)
        self._denoiser = create_denoiser(self._cfg)
        self._segmenter = SpeechSegmenter(self._cfg)
        self._summarize = bool(command.get("summarize", True))
        self._consumer = asyncio.create_task(self._consume())
        log.info("Встреча #%d «%s» началась (%s)", self._meeting_id, title, self._mode)
        await self._send({
            "type": "ready", "meeting_id": self._meeting_id,
            "title": title, "meeting_mode": self._mode,
        })

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
            if match is not None:
                # Неопознанный обрывок донором не становится: мы не знаем, чей
                # он, и приписывать по нему следующие короткие реплики нельзя.
                self._last_by_channel[dominance] = (match, segment.end_s)
                self._recent_speakers[match.speaker_id] = segment.end_s
            row = crud.add_segment(
                db, meeting_id, match.speaker_id if match else None, dominance,
                segment.start_s, segment.end_s, text,
                match.similarity if match else None,
            )
            segment_id = row.id

        # Ничью реплику в участники не записываем: счётчик уходит в промпт, и
        # «Неизвестный (7 реплик)» модель приняла бы за ещё одного человека.
        if match is not None:
            self._participants[match.speaker_id] += 1

        if match is not None and match.is_new:
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
                "similarity": match.similarity if match else None,
                # null — реплика ничья: клиент покажет её как «Неизвестный»
                "speaker": {
                    "id": match.speaker_id,
                    "name": match.name,
                    "is_self": match.is_self,
                } if match else None,
            },
        })

    async def _match_speaker(
        self, db, segment: SpeechSegment, dominance: str
    ) -> Optional[MatchResult]:
        """Кто это сказал. None — опознать не по чему, реплика остаётся ничьей."""
        # duration_s включает паддинг VAD по краям — вычитаем его, чтобы порог
        # сравнивался с длительностью самой речи (иначе правило не срабатывает)
        speech_s = segment.duration_s - 2 * self._cfg.vad_pad_ms / 1000
        if speech_s < self._cfg.speaker_min_embed_s:
            donor = self._short_segment_donor(dominance, segment.start_s)
            if donor is not None:
                return MatchResult(donor.speaker_id, donor.name, donor.is_self, None, False)
            # Донора нет: это первая реплика встречи или прошло больше четырёх
            # секунд. Эмбеддер на таком обрывке считает не голос, а что придётся,
            # и заводит нового «Спикера N» на каждое «ага» — ровно то, на что
            # жаловался куратор (замечание №13). Оставляем реплику без имени:
            # «Неизвестный» честнее выдуманного участника, и в базу голосов
            # такой обрывок не попадает, потому что эмбеддер даже не зовётся.
            return None
        embedding = await asyncio.to_thread(self._embedder.embed, segment.audio)
        recent = frozenset(
            speaker_id for speaker_id, end_s in self._recent_speakers.items()
            if segment.start_s - end_s <= self._cfg.speaker_recent_window_s
        )
        return self._registry.match_all(
            db, embedding, mic_dominant=dominance == "mic", recent_ids=recent,
            audio=segment.audio, owner_id=self._user_id,
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

    def on_speakers_merged(self, source_id: int, target_id: int, name: str,
                           was_named: list[str]) -> None:
        """Двух спикеров объединили посреди встречи — приводим память в порядок.

        Профиля source_id в базе больше нет, а сессия помнит его в трёх местах:
        как донора для коротких реплик (иначе следующее «ага» уедет на удалённого),
        как недавно говорившего и в счётчике участников.

        Окно разговора отсюда ушло вместе с подсказками, поэтому переименование
        строк «Имя: текст» больше не нужно: этим занимается приложение, у
        которого теперь своя лента. Аргумент was_named остался — его шлёт REST,
        и ломать протокол уведомления ради одного неиспользуемого поля незачем.
        """
        if source_id == target_id:
            return
        moved = self._participants.pop(source_id, 0)
        if moved:
            self._participants[target_id] += moved
        self._recent_speakers.pop(source_id, None)
        for channel, (match, ended_at) in list(self._last_by_channel.items()):
            if match.speaker_id == source_id:
                self._last_by_channel[channel] = (
                    replace(match, speaker_id=target_id, name=name), ended_at,
                )

    @staticmethod
    def _mmss(seconds: float) -> str:
        return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"

    # ------------------------------------------------------------------ завершение

    async def _finalize(self) -> None:
        _LIVE_SESSIONS.discard(self)
        if self._meeting_id is None:
            return
        meeting_id, self._meeting_id = self._meeting_id, None

        try:
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
                crud.end_meeting(
                    db, meeting_id,
                    status="summarizing" if self._summarize else "done",
                )
            log.info("Встреча #%d завершена", meeting_id)
            await self._send({"type": "stopped", "meeting_id": meeting_id})
            if self._summarize:
                self._on_meeting_ended(meeting_id)

    async def _send(self, payload: dict) -> None:
        try:
            await self._ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass  # клиент мог отключиться — события уходят в БД в любом случае
