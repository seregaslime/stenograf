"""Живая сессия встречи поверх WebSocket.

Клиент шлёт бинарные кадры: [1 байт канала (0=mic, 1=system)] + PCM16LE 16 кГц mono,
и JSON-команды (auth / start / stop / hints / hint_now). Сервер отвечает
JSON-событиями: ready, segment, speaker_new, hint, stopped, error.

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
import difflib
import json
import logging
import time
import wave
from collections import Counter, deque
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
from .llm import prompts
from .llm.base import LlmError
from .llm.router import LlmRouter

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


def _log_hint_failure(task: asyncio.Task) -> None:
    """Фоновая подсказка теперь живёт отдельной задачей, и её исключение никто
    не ждёт — без этого неожиданный сбой пропал бы совсем беззвучно."""
    if task.cancelled():
        return  # сняли ради вопроса человека — штатный исход
    if task.exception() is not None:
        log.exception("Фоновая подсказка упала", exc_info=task.exception())


class LiveSession:
    def __init__(
        self,
        ws: WebSocket,
        cfg: Settings,
        transcriber: Transcriber,
        embedder: VoiceEmbedder,
        registry: SpeakerRegistry,
        llm: LlmRouter,
        on_meeting_ended,  # callback(meeting_id) — запускает суммаризацию
    ):
        self._ws = ws
        self._cfg = cfg
        self._transcriber = transcriber
        self._embedder = embedder
        self._registry = registry
        self._llm = llm
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
        self._summarize = True  # составлять ли резюме по завершении (выбор при старте)
        self._meeting_title = ""          # уходит в промпт: модель должна видеть тему
        self._mode = prompts.DEFAULT_MODE  # тип встречи (планёрка/собеседование/…)
        # speaker_id → число реплик, для промпта. Именно id, а не имя: имя меняют
        # прямо во время встречи, и счётчик по строке разъезжался бы на двух
        # участников — «Спикер 3 (12 реплик)» и «Иван (4 реплики)» вместо одного.
        self._participants: Counter[int] = Counter()
        # Историю держим большой всегда: окно режется срезом по бюджету провайдера,
        # маленькая дека обнулила бы большое API-окно.
        self._recent: deque[str] = deque(maxlen=cfg.hints_recent_maxlen)
        # Состояние адаптивных подсказок (см. _hints_loop / _emit_hint)
        self._chars_since_hint = 0        # сколько нового текста с прошлой подсказки
        self._last_hint_at = 0.0          # time.monotonic() последней подсказки
        self._recent_hints: deque[str] = deque(maxlen=cfg.hints_memory)  # против повторов
        self._hint_in_flight = False      # занят LLM (любой запрос) — не запускать вторую
        # Занят тем, что человек запросил САМ (кнопка «подсказать сейчас» или
        # вопрос по выделенным репликам). Отдельно от _hint_in_flight, потому что
        # у этих двух разный приоритет: фоновая подсказка необязательна, а вопрос
        # — нет. Раньше флаг был один, и подсказка, ждущая восстановления минутного
        # лимита (это десятки секунд), отбивала вопрос сообщением про «прошлый
        # вопрос», которого человек не задавал.
        self._explicit_in_flight = False
        # Фоновая подсказка живёт своей задачей — чтобы явный запрос мог её снять.
        self._auto_hint_task: Optional[asyncio.Task] = None
        self._hint_fail_streak = 0        # ошибок LLM подряд
        self._hint_backoff_until = 0.0    # до этого времени не пробовать (бэкофф)
        self._skip_streak = 0             # сколько раз подряд модель промолчала
        self._hint_gap_s = cfg.hints_min_gap_s  # текущая пауза (после SKIP — короче)
        # Граница «модель это уже видела». Счётчик монотонный, а не индекс в
        # _recent: дека переполняется и выбрасывает старое слева, из-за чего
        # сохранённый индекс через 400 реплик указывал бы не туда.
        self._lines_total = 0             # всего реплик за встречу
        self._hinted_at_line = 0          # сколько их было на момент прошлой подсказки
        # Где именно ассистент подсказывал: (номер реплики, текст). Вплетаем в
        # контекст, чтобы модель видела свои ответы и не отвечала повторно.
        self._hint_log: deque[tuple[int, str]] = deque(maxlen=20)
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
            await self._send({"type": "error", "message": f"{причина}. Настройки → Токен сервера."})
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
        elif kind == "hints":
            self._hints_enabled = bool(command.get("enabled"))
        elif kind == "auth":
            pass  # уже проверен до начала сессии; повтор игнорируем
        elif kind == "hint_now":
            await self._emit_hint(force=True)  # «подсказать сейчас» — в обход триггера
        elif kind == "ask":
            await self._answer(
                question=str(command.get("question", "")).strip(),
                segment_ids=command.get("segment_ids") or [],
            )
        else:
            await self._send({"type": "error", "message": f"Неизвестная команда: {kind}"})

    async def _start(self, command: dict) -> None:
        if self._meeting_id is not None:
            await self._send({"type": "error", "message": "Встреча уже идёт"})
            return
        # Старый клиент поля не шлёт — normalize_mode вернёт режим по умолчанию
        self._mode = prompts.normalize_mode(command.get("meeting_mode"))
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
        self._hints_enabled = bool(command.get("hints"))
        self._summarize = bool(command.get("summarize", True))
        self._consumer = asyncio.create_task(self._consume())
        self._hints_task = asyncio.create_task(self._hints_loop())
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

        self._recent.append(f"{match.name if match else 'Неизвестный'}: {text}")
        self._lines_total += 1
        # Ничью реплику в участники не записываем: счётчик уходит в промпт, и
        # «Неизвестный (7 реплик)» модель приняла бы за ещё одного человека.
        if match is not None:
            self._participants[match.speaker_id] += 1
        self._chars_since_hint += len(text)

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

    # ------------------------------------------------------------------ подсказки (демо)

    async def _hints_loop(self) -> None:
        """Подсказку выдаём не по таймеру, а когда накопилось достаточно нового
        разговора и прошёл минимальный интервал (быстрый API это позволяет)."""
        while self._meeting_id is not None:
            await asyncio.sleep(self._cfg.hints_poll_s)
            # Занятость проверяем здесь, а не только внутри _emit_hint: иначе
            # новая задача затёрла бы ссылку на уже считающуюся, и снять ту ради
            # вопроса человека было бы нечем.
            if (self._hints_enabled and not self._hint_in_flight
                    and self._should_hint(time.monotonic())):
                # Отдельной задачей, а не await прямо здесь: иначе снять подсказку
                # ради вопроса человека можно было бы только вместе со всем циклом.
                self._auto_hint_task = asyncio.create_task(self._emit_hint())
                self._auto_hint_task.add_done_callback(_log_hint_failure)

    async def _cancel_auto_hint(self) -> None:
        """Снимает фоновую подсказку ради того, что человек запросил сам.

        Подсказка необязательна: модель и без того вправе промолчать, и пропуск
        ничего не стоит. Вопрос — наоборот, единственное, ради чего человек в
        этот момент смотрит на экран. Ждать её нельзя: она может стоять в
        пейсинге по минутному лимиту почти минуту.
        """
        task, self._auto_hint_task = self._auto_hint_task, None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass  # сняли сами, это штатный исход, а не сбой

    def _should_hint(self, now: float) -> bool:
        """Пора ли подсказывать: не в периоде бэкоффа, накопилось нового текста
        и прошёл минимальный интервал с прошлой подсказки (_hint_gap_s растёт
        при серии SKIP — см. _on_skip)."""
        return (
            now >= self._hint_backoff_until
            and self._chars_since_hint >= self._cfg.hints_min_new_chars
            and now - self._last_hint_at >= self._hint_gap_s
        )

    def _on_skip(self) -> None:
        """Модель промолчала — это не ошибка, а норма.

        Счётчик текста уже сброшен (материал модель посмотрела и признала
        непригодным — повторять после +50 символов бессмысленно и дорого), но
        _recent не трогаем: на следующей попытке модель увидит и старое, и новое.
        Пауза после SKIP короче обычной — разговор в любой момент может дойти до
        важного; серия SKIP подряд растит её линейно, чтобы болтовня ни о чём не
        жгла CPU. Бэкофф по ошибкам здесь намеренно не участвует.
        """
        self._skip_streak += 1
        self._hint_gap_s = min(
            self._cfg.hints_skip_gap_s * self._skip_streak,
            self._cfg.hints_skip_max_gap_s,
        )

    def _split_window(self, budget_chars: int, force: bool) -> tuple[str, str]:
        """Делит разговор на (контекст, новое) по границе прошлой подсказки.

        Возвращает пустой контекст, если границы нет (первая подсказка) или если
        нажата кнопка: там пользователь просит посмотреть на разговор целиком.

        Новое отдаём целиком, контекстом добираем остаток бюджета — реагировать
        надо на свежее, а старое нужно лишь чтобы понимать, о чём речь.
        """
        lines = list(self._recent)
        first_no = self._lines_total - len(lines)  # номер самой старой реплики в деке

        # Вплетаем подсказки туда, где они прозвучали: подсказка с номером n была
        # выдана после всех реплик с номерами < n. Делаем это ВСЕГДА, в том числе
        # на force: «смотреть на весь разговор» не значит «забыть свои ответы» —
        # без пометок модель снова видит закрытые вопросы открытыми.
        pending = [(no, text) for no, text in self._hint_log if no > first_no]
        rendered: list[str] = []
        new_from: Optional[int] = None  # индекс, с которого начинается «новое»
        for offset, line in enumerate(lines):
            no = first_no + offset
            while pending and pending[0][0] <= no:
                rendered.append(f"  [ты подсказал: {pending.pop(0)[1]}]")
            if new_from is None and no >= self._hinted_at_line:
                new_from = len(rendered)
            rendered.append(line)
        rendered.extend(f"  [ты подсказал: {text}]" for _, text in pending)

        # Делить не на что: первая подсказка, или нового не появилось, или
        # нажата кнопка — там человек просит посмотреть на разговор целиком.
        if force or not new_from:
            return "", "\n".join(rendered)[-budget_chars:]

        new_text = "\n".join(rendered[new_from:])[-budget_chars:]
        left = max(0, budget_chars - len(new_text))
        return ("\n".join(rendered[:new_from])[-left:] if left else ""), new_text

    def _window_chars(self, budget) -> int:
        """Сколько символов разговора влезает в один запрос подсказки.

        Из тарифного бюджета вычитаем сам промпт: обрезав разговор ровно по
        бюджету, мы отправили бы его ВМЕСТЕ с инструкциями и всё равно упёрлись
        бы в лимит — просто позже и непонятнее. Инструкции стоят около 1100
        токенов, то есть больше половины бюджета на бесплатном тарифе.

        Размер промпта не забиваем числом, а меряем сборкой с пустым
        транскриптом: поменяются промпты — пересчитается само, и в проекте не
        появится очередной константы, подобранной на глаз.

        Меряем вариантом с allow_skip=True — он длиннее (в нём есть правило
        молчания), то есть оценка получается с запасом в безопасную сторону.
        """
        if not budget.hints_tokens:
            return budget.hints_chars  # тарифного лимита нет — как раньше
        system, prompt = prompts.build_hint_prompt(
            mode=self._mode, transcript="", earlier="",
            previous="\n".join(self._recent_hints) or "—",
            title=self._meeting_title, participants=self._participants_line(),
            detailed=budget.detailed, allow_skip=True,
        )
        overhead = (len(system) + len(prompt)) / self._cfg.chars_per_token
        free = int((budget.hints_tokens - overhead) * self._cfg.chars_per_token)
        # Нижняя граница: если бюджета не хватает даже на инструкции, подсказки
        # на этом тарифе невозможны. Отправляем минимум и даём провайдеру
        # ответить 413 — теперь он объясняет, что делать, вместо тихой смерти.
        return max(self._cfg.hints_min_context_chars, min(budget.hints_chars, free))

    def _participants_line(self) -> str:
        """Кто говорил и сколько — строкой для промпта.

        Имена читаем из БД в момент сборки промпта, а не запоминаем при реплике:
        участника переименовывают по ходу встречи, и запомненное имя устарело бы
        ровно тогда, когда оно и становится осмысленным.
        """
        if not self._participants:
            return ""
        with session_scope() as db:
            names = crud.speaker_names(db, list(self._participants))
        return ", ".join(
            f"{names.get(speaker_id, 'Неизвестный')} ({n} реплик)"
            for speaker_id, n in self._participants.most_common()
        )

    async def _hint_delta(self, kind: str, chunk: str) -> None:
        """Куски подсказки по кнопке «подсказать сейчас».

        Мысли сюда не шлём: панель подсказок узкая, и рассуждения на английском
        в ней только мешали бы. Их место — окно чата, где человек сам решил
        спросить и готов читать длинный ответ.
        """
        if kind == "text":
            await self._send({"type": "hint_delta", "text": chunk})

    def _is_duplicate(self, hint: str) -> bool:
        """Почти-дубль недавней подсказки (сравнение без учёта регистра)."""
        candidate = hint.casefold()
        return any(
            difflib.SequenceMatcher(None, candidate, prev.casefold()).ratio()
            >= self._cfg.hints_dup_ratio
            for prev in self._recent_hints
        )

    async def _emit_hint(self, force: bool = False) -> None:
        """Единая точка генерации подсказки — из адаптивного цикла и из кнопки
        «подсказать сейчас» (force=True, в обход триггера и флага включённости).

        В авто-режиме модель вправе промолчать (вернуть SKIP) — тогда клиенту
        ничего не уходит. По кнопке молчать нельзя: пользователь спросил явно,
        поэтому allow_skip=False, дедуп отключён, а на каждый отказ он получает
        внятный ответ вместо тишины.
        """
        if force:
            # Спросили явно: уступает только другому явному запросу, а фоновую
            # подсказку снимает.
            if self._explicit_in_flight:
                await self._send({
                    "type": "hint_error", "message": "Подсказка уже готовится…",
                })
                return
            await self._cancel_auto_hint()
        elif self._hint_in_flight:
            return
        budget = self._llm.budget  # читаем на каждый вызов — провайдера могли сменить
        earlier, window = self._split_window(self._window_chars(budget), force)
        if len(window) + len(earlier) < self._cfg.hints_min_context_chars:
            if force:
                await self._send({
                    "type": "hint_error",
                    "message": "Пока слишком мало разговора для подсказки.",
                })
            return  # счётчики не трогаем — контекст копится дальше
        self._hint_in_flight = True
        self._explicit_in_flight = force
        self._chars_since_hint = 0
        self._last_hint_at = time.monotonic()
        try:
            system, prompt = prompts.build_hint_prompt(
                mode=self._mode,
                transcript=window,
                earlier=earlier,
                previous="\n".join(self._recent_hints) or "—",
                title=self._meeting_title,
                participants=self._participants_line(),
                detailed=budget.detailed,
                allow_skip=not force,
            )
            # Печатаем только по кнопке. Автоподсказку стримить нечем: модель
            # вправе промолчать, и молчание приходит словом SKIP в тексте
            # ответа — человек увидел бы, как в панели появляется «SKIP» и
            # исчезает. По кнопке молчать нельзя (allow_skip=False), и там
            # печатать безопасно.
            raw = await self._llm.hint(
                prompt, system=system, temperature=self._cfg.hints_temperature,
                on_delta=self._hint_delta if force else None,
            )
            self._hint_fail_streak = 0
            hint = prompts.parse_hint(raw, min_chars=self._cfg.hints_min_len_chars)
            # Модель этот текст посмотрела — двигаем границу, что бы она ни
            # ответила. Иначе отвергнутый фрагмент вернётся на следующей попытке,
            # и так по кругу, пока модель не надумает подсказку на пустом месте.
            # Не двигаем только при ошибке связи (ниже): там она текста не видела.
            self._hinted_at_line = self._lines_total
            if hint is None:  # модель промолчала
                if force:
                    await self._send({
                        "type": "hint_error",
                        "message": "Модель не нашла, что подсказать. Попробуйте позже.",
                    })
                else:
                    self._on_skip()
                return
            self._skip_streak = 0
            self._hint_gap_s = self._cfg.hints_min_gap_s
            if force or not self._is_duplicate(hint):
                self._recent_hints.append(hint)
                self._hint_log.append((self._lines_total, hint))
                await self._send({"type": "hint", "text": hint})
        except LlmError as exc:
            # Один сбой не выключает подсказки: наращиваем бэкофф и гасим только
            # после нескольких ошибок подряд (важно при нестабильной сети).
            self._hint_fail_streak += 1
            self._hint_backoff_until = time.monotonic() + min(
                self._cfg.hints_min_gap_s * 2 ** self._hint_fail_streak,
                self._cfg.hints_max_backoff_s,
            )
            if self._hint_fail_streak >= self._cfg.hints_max_fails:
                self._hints_enabled = False
                await self._send({
                    "type": "hint_error",
                    "message": "Подсказки приостановлены после нескольких ошибок связи с LLM.",
                })
            elif self._hint_fail_streak == 1:
                await self._send({"type": "hint_error", "message": str(exc)})
        finally:
            self._hint_in_flight = False
            self._explicit_in_flight = False

    # ------------------------------------------------------------------ вопрос от участника

    async def _answer(self, question: str, segment_ids: list) -> None:
        """Отвечает на вопрос участника, заданный из окна чата.

        Отличие от подсказки принципиальное: там модель сама решает, о чём
        говорить, и вправе промолчать. Здесь спросил человек — ответ обязателен,
        а тему задавать модели не надо.

        Выделенные реплики (если показал) идут отдельным блоком «вопрос про
        них», остальной разговор — контекстом. Молчать нельзя, дедупликация не
        нужна: два одинаковых вопроса — это два вопроса.
        """
        if self._meeting_id is None:
            return
        question = (question or "").strip()  # чистим здесь же: вызвать могут не только из _on_command
        if not question and not segment_ids:
            await self._send({
                "type": "answer_error", "message": "Пустой вопрос.",
            })
            return
        # Отказываем только настоящему прошлому вопросу. Фоновую подсказку —
        # снимаем: она необязательна, а её ожидание лимита длится десятки секунд.
        if self._explicit_in_flight:
            await self._send({
                "type": "answer_error", "message": "Модель ещё отвечает на прошлый вопрос…",
            })
            return
        await self._cancel_auto_hint()

        # Приводим id к целым: клиент может прислать что угодно, а дальше они
        # уходят в запрос к БД
        ids = [
            int(i) for i in segment_ids
            if isinstance(i, (int, float, str)) and str(i).lstrip("-").isdigit()
        ]
        with session_scope() as db:
            quoted_rows = crud.segments_by_ids(db, self._meeting_id, ids)
            quoted = "\n".join(
                f"[{self._mmss(row.start_s)}] "
                f"{row.speaker.name if row.speaker else 'Неизвестный'}: {row.text}"
                for row in quoted_rows
            )

        budget = self._llm.budget
        earlier = "\n".join(self._recent)[-self._window_chars(budget):]
        system, prompt = prompts.build_answer_prompt(
            mode=self._mode, question=question or prompts.ASK_ABOUT_SELECTED,
            quoted=quoted, earlier=earlier,
            title=self._meeting_title, participants=self._participants_line(),
        )

        self._hint_in_flight = True
        self._explicit_in_flight = True

        async def on_delta(kind: str, chunk: str) -> None:
            # Человек ждёт ответа и смотрит на экран: первый кусок приходит через
            # полсекунды вместо нескольких секунд тишины. Мысли идут отдельным
            # событием — показывать их вперемешку с ответом нельзя.
            await self._send({
                "type": "answer_reasoning" if kind == "reasoning" else "answer_delta",
                "text": chunk,
            })

        try:
            raw = await self._llm.hint(
                prompt, system=system, temperature=self._cfg.hints_temperature,
                on_delta=on_delta,
            )
            text = raw.strip()
            if not text:
                await self._send({
                    "type": "answer_error", "message": "Модель вернула пустой ответ.",
                })
                return
            await self._send({"type": "answer", "text": text})
        except LlmError as exc:
            # Бэкофф подсказок здесь не трогаем: человек спросил явно, и глушить
            # его вопросы из-за неудач фонового цикла нельзя.
            await self._send({"type": "answer_error", "message": str(exc)})
        finally:
            self._hint_in_flight = False
            self._explicit_in_flight = False

    def on_speakers_merged(self, source_id: int, target_id: int, name: str,
                           was_named: list[str]) -> None:
        """Двух спикеров объединили посреди встречи — приводим память в порядок.

        Профиля source_id в базе больше нет, а сессия помнит его в трёх местах:
        как донора для коротких реплик (иначе следующее «ага» уедет на удалённого),
        как недавно говорившего и в счётчике участников. Счётчик особенно важен:
        не сложив его, мы отдали бы модели на одного участника больше, чем есть.
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
        # Окно разговора для подсказок хранится строками «Имя: текст», и после
        # слияния в нём соседствовали бы два имени одного человека — модель
        # считала бы, что говорили двое.
        if was_named:
            self._recent = deque(
                (self._rename_line(line, was_named, name) for line in self._recent),
                maxlen=self._recent.maxlen,
            )

    @staticmethod
    def _rename_line(line: str, was_named: list[str], name: str) -> str:
        for old in was_named:
            if line.startswith(f"{old}: "):
                return f"{name}: {line[len(old) + 2:]}"
        return line

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
            if self._hints_task:
                self._hints_task.cancel()
            if self._auto_hint_task:
                self._auto_hint_task.cancel()  # цикл её уже не снимет: она сама по себе
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
