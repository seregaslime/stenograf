"""Глобальная библиотека голосов.

Спикер — это человек; у него 1..N «отпечатков» голоса (центроидов
ECAPA-эмбеддингов): гарнитура, телефон и ноутбук звучат по-разному. Новый
речевой сегмент сверяется со ВСЕМИ отпечатками всех спикеров по косинусной
близости — так человек узнаётся на следующей встрече в любом «звучании».

Объединение профилей (один человек распознался как два) не усредняет
отпечатки, а переносит их под одного спикера — данные не теряются.
"""
import logging
import shutil
import threading
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import SAMPLE_RATE, Settings
from ..db import crud
from ..db.models import Segment, Speaker, VoicePrint

log = logging.getLogger(__name__)


@dataclass
class MatchResult:
    speaker_id: int
    name: str
    is_self: bool
    similarity: Optional[float]  # None — сегмент был слишком коротким для эмбеддинга
    is_new: bool


@dataclass
class _Print:
    id: int
    vector: np.ndarray
    count: int


class SpeakerRegistry:
    def __init__(self, cfg: Settings):
        self._cfg = cfg
        self._lock = threading.Lock()
        # Библиотека у каждого своя: owner_id -> speaker_id -> отпечатки.
        # Ключ None — личный сервер, людей на нём не заводили. Пересечений нет
        # намеренно: сравнение идёт только со своими отпечатками, и чужие голоса
        # не участвуют в конкуренции за порог (лишние кандидаты — это лишние
        # шансы промахнуться), а «Вы» у каждого свой человек.
        self._prints: dict[Optional[int], dict[int, list[_Print]]] = {}
        self._self_ids: dict[Optional[int], int] = {}  # owner_id -> профиль «Вы»

    # --- загрузка / self ---

    def load(self, db: Session) -> None:
        """Поднимает библиотеки всех владельцев в память.

        Профиль «Вы» здесь не создаётся: владелец известен только когда придёт
        его сессия, а заводить пустые профили всем заранее незачем.
        """
        with self._lock:
            self._prints.clear()
            self._self_ids.clear()
            for speaker in db.scalars(select(Speaker).where(Speaker.is_self.is_(True))):
                self._self_ids[speaker.owner_id] = speaker.id
            total = 0
            строки = db.execute(
                select(VoicePrint, Speaker.owner_id).join(
                    Speaker, VoicePrint.speaker_id == Speaker.id
                )
            ).all()
            for row, owner_id in строки:
                vector = np.frombuffer(row.centroid, dtype=np.float32).copy()
                self._prints.setdefault(owner_id, {}).setdefault(row.speaker_id, []).append(
                    _Print(row.id, vector, row.embedding_count)
                )
                total += 1
            log.info(
                "Библиотеки голосов: %d владельцев, %d спикеров, %d отпечатков",
                len(self._prints),
                sum(len(люди) for люди in self._prints.values()),
                total,
            )

    def self_id(self, db: Session, owner_id: Optional[int] = None) -> int:
        """Профиль «Вы» этого человека, создаётся при первом обращении.

        Метод, а не свойство: до появления людей профиль был один на сервер и
        заводился при старте, теперь он у каждого свой и нужна база под рукой.
        """
        with self._lock:
            известен = self._self_ids.get(owner_id)
        if известен is not None:
            return известен
        speaker = crud.get_or_create_self_speaker(db, owner_id)
        with self._lock:
            self._self_ids[owner_id] = speaker.id
        return speaker.id

    def prints_of(self, speaker_id: int, owner_id: Optional[int] = None) -> list[_Print]:
        """Отпечатки одного профиля — для тестов и отладки диаризации.

        Раньше тесты индексировали внутренний словарь напрямую; после разделения
        библиотек по владельцам это стало двухуровневым обращением, которое
        пришлось бы повторять в каждой проверке.
        """
        with self._lock:
            return list(self._prints.get(owner_id, {}).get(speaker_id, []))

    # --- сопоставление ---

    def match_all(
        self,
        db: Session,
        embedding: np.ndarray,
        mic_dominant: bool,
        recent_ids: frozenset[int] = frozenset(),
        audio: Optional[np.ndarray] = None,
        owner_id: Optional[int] = None,
    ) -> MatchResult:
        """Ищет владельца голоса по отпечаткам всех спикеров ЭТОГО человека.

        Чужие библиотеки не просматриваются: голоса коллег — это лишние
        кандидаты, каждый из которых может выиграть по близости и увести
        реплику не туда.

        Живой голос от фразы к фразе гуляет сильнее порога, поэтому кроме
        близости работают два приора (скидки к порогу, берётся большая):
        - mic_dominant: голос пришёл в основном из микрофона — человек в
          комнате, скорее всего владелец («Вы»). Первый такой голос обучает
          профиль «Вы» автоматически;
        - recent_ids: кто говорил в последние полминуты, вероятно, говорит
          и сейчас — склеивает монолог, который иначе рассыпался бы на
          «Спикер N» на каждой просадке близости.

        Если голос узнан со скидкой (ниже основного порога) в достаточно
        длинной реплике, её «звучание» добавляется отдельным отпечатком — так
        профиль набирает варианты (гарнитура, комната, простуда), не размывая
        главный отпечаток. Пограничные коротыши отпечатков не оставляют.

        audio — реплика сегмента: сохраняется рядом с новым отпечатком, чтобы
        «звучание» можно было прослушать на вкладке «Спикеры».
        """
        cfg = self._cfg
        # Профиль «Вы» этого человека — вне замка: он может создаваться в базе
        self_id = self.self_id(db, owner_id)
        with self._lock:
            библиотека = self._prints.setdefault(owner_id, {})
            self_prints = библиотека.get(self_id, [])
            if mic_dominant and not self_prints:
                self._add_print(db, self_id, embedding, audio, owner_id)
                speaker = db.get(Speaker, self_id)
                log.info("Профиль «Вы» обучен по первому голосу из микрофона")
                return MatchResult(self_id, speaker.name, True, None, False)

            def required(speaker_id: int) -> float:
                bonus = 0.0
                if mic_dominant and speaker_id == self_id:
                    bonus = cfg.speaker_self_bonus
                if speaker_id in recent_ids:
                    bonus = max(bonus, cfg.speaker_recent_bonus)
                return cfg.speaker_match_threshold - bonus

            # Лучшая близость нужна только для журнала: по ней видно, насколько
            # мимо промахнулись, когда завели нового спикера. Сам «лучший»
            # отпечаток при этом не используется — линтер это и заметил.
            best_sim = -1.0
            matched: Optional[tuple[int, _Print, float]] = None
            for speaker_id, prints in библиотека.items():
                for print_ in prints:
                    sim = float(np.dot(print_.vector, embedding))
                    best_sim = max(best_sim, sim)
                    if sim >= required(speaker_id) and (matched is None or sim > matched[2]):
                        matched = (speaker_id, print_, sim)

            if matched is not None:
                speaker_id, print_, sim = matched
                is_self = speaker_id == self_id
                # Отпечатки «Вы» пополняем только голосом из микрофона, чтобы не
                # размывать их звуком из звонка (и наоборот для остальных)
                if is_self == mic_dominant:
                    if sim >= cfg.speaker_match_threshold:
                        self._update_print(db, print_, embedding)
                    elif (
                        len(библиотека[speaker_id]) < cfg.speaker_max_prints
                        and audio is not None
                        and len(audio) / SAMPLE_RATE >= cfg.speaker_print_min_s
                    ):
                        # узнали со скидкой в длинной реплике — другое «звучание»,
                        # новый отпечаток; пограничный коротыш не меняет профиль
                        self._add_print(db, speaker_id, embedding, audio, owner_id)
                speaker = db.get(Speaker, speaker_id)
                return MatchResult(speaker_id, speaker.name, is_self, round(sim, 3), False)

            speaker = crud.create_speaker(db, owner_id)
            self._add_print(db, speaker.id, embedding, audio, owner_id)
            log.info("Новый профиль голоса: %s (лучшая близость была %.3f)", speaker.name, best_sim)
            return MatchResult(speaker.id, speaker.name, False,
                               round(best_sim, 3) if best_sim > -1 else None, True)

    def _update_print(self, db: Session, print_: _Print, embedding: np.ndarray) -> None:
        """Скользящее среднее отпечатка; вес старого центроида ограничен, чтобы
        отпечаток мог медленно «дрейфовать» за голосом."""
        capped = min(print_.count, self._cfg.speaker_centroid_max_count)
        new = (print_.vector * capped + embedding) / (capped + 1)
        norm = np.linalg.norm(new)
        if norm > 0:
            new = new / norm
        print_.vector = new.astype(np.float32)
        print_.count += 1
        row = db.get(VoicePrint, print_.id)
        row.centroid = print_.vector.tobytes()
        row.embedding_count = print_.count

    def _add_print(
        self, db: Session, speaker_id: int, embedding: np.ndarray,
        audio: Optional[np.ndarray] = None, owner_id: Optional[int] = None,
    ) -> None:
        row = VoicePrint(
            speaker_id=speaker_id,
            centroid=embedding.astype(np.float32).tobytes(),
            embedding_count=1,
        )
        if audio is not None:
            row.audio_path, row.audio_duration_s = self._save_print_audio(speaker_id, audio)
        db.add(row)
        db.flush()
        self._prints.setdefault(owner_id, {}).setdefault(speaker_id, []).append(
            _Print(row.id, embedding.astype(np.float32), 1)
        )

    def _save_print_audio(self, speaker_id: int, audio: np.ndarray) -> tuple[str, float]:
        """Пишет wav-фрагмент реплики, из которой родился отпечаток, — чтобы
        «звучание» можно было прослушать. Длинные реплики обрезаются."""
        audio = audio[: int(self._cfg.speaker_print_audio_max_s * SAMPLE_RATE)]
        directory = self._cfg.samples_dir / f"spk_{speaker_id}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{uuid.uuid4().hex[:8]}.wav"
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(SAMPLE_RATE)
            f.writeframes(pcm.tobytes())
        return str(path), len(audio) / SAMPLE_RATE

    # --- ручные операции (вкладка «Спикеры») ---

    def merge(self, db: Session, id_a: int, id_b: int,
              owner_id: Optional[int] = None) -> dict:
        """Объединяет двух спикеров в одного. Целевой профиль выбирается сам:
        «Вы» > человеческое имя > больше реплик > меньший id. Отпечатки голоса
        и образцы переносятся, сегменты всех встреч переписываются.

        Оба профиля обязаны принадлежать тому, кто объединяет: иначе чужой
        профиль можно было бы «слить» в свой и забрать вместе с ним отпечатки
        голоса и реплики чужих встреч.
        """
        if id_a == id_b:
            raise ValueError("Нельзя объединить спикера с самим собой")
        a = crud.speaker_for_owner(db, id_a, owner_id)
        b = crud.speaker_for_owner(db, id_b, owner_id)
        if a is None or b is None:
            raise ValueError("Спикер не найден")

        def rank(speaker: Speaker) -> tuple:
            segments = db.scalar(
                select(func.count(Segment.id)).where(Segment.speaker_id == speaker.id)
            )
            return (speaker.is_self, self._has_custom_name(speaker), segments, -speaker.id)

        target, source = (a, b) if rank(a) >= rank(b) else (b, a)
        # Имена, под которыми эти двое уже успели прозвучать. Нужны тем, кто
        # держит их строками: живая сессия помнит окно разговора как «Имя: текст»,
        # и после слияния в нём остались бы оба имени одного человека.
        was_named = [source.name, target.name]
        if not self._has_custom_name(target) and self._has_custom_name(source):
            target.name = source.name

        with self._lock:
            библиотека = self._prints.setdefault(owner_id, {})
            moved_prints = библиотека.pop(source.id, [])
            библиотека.setdefault(target.id, []).extend(moved_prints)
        # Переприсваиваем родителя: объект атомарно переезжает между
        # коллекциями, и cascade delete-orphan при удалении source его не тронет
        target_dir = self._cfg.samples_dir / f"spk_{target.id}"
        for print_row in list(source.voiceprints):
            if print_row.audio_path and Path(print_row.audio_path).exists():
                target_dir.mkdir(parents=True, exist_ok=True)
                new_path = target_dir / Path(print_row.audio_path).name
                shutil.move(print_row.audio_path, new_path)
                print_row.audio_path = str(new_path)
            print_row.speaker = target
        db.flush()
        shutil.rmtree(self._cfg.samples_dir / f"spk_{source.id}", ignore_errors=True)
        moved = crud.reassign_segments(db, source.id, target.id)
        source_name = source.name
        db.delete(source)
        log.info(
            "Merge: «%s» → «%s», сегментов: %d, отпечатков теперь: %d",
            source_name, target.name, moved,
            len(self._prints.get(owner_id, {}).get(target.id, [])),
        )
        return {
            "target_id": target.id, "name": target.name, "moved_segments": moved,
            # source_id — чтобы вызывающий знал, какого профиля больше нет:
            # целевого выбирает сервер, и клиент заранее его не знает.
            "source_id": source.id,
            "was_named": [name for name in was_named if name != target.name],
        }

    @staticmethod
    def _has_custom_name(speaker: Speaker) -> bool:
        return bool(speaker.name) and speaker.name != f"Спикер {speaker.id}"

    def remove_print(self, db: Session, speaker_id: int, print_id: int,
                     owner_id: Optional[int] = None) -> bool:
        """Удаляет один отпечаток голоса (из памяти и из БД).

        Профиль остаётся: человек с сегментами не пропадает из встреч, просто
        перестаёт узнаваться этим «звучанием». Если удалить все отпечатки
        «Вы» — профиль заново обучится по первому голосу из микрофона."""
        # Владельца проверяем по спикеру: чужой отпечаток не должен удаляться
        # по угаданному номеру.
        свой = crud.speaker_for_owner(db, speaker_id, owner_id) is not None
        row = db.get(VoicePrint, print_id)
        found_db = свой and row is not None and row.speaker_id == speaker_id
        with self._lock:
            библиотека = self._prints.setdefault(owner_id, {})
            prints = библиотека.get(speaker_id, []) if свой else []
            kept = [p for p in prints if p.id != print_id]
            found_memory = len(kept) != len(prints)
            if found_memory:
                if kept:
                    библиотека[speaker_id] = kept
                else:
                    библиотека.pop(speaker_id, None)
        if found_db:
            if row.audio_path:
                Path(row.audio_path).unlink(missing_ok=True)
            db.delete(row)
        if found_db or found_memory:
            log.info("Удалён отпечаток #%d спикера #%d", print_id, speaker_id)
        return found_db or found_memory

    def forget(self, speaker_id: int, owner_id: Optional[int] = None) -> None:
        with self._lock:
            self._prints.get(owner_id, {}).pop(speaker_id, None)
