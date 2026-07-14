"""Глобальная библиотека голосов.

Спикер — это человек; у него 1..N «отпечатков» голоса (центроидов
ECAPA-эмбеддингов): гарнитура, телефон и ноутбук звучат по-разному. Новый
речевой сегмент сверяется со ВСЕМИ отпечатками всех спикеров по косинусной
близости — так человек узнаётся на следующей встрече в любом «звучании».

Объединение профилей (один человек распознался как два) не усредняет
отпечатки, а переносит их под одного спикера — данные не теряются.
"""
import logging
import threading
import uuid
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import SAMPLE_RATE, Settings
from ..db import crud
from ..db.models import Segment, Speaker, SpeakerSample, VoicePrint

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
        self._prints: dict[int, list[_Print]] = {}  # speaker_id -> отпечатки
        self._self_id: Optional[int] = None

    # --- загрузка / self ---

    def load(self, db: Session) -> None:
        with self._lock:
            self._prints.clear()
            self._self_id = crud.get_or_create_self_speaker(db).id
            total = 0
            for row in db.scalars(select(VoicePrint)):
                vector = np.frombuffer(row.centroid, dtype=np.float32).copy()
                self._prints.setdefault(row.speaker_id, []).append(
                    _Print(row.id, vector, row.embedding_count)
                )
                total += 1
            log.info(
                "Библиотека голосов: %d спикеров, %d отпечатков", len(self._prints), total
            )

    @property
    def self_id(self) -> int:
        assert self._self_id is not None, "registry.load() не вызван"
        return self._self_id

    # --- сопоставление ---

    def match_all(
        self,
        db: Session,
        embedding: np.ndarray,
        mic_dominant: bool,
        recent_ids: frozenset[int] = frozenset(),
    ) -> MatchResult:
        """Ищет владельца голоса по отпечаткам ВСЕХ спикеров, включая «Вы».

        Живой голос от фразы к фразе гуляет сильнее порога, поэтому кроме
        близости работают два приора (скидки к порогу, берётся большая):
        - mic_dominant: голос пришёл в основном из микрофона — человек в
          комнате, скорее всего владелец («Вы»). Первый такой голос обучает
          профиль «Вы» автоматически;
        - recent_ids: кто говорил в последние полминуты, вероятно, говорит
          и сейчас — склеивает монолог, который иначе рассыпался бы на
          «Спикер N» на каждой просадке близости.

        Если голос узнан со скидкой (ниже основного порога), его «звучание»
        добавляется отдельным отпечатком — так профиль набирает варианты
        (гарнитура, комната, простуда), а не размывает главный отпечаток.
        """
        cfg = self._cfg
        with self._lock:
            self_prints = self._prints.get(self._self_id, [])
            if mic_dominant and not self_prints:
                self._add_print(db, self._self_id, embedding)
                speaker = db.get(Speaker, self._self_id)
                log.info("Профиль «Вы» обучен по первому голосу из микрофона")
                return MatchResult(self._self_id, speaker.name, True, None, False)

            def required(speaker_id: int) -> float:
                bonus = 0.0
                if mic_dominant and speaker_id == self._self_id:
                    bonus = cfg.speaker_self_bonus
                if speaker_id in recent_ids:
                    bonus = max(bonus, cfg.speaker_recent_bonus)
                return cfg.speaker_match_threshold - bonus

            best: Optional[tuple[int, _Print]] = None
            best_sim = -1.0
            matched: Optional[tuple[int, _Print, float]] = None
            for speaker_id, prints in self._prints.items():
                for print_ in prints:
                    sim = float(np.dot(print_.vector, embedding))
                    if sim > best_sim:
                        best, best_sim = (speaker_id, print_), sim
                    if sim >= required(speaker_id) and (matched is None or sim > matched[2]):
                        matched = (speaker_id, print_, sim)

            if matched is not None:
                speaker_id, print_, sim = matched
                is_self = speaker_id == self._self_id
                # Отпечатки «Вы» пополняем только голосом из микрофона, чтобы не
                # размывать их звуком из звонка (и наоборот для остальных)
                if is_self == mic_dominant:
                    if sim < cfg.speaker_match_threshold and \
                            len(self._prints[speaker_id]) < cfg.speaker_max_prints:
                        # узнали со скидкой — это другое «звучание», новый отпечаток
                        self._add_print(db, speaker_id, embedding)
                    else:
                        self._update_print(db, print_, embedding)
                speaker = db.get(Speaker, speaker_id)
                return MatchResult(speaker_id, speaker.name, is_self, round(sim, 3), False)

            speaker = crud.create_speaker(db)
            self._add_print(db, speaker.id, embedding)
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

    def _add_print(self, db: Session, speaker_id: int, embedding: np.ndarray) -> None:
        row = VoicePrint(
            speaker_id=speaker_id,
            centroid=embedding.astype(np.float32).tobytes(),
            embedding_count=1,
        )
        db.add(row)
        db.flush()
        self._prints.setdefault(speaker_id, []).append(
            _Print(row.id, embedding.astype(np.float32), 1)
        )

    # --- аудио-образцы голоса ---

    def maybe_save_sample(self, db: Session, speaker_id: int, audio: np.ndarray) -> None:
        duration = len(audio) / SAMPLE_RATE
        if not (self._cfg.speaker_sample_min_s <= duration <= self._cfg.speaker_sample_max_s):
            return
        existing = db.scalar(
            select(func.count(SpeakerSample.id)).where(SpeakerSample.speaker_id == speaker_id)
        )
        if existing >= self._cfg.speaker_max_samples:
            return
        directory = self._cfg.samples_dir / f"spk_{speaker_id}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{uuid.uuid4().hex[:8]}.wav"
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(SAMPLE_RATE)
            f.writeframes(pcm.tobytes())
        db.add(SpeakerSample(speaker_id=speaker_id, path=str(path), duration_s=duration))

    # --- ручные операции (вкладка «Спикеры») ---

    def merge(self, db: Session, id_a: int, id_b: int) -> dict:
        """Объединяет двух спикеров в одного. Целевой профиль выбирается сам:
        «Вы» > человеческое имя > больше реплик > меньший id. Отпечатки голоса
        и образцы переносятся, сегменты всех встреч переписываются."""
        if id_a == id_b:
            raise ValueError("Нельзя объединить спикера с самим собой")
        a = db.get(Speaker, id_a)
        b = db.get(Speaker, id_b)
        if a is None or b is None:
            raise ValueError("Спикер не найден")

        def rank(speaker: Speaker) -> tuple:
            segments = db.scalar(
                select(func.count(Segment.id)).where(Segment.speaker_id == speaker.id)
            )
            return (speaker.is_self, self._has_custom_name(speaker), segments, -speaker.id)

        target, source = (a, b) if rank(a) >= rank(b) else (b, a)
        if not self._has_custom_name(target) and self._has_custom_name(source):
            target.name = source.name

        with self._lock:
            moved_prints = self._prints.pop(source.id, [])
            self._prints.setdefault(target.id, []).extend(moved_prints)
        # Переприсваиваем родителя: объект атомарно переезжает между
        # коллекциями, и cascade delete-orphan при удалении source его не тронет
        for print_row in list(source.voiceprints):
            print_row.speaker = target
        for sample in list(source.samples):
            sample.speaker = target
        db.flush()
        moved = crud.reassign_segments(db, source.id, target.id)
        source_name = source.name
        db.delete(source)
        log.info(
            "Merge: «%s» → «%s», сегментов: %d, отпечатков теперь: %d",
            source_name, target.name, moved, len(self._prints.get(target.id, [])),
        )
        return {"target_id": target.id, "name": target.name, "moved_segments": moved}

    @staticmethod
    def _has_custom_name(speaker: Speaker) -> bool:
        return bool(speaker.name) and speaker.name != f"Спикер {speaker.id}"

    def remove_print(self, db: Session, speaker_id: int, print_id: int) -> bool:
        """Удаляет один отпечаток голоса (из памяти и из БД).

        Профиль остаётся: человек с сегментами не пропадает из встреч, просто
        перестаёт узнаваться этим «звучанием». Если удалить все отпечатки
        «Вы» — профиль заново обучится по первому голосу из микрофона."""
        row = db.get(VoicePrint, print_id)
        found_db = row is not None and row.speaker_id == speaker_id
        with self._lock:
            prints = self._prints.get(speaker_id, [])
            kept = [p for p in prints if p.id != print_id]
            found_memory = len(kept) != len(prints)
            if found_memory:
                if kept:
                    self._prints[speaker_id] = kept
                else:
                    self._prints.pop(speaker_id, None)
        if found_db:
            db.delete(row)
        if found_db or found_memory:
            log.info("Удалён отпечаток #%d спикера #%d", print_id, speaker_id)
        return found_db or found_memory

    def forget(self, speaker_id: int) -> None:
        with self._lock:
            self._prints.pop(speaker_id, None)
