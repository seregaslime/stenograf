"""Глобальная библиотека голосов.

Профили спикеров живут между встречами: для каждого хранится центроид
ECAPA-эмбеддингов. Новый речевой сегмент сверяется с библиотекой по косинусной
близости — так человек «узнаётся» на следующей встрече. Ошибки (один человек
на разных микрофонах = два профиля) чинятся ручным merge.
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
from ..db.models import Speaker, SpeakerSample

log = logging.getLogger(__name__)


@dataclass
class MatchResult:
    speaker_id: int
    name: str
    is_self: bool
    similarity: Optional[float]  # None — сегмент был слишком коротким для эмбеддинга
    is_new: bool


class SpeakerRegistry:
    def __init__(self, cfg: Settings):
        self._cfg = cfg
        self._lock = threading.Lock()
        # speaker_id -> (центроид, число эмбеддингов в нём)
        self._centroids: dict[int, tuple[np.ndarray, int]] = {}
        self._self_id: Optional[int] = None

    # --- загрузка / self ---

    def load(self, db: Session) -> None:
        with self._lock:
            self._centroids.clear()
            self._self_id = crud.get_or_create_self_speaker(db).id
            for speaker in db.scalars(select(Speaker)):
                if speaker.centroid is not None:
                    vec = np.frombuffer(speaker.centroid, dtype=np.float32).copy()
                    self._centroids[speaker.id] = (vec, speaker.embedding_count)
            log.info("Библиотека голосов: %d профилей с центроидами", len(self._centroids))

    @property
    def self_id(self) -> int:
        assert self._self_id is not None, "registry.load() не вызван"
        return self._self_id

    # --- сопоставление ---

    def match_system(self, db: Session, embedding: np.ndarray) -> MatchResult:
        """Ищет владельца голоса среди известных профилей, иначе создаёт новый."""
        with self._lock:
            best_id, best_sim = None, -1.0
            for speaker_id, (centroid, _count) in self._centroids.items():
                if speaker_id == self._self_id:
                    continue  # голос владельца микрофона идёт отдельным каналом
                sim = float(np.dot(centroid, embedding))
                if sim > best_sim:
                    best_id, best_sim = speaker_id, sim

            if best_id is not None and best_sim >= self._cfg.speaker_match_threshold:
                self._update_centroid(db, best_id, embedding)
                speaker = db.get(Speaker, best_id)
                return MatchResult(best_id, speaker.name, False, round(best_sim, 3), False)

            speaker = crud.create_speaker(db)
            self._set_centroid(db, speaker.id, embedding, 1)
            log.info("Новый профиль голоса: %s (лучшая близость была %.3f)", speaker.name, best_sim)
            return MatchResult(speaker.id, speaker.name, False,
                               round(best_sim, 3) if best_sim > -1 else None, True)

    def _update_centroid(self, db: Session, speaker_id: int, embedding: np.ndarray) -> None:
        centroid, count = self._centroids[speaker_id]
        capped = min(count, self._cfg.speaker_centroid_max_count)
        new = (centroid * capped + embedding) / (capped + 1)
        norm = np.linalg.norm(new)
        if norm > 0:
            new = new / norm
        self._set_centroid(db, speaker_id, new.astype(np.float32), count + 1)

    def _set_centroid(self, db: Session, speaker_id: int, vec: np.ndarray, count: int) -> None:
        self._centroids[speaker_id] = (vec, count)
        speaker = db.get(Speaker, speaker_id)
        speaker.centroid = vec.tobytes()
        speaker.embedding_count = count

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

    def merge(self, db: Session, source_id: int, target_id: int) -> dict:
        """Сливает source в target: сегменты всех встреч, образцы, центроиды."""
        source = db.get(Speaker, source_id)
        target = db.get(Speaker, target_id)
        if source is None or target is None:
            raise ValueError("Спикер не найден")
        if source_id == target_id:
            raise ValueError("Нельзя объединить спикера с самим собой")
        if source.is_self:
            # «Вы» всегда остаётся — меняем направление слияния
            source, target = target, source
            source_id, target_id = target_id, source_id

        with self._lock:
            src = self._centroids.pop(source_id, None)
            dst = self._centroids.get(target_id)
            if src is not None and dst is not None:
                src_vec, src_n = src
                dst_vec, dst_n = dst
                merged = (dst_vec * dst_n + src_vec * src_n) / (dst_n + src_n)
                norm = np.linalg.norm(merged)
                if norm > 0:
                    merged = merged / norm
                self._set_centroid(db, target_id, merged.astype(np.float32), dst_n + src_n)
            elif src is not None:
                self._set_centroid(db, target_id, src[0], src[1])

        moved = crud.reassign_segments(db, source_id, target_id)
        for sample in list(source.samples):
            sample.speaker_id = target_id
        db.delete(source)
        log.info("Merge: «%s» → «%s», перенесено сегментов: %d", source.name, target.name, moved)
        return {"target_id": target_id, "moved_segments": moved}

    def forget(self, speaker_id: int) -> None:
        with self._lock:
            self._centroids.pop(speaker_id, None)
