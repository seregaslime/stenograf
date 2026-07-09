"""Юнит-тесты SpeakerRegistry — логики «кто говорит».

Вместо реальных голосов используются синтетические 192-мерные векторы
с точно заданной косинусной близостью: так проверяется именно решающая
логика (порог, дрейф центроида, merge), а не качество ECAPA.
"""
import numpy as np
import pytest

from app.db import crud
from app.diarization.registry import SpeakerRegistry

DIM = 192  # размерность ECAPA-эмбеддинга


def unit(vector: np.ndarray) -> np.ndarray:
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def vec_with_similarity(base: np.ndarray, sim: float, rng: np.random.Generator) -> np.ndarray:
    """Единичный вектор с заданной косинусной близостью к base."""
    ortho = rng.standard_normal(base.shape).astype(np.float32)
    ortho -= ortho.dot(base) * base
    ortho = unit(ortho)
    return unit(sim * base + np.sqrt(1.0 - sim**2) * ortho)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def registry(cfg, db_session) -> SpeakerRegistry:
    reg = SpeakerRegistry(cfg)
    reg.load(db_session)
    return reg


def test_first_voice_creates_new_profile(registry, db_session, rng):
    match = registry.match_system(db_session, unit(rng.standard_normal(DIM)))
    assert match.is_new
    assert not match.is_self


def test_same_voice_matches_same_profile(registry, db_session, rng):
    voice = unit(rng.standard_normal(DIM))
    first = registry.match_system(db_session, voice)
    second = registry.match_system(db_session, voice)
    assert not second.is_new
    assert second.speaker_id == first.speaker_id
    assert second.similarity == pytest.approx(1.0, abs=1e-3)


def test_below_threshold_creates_new_profile(registry, db_session, cfg, rng):
    base = unit(rng.standard_normal(DIM))
    first = registry.match_system(db_session, base)
    near_miss = vec_with_similarity(base, cfg.speaker_match_threshold - 0.03, rng)
    second = registry.match_system(db_session, near_miss)
    assert second.is_new
    assert second.speaker_id != first.speaker_id


def test_above_threshold_matches(registry, db_session, cfg, rng):
    base = unit(rng.standard_normal(DIM))
    first = registry.match_system(db_session, base)
    close = vec_with_similarity(base, cfg.speaker_match_threshold + 0.03, rng)
    second = registry.match_system(db_session, close)
    assert not second.is_new
    assert second.speaker_id == first.speaker_id


def test_three_distinct_voices_stay_separate(registry, db_session, rng):
    """Три человека с непохожими голосами: реплики вперемешку, профилей — три.

    Случайные векторы в 192 измерениях почти ортогональны (близость ~0), у
    реальных непохожих голосов она 0.1–0.3 — здесь моделируется удобный случай.
    """
    bases = [unit(rng.standard_normal(DIM)) for _ in range(3)]
    ids: dict[int, set[int]] = {0: set(), 1: set(), 2: set()}
    for round_ in range(4):  # 4 реплики каждого, по кругу — как в разговоре
        for person, base in enumerate(bases):
            utterance = vec_with_similarity(base, 0.9, rng)  # голос слегка «плавает»
            match = registry.match_system(db_session, utterance)
            ids[person].add(match.speaker_id)
    for person, speaker_ids in ids.items():
        assert len(speaker_ids) == 1, f"человека {person} распознало как {len(speaker_ids)} разных"
    all_ids = {next(iter(s)) for s in ids.values()}
    assert len(all_ids) == 3, "разных людей склеило в один профиль"


def test_similar_voices_merge_at_low_threshold(registry, db_session, cfg, rng):
    """Обратная сторона низкого порога: голоса с близостью выше порога
    считаются одним человеком. При threshold=0.35 два похожих голоса
    (близость 0.5 — бывает у людей одного пола и тембра) склеятся."""
    base = unit(rng.standard_normal(DIM))
    first = registry.match_system(db_session, base)
    similar_person = vec_with_similarity(base, cfg.speaker_match_threshold + 0.15, rng)
    second = registry.match_system(db_session, similar_person)
    assert not second.is_new
    assert second.speaker_id == first.speaker_id


def test_self_voice_is_excluded_from_matching(registry, db_session, rng):
    """Голос владельца микрофона идёт отдельным каналом; даже идентичный
    вектор в системном канале не должен сматчиться с профилем «Вы»."""
    voice = unit(rng.standard_normal(DIM))
    registry._add_print(db_session, registry.self_id, voice)
    match = registry.match_system(db_session, voice)
    assert match.speaker_id != registry.self_id
    assert match.is_new


def test_centroid_drifts_toward_recent_voice(registry, db_session, rng):
    """Отпечаток — скользящее среднее: после серии реплик он смещается к
    текущему звучанию голоса (человек охрип, сменил гарнитуру)."""
    base = unit(rng.standard_normal(DIM))
    drifted = vec_with_similarity(base, 0.85, rng)  # «новое звучание»
    registry.match_system(db_session, base)
    print_ = next(iter(p for pid, prints in registry._prints.items()
                       for p in prints if pid != registry.self_id))
    before = float(print_.vector.dot(drifted))
    for _ in range(10):
        registry.match_system(db_session, drifted)
    after = float(print_.vector.dot(drifted))
    assert after > before
    assert print_.count == 11


def test_merge_moves_prints_and_segments(registry, db_session, rng):
    """Один человек распознался как два профиля → merge переносит отпечатки
    и реплики, лишний профиль удаляется, данные не теряются."""
    voice_a = unit(rng.standard_normal(DIM))
    voice_b = unit(rng.standard_normal(DIM))
    a = registry.match_system(db_session, voice_a)
    b = registry.match_system(db_session, voice_b)
    meeting = crud.create_meeting(db_session, "тест", record_audio=False)
    crud.add_segment(db_session, meeting.id, a.speaker_id, "system", 0.0, 1.0, "раз", 0.9)
    crud.add_segment(db_session, meeting.id, b.speaker_id, "system", 1.0, 2.0, "два", 0.9)

    result = registry.merge(db_session, a.speaker_id, b.speaker_id)

    target = result["target_id"]
    assert result["moved_segments"] == 1
    assert len(registry._prints[target]) == 2
    # оба голоса теперь узнаются как один человек
    for voice in (voice_a, voice_b):
        match = registry.match_system(db_session, voice)
        assert match.speaker_id == target


def test_forget_removes_profile(registry, db_session, rng):
    voice = unit(rng.standard_normal(DIM))
    first = registry.match_system(db_session, voice)
    registry.forget(first.speaker_id)
    second = registry.match_system(db_session, voice)
    assert second.is_new
    assert second.speaker_id != first.speaker_id
