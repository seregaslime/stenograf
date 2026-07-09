"""Интеграционный тест: реальная ECAPA + три синтезированных голоса macOS.

Запуск (небыстрый, грузит модель): .venv/bin/python -m pytest -m integration

Проверяет всю цепочку «звук → эмбеддинг → профиль» на трёх разных голосах.
Синтетические голоса стабильнее человеческих, поэтому это дымовой тест
пайплайна; для оценки живых голосов есть scripts/eval_voices.py.
"""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))
import eval_voices  # noqa: E402

from app.config import SAMPLE_RATE, Settings  # noqa: E402
from app.diarization.embedder import VoiceEmbedder  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def synth_embeddings(tmp_path_factory):
    if shutil.which("say") is None:
        pytest.skip("нет команды say (не macOS)")
    people = eval_voices.synthesize_demo(tmp_path_factory.mktemp("voices"))
    if len(people) < 3:
        pytest.skip("установлено меньше трёх голосов macOS")
    cfg = Settings(_env_file=None)
    embedder = VoiceEmbedder(cfg)
    return {
        name: [embedder.embed(eval_voices.load_wav_16k(f)) for f in files]
        for name, files in people.items()
    }


def test_own_phrases_closer_than_foreign(synth_embeddings):
    """Худшая близость «своих» фраз должна быть выше лучшей «чужой» —
    иначе никакой порог не разделит эти голоса."""
    import itertools

    intra = [
        float(a.dot(b))
        for vectors in synth_embeddings.values()
        for a, b in itertools.combinations(vectors, 2)
    ]
    inter = [
        float(x.dot(y))
        for va, vb in itertools.combinations(synth_embeddings.values(), 2)
        for x in va for y in vb
    ]
    assert min(intra) > max(inter), (
        f"голоса неразделимы: своя min {min(intra):.3f} ≤ чужая max {max(inter):.3f}"
    )


def test_current_threshold_separates_three_voices(synth_embeddings):
    """С порогом из конфига три голоса дают ровно три профиля,
    без раздвоений и склеек."""
    cfg = Settings(_env_file=None)
    names = list(synth_embeddings)
    profiles, splits, merges = eval_voices.simulate(
        names, synth_embeddings, cfg.speaker_match_threshold
    )
    assert (profiles, splits, merges) == (3, 0, 0)
