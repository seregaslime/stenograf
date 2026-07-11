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

from app.config import SAMPLE_RATE, Settings  # noqa: E402, F401
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


def test_speaker_echo_yields_single_segment(tmp_path):
    """Эхо из колонок: голос из звонка играет в комнате и попадает в микрофон
    с задержкой. В смешанном потоке это ОДИН сегмент (копия совпадает по
    времени с оригиналом), и доминанта — системный канал."""
    import subprocess

    import numpy as np

    from app.audio.mixer import ChannelMixer
    from app.audio.vad import SpeechSegmenter

    if shutil.which("say") is None:
        pytest.skip("нет команды say (не macOS)")
    wav = tmp_path / "phrase.wav"
    subprocess.run(
        ["say", "-v", "Milena", "-o", str(wav), "--data-format=LEI16@16000",
         "Коллеги, давайте обсудим результаты тестирования новой версии"],
        check=True, capture_output=True,
    )
    voice = eval_voices.load_wav_16k(wav)

    delay = int(0.04 * SAMPLE_RATE)  # звук долетел до микрофона за ~40 мс
    system = np.concatenate([voice, np.zeros(SAMPLE_RATE, dtype=np.float32)])
    mic = np.concatenate([np.zeros(delay, dtype=np.float32), voice * 0.3,
                          np.zeros(SAMPLE_RATE - delay, dtype=np.float32)])

    cfg = Settings(_env_file=None)
    mixer = ChannelMixer(cfg)
    segmenter = SpeechSegmenter(cfg)
    segments = []
    chunk = 1600  # кадры по 100 мс, как шлёт клиент
    for i in range(0, len(system), chunk):
        for channel, stream in (("mic", mic), ("system", system)):
            for mixed in mixer.feed(channel, stream[i:i + chunk]):
                segments += segmenter.feed(mixed)
    for mixed in mixer.flush():
        segments += segmenter.feed(mixed)
    segments += segmenter.flush()

    assert len(segments) == 1, f"эхо раздвоило фразу: {len(segments)} сегмента"
    assert mixer.dominance(segments[0].start_s, segments[0].end_s) == "system"
