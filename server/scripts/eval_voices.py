"""Оценка распознавания голосов: подходит ли порог speaker_match_threshold.

Берёт записи нескольких людей, считает ECAPA-эмбеддинги и показывает:
  1) насколько похожи фразы одного человека и разных людей (мин/среднее/макс);
  2) что получится при разных порогах — прогоняет фразы вперемешку через
     настоящий SpeakerRegistry и считает ошибки двух типов:
     «раздвоило» (один человек стал несколькими профилями) и
     «склеило» (разные люди попали в один профиль).

Как записать материал: каждый участник наговаривает 4–6 фраз по 2–5 секунд
(любым диктофоном), файлы складываются по папкам — папка = человек:

    voices/
      sergey/fraza1.wav fraza2.wav ...
      drug/...
      tretiy/...

Запуск из папки server:
    .venv/bin/python scripts/eval_voices.py --dir voices/
    .venv/bin/python scripts/eval_voices.py --samples   # образцы из вкладки «Спикеры»
    .venv/bin/python scripts/eval_voices.py --synth     # демо на голосах macOS (say)

Формат файлов: wav/m4a/mp3 с любой частотой — всё, что не 16 кГц mono wav,
конвертируется встроенным в macOS afconvert.
"""
import argparse
import itertools
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SAMPLE_RATE, Settings  # noqa: E402
from app.diarization.embedder import VoiceEmbedder  # noqa: E402


# ------------------------------------------------------------------ загрузка аудио

def _read_wave_16k(path: Path) -> np.ndarray | None:
    """Читает wav, если он уже 16 кГц mono PCM16, иначе None."""
    with wave.open(str(path)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            return None
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return data.astype(np.float32) / 32768.0


def load_wav_16k(path: Path) -> np.ndarray:
    try:
        audio = _read_wave_16k(path)
        if audio is not None:
            return audio
    except (wave.Error, EOFError):
        pass  # не wav — конвертируем
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        converted = Path(tmp.name)
    try:
        result = subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1",
             str(path), str(converted)],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Не удалось прочитать {path}: {result.stderr.decode().strip()}")
        return _read_wave_16k(converted)
    finally:
        converted.unlink(missing_ok=True)


def collect_from_dir(root: Path) -> dict[str, list[Path]]:
    people: dict[str, list[Path]] = {}
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for f in person_dir.iterdir()
                       if f.suffix.lower() in (".wav", ".m4a", ".mp3", ".flac", ".ogg"))
        if files:
            people[person_dir.name] = files
    return people


def collect_from_samples(cfg: Settings) -> dict[str, list[Path]]:
    """Аудио «звучаний» голоса, сохранённых при создании отпечатков (вкладка «Спикеры»)."""
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker

    from app.db.models import Speaker, VoicePrint

    engine = create_engine(f"sqlite:///{cfg.db_path}")
    session = sessionmaker(bind=engine)()
    people: dict[str, list[Path]] = {}
    try:
        for speaker in session.scalars(select(Speaker)):
            paths = [Path(p.audio_path) for p in session.scalars(
                select(VoicePrint).where(VoicePrint.speaker_id == speaker.id))
                if p.audio_path]
            paths = [p for p in paths if p.exists()]
            if paths:
                people[speaker.name] = paths
    finally:
        session.close()
    return people


_SYNTH_PHRASES = [
    "Коллеги, давайте начнём встречу и обсудим план на неделю",
    "У меня есть несколько вопросов по текущему статусу проекта",
    "Предлагаю перенести обсуждение бюджета на следующий раз",
    "Мне кажется, стоит сначала посмотреть на результаты тестирования",
    "Хорошо, тогда я подготовлю отчёт к завтрашнему утру",
]

# Голоса macOS для демо: три «человека». У Милены русский родной, у остальных
# акцент — для эмбеддинга важен тембр, а не произношение.
_SYNTH_VOICES = ["Milena", "Daniel", "Anna"]


def synthesize_demo(out_dir: Path) -> dict[str, list[Path]]:
    people: dict[str, list[Path]] = {}
    for voice in _SYNTH_VOICES:
        directory = out_dir / voice.lower()
        directory.mkdir(parents=True, exist_ok=True)
        files = []
        for i, phrase in enumerate(_SYNTH_PHRASES):
            path = directory / f"phrase_{i}.wav"
            result = subprocess.run(
                ["say", "-v", voice, "-o", str(path),
                 "--data-format=LEI16@16000", phrase],
                capture_output=True,
            )
            if result.returncode != 0:
                print(f"  голос {voice} недоступен — пропускаю "
                      f"({result.stderr.decode().strip()})")
                break
            files.append(path)
        if files:
            people[voice] = files
    return people


# ------------------------------------------------------------------ анализ

def similarity_report(names: list[str], embeddings: dict[str, list[np.ndarray]]) -> tuple[float, float]:
    """Печатает статистику близостей; возвращает (худшая_своя, лучшая_чужая)."""
    print("\n— Близость фраз ОДНОГО человека (интра):")
    worst_intra = 1.0
    for name in names:
        vectors = embeddings[name]
        sims = [float(a.dot(b)) for a, b in itertools.combinations(vectors, 2)]
        if not sims:
            print(f"  {name:20s} — только одна фраза, нечего сравнивать")
            continue
        worst_intra = min(worst_intra, min(sims))
        print(f"  {name:20s} мин {min(sims):.3f}   среднее {np.mean(sims):.3f}   макс {max(sims):.3f}")

    print("\n— Близость фраз РАЗНЫХ людей (интер):")
    best_inter = -1.0
    for a, b in itertools.combinations(names, 2):
        sims = [float(x.dot(y)) for x in embeddings[a] for y in embeddings[b]]
        best_inter = max(best_inter, max(sims))
        print(f"  {a} ↔ {b:15s} мин {min(sims):.3f}   среднее {np.mean(sims):.3f}   макс {max(sims):.3f}")
    return worst_intra, best_inter


def simulate(names: list[str], embeddings: dict[str, list[np.ndarray]], threshold: float) -> tuple[int, int, int]:
    """Прогоняет фразы по кругу через настоящий SpeakerRegistry.

    Возвращает (профилей_создано, раздвоений, склеек)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.db import models
    from app.diarization.registry import SpeakerRegistry

    engine = create_engine("sqlite://")
    models.Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, expire_on_commit=False)()
    cfg = Settings(speaker_match_threshold=threshold, _env_file=None)
    registry = SpeakerRegistry(cfg)
    registry.load(db)

    assigned: dict[str, set[int]] = {name: set() for name in names}
    # реплики вперемешку, как в живом разговоре
    for utterances in itertools.zip_longest(*(embeddings[n] for n in names)):
        for name, vector in zip(names, utterances):
            if vector is not None:
                assigned[name].add(
                    registry.match_all(db, vector, mic_dominant=False).speaker_id
                )

    profiles = set().union(*assigned.values())
    splits = sum(1 for ids in assigned.values() if len(ids) > 1)
    merges = sum(1 for a, b in itertools.combinations(names, 2) if assigned[a] & assigned[b])
    db.close()
    return len(profiles), splits, merges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dir", type=Path, help="папка: подпапка = человек, внутри wav-файлы")
    source.add_argument("--samples", action="store_true", help="образцы из вкладки «Спикеры»")
    source.add_argument("--synth", action="store_true", help="демо на синтезированных голосах macOS")
    args = parser.parse_args()

    cfg = Settings()
    if args.dir:
        people = collect_from_dir(args.dir)
    elif args.samples:
        people = collect_from_samples(cfg)
    else:
        tmp = Path(tempfile.mkdtemp(prefix="stenograf_voices_"))
        print(f"Синтез демо-голосов ({', '.join(_SYNTH_VOICES)}) в {tmp}...")
        people = synthesize_demo(tmp)

    people = {name: files for name, files in people.items() if len(files) >= 2}
    if len(people) < 2:
        sys.exit("Нужно минимум два человека и по 2+ фразы у каждого — сравнивать нечего.")

    print(f"\nУчастники: {', '.join(f'{n} ({len(f)} фраз)' for n, f in people.items())}")
    print("Загрузка ECAPA и расчёт эмбеддингов (первый раз — до минуты)...")
    embedder = VoiceEmbedder(cfg)
    names = list(people)
    embeddings: dict[str, list[np.ndarray]] = {}
    for name, files in people.items():
        vectors = []
        for path in files:
            audio = load_wav_16k(path)
            if len(audio) / SAMPLE_RATE < cfg.speaker_min_embed_s:
                print(f"  {path.name}: короче {cfg.speaker_min_embed_s} с — пропущен")
                continue
            vectors.append(embedder.embed(audio))
        embeddings[name] = vectors

    worst_intra, best_inter = similarity_report(names, embeddings)

    print(f"\n— Зазор: худшая «своя» близость {worst_intra:.3f} против "
          f"лучшей «чужой» {best_inter:.3f}")
    if worst_intra > best_inter:
        middle = (worst_intra + best_inter) / 2
        print(f"  Голоса разделимы. Надёжный порог — примерно {middle:.2f} "
              f"(между {best_inter:.2f} и {worst_intra:.2f}).")
    else:
        print("  Диапазоны пересекаются: идеального порога нет, будут ошибки. "
              "Помогают более длинные фразы и одинаковые условия записи.")

    print(f"\n— Симуляция реального алгоритма (фразы вперемешку), "
          f"людей на входе: {len(names)}")
    print(f"  {'порог':>6s}  {'профилей':>8s}  {'раздвоено':>9s}  {'склеено пар':>11s}")
    for threshold in np.arange(0.20, 0.61, 0.05):
        profiles, splits, merges = simulate(names, embeddings, round(float(threshold), 2))
        ok = "  ✓" if (profiles == len(names) and not splits and not merges) else ""
        current = "  ← текущий" if abs(threshold - cfg.speaker_match_threshold) < 0.024 else ""
        print(f"  {threshold:6.2f}  {profiles:8d}  {splits:9d}  {merges:11d}{ok}{current}")
    print("\n  ✓ — все распознаны правильно; «раздвоено» — человек стал несколькими"
          "\n  профилями (порог велик); «склеено» — разные люди в одном (порог мал).")


if __name__ == "__main__":
    main()
