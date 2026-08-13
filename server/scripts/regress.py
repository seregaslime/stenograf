"""Регрессионный прогон: не стало ли хуже после правки конвейера.

Замечание куратора №1. Тесты проверяют логику, но не качество: после правки VAD,
ASR или диаризации всё зелёное, а распознаёт хуже — и заметить это можно только
на слух, случайно и поздно.

Прогон считает два числа на эталонном наборе:

  WER          доля ошибок в словах. Эталон известен точно: фразы синтезированы
               (scripts/make_regress_fixtures.py), а не расшифрованы на слух;
  диаризация   раздвоения и склейки на трёх голосах — один человек стал двумя
               профилями или два человека попали в один.

Оба сравниваются с baseline.json. Стало хуже допуска — выход с кодом 1.

Запуск из папки server:
    .venv/bin/python scripts/regress.py                 # проверить
    .venv/bin/python scripts/regress.py --update        # записать новый эталон

Модели лежат в контейнере, поэтому обычно так:
    docker compose cp server/scripts/regress.py server:/tmp/regress.py
    docker compose cp server/tests/fixtures server:/tmp/fixtures
    docker compose exec -e PYTHONPATH=/srv server python /tmp/regress.py \\
        --fixtures /tmp/fixtures/regress
"""
import argparse
import asyncio
import itertools
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings, settings  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "regress"
BASELINE = FIXTURES / "baseline.json"

# Допуски. WER скачет от прогона к прогону из-за недетерминированности ASR, и
# слишком узкий допуск сделал бы прогон красным без причины — а красный без
# причины перестают читать. 0.02 — это два слова из сотни.
WER_TOLERANCE = 0.02


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# Знаки и регистр из сравнения убираем: GigaAM ставит пунктуацию сам, и без
# нормализации «недели.» считалось бы ошибкой против «недели». Мы меряем
# распознавание слов, а не расстановку точек.
_PUNCT = str.maketrans("", "", ".,!?;:—–-«»\"'()")


def _words(text: str) -> list[str]:
    return text.lower().replace("ё", "е").translate(_PUNCT).split()


def wer(reference: str, hypothesis: str) -> float:
    """Доля ошибок в словах (расстояние Левенштейна по словам)."""
    ref, hyp = _words(reference), _words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)] / len(ref)


async def measure_wer(root: Path, reference: dict) -> float:
    from app.asr.transcriber import Transcriber

    tr = Transcriber(settings)
    tr.load()
    ошибки = []
    for item in reference["speech"]:
        текст = await tr.transcribe(read_wav(root / item["file"]))
        доля = wer(item["text"], текст)
        ошибки.append(доля)
        метка = "  " if доля <= 0.25 else "!!"
        print(f"  {метка} {доля:.2f}  {текст[:60]}")
    return float(np.mean(ошибки)) if ошибки else 1.0


def measure_diarization(root: Path, reference: dict) -> tuple[int, int, int]:
    """Прогоняет голоса вперемешку через настоящий SpeakerRegistry.

    Возвращает (профилей, раздвоений, склеек). Логика повторяет eval_voices.py:
    там она для подбора порога, здесь — для проверки, что порог всё ещё держит.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.db import models
    from app.diarization.embedder import VoiceEmbedder
    from app.diarization.registry import SpeakerRegistry

    embedder = VoiceEmbedder(settings)
    векторы = {
        имя: [embedder.embed(read_wav(root / f)) for f in files]
        for имя, files in reference["voices"].items()
    }

    engine = create_engine("sqlite://")
    models.Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, expire_on_commit=False)()
    registry = SpeakerRegistry(Settings(_env_file=None))
    registry.load(db)

    имена = list(векторы)
    назначено: dict[str, set[int]] = {имя: set() for имя in имена}
    for реплики in itertools.zip_longest(*(векторы[имя] for имя in имена)):
        for имя, вектор in zip(имена, реплики):
            if вектор is not None:
                назначено[имя].add(
                    registry.match_all(db, вектор, mic_dominant=False).speaker_id
                )
    db.close()

    профилей = len(set().union(*назначено.values()))
    раздвоений = sum(1 for ids in назначено.values() if len(ids) > 1)
    склеек = sum(1 for a, b in itertools.combinations(имена, 2)
                 if назначено[a] & назначено[b])
    return профилей, раздвоений, склеек


def compare(текущее: dict, эталон: dict) -> list[str]:
    """Что стало хуже. Пустой список — регрессии нет."""
    беды = []
    if текущее["wer"] > эталон["wer"] + WER_TOLERANCE:
        беды.append(f"WER вырос: {эталон['wer']:.3f} → {текущее['wer']:.3f} "
                    f"(допуск {WER_TOLERANCE})")
    for ключ, что in (("splits", "раздвоений"), ("merges", "склеек")):
        if текущее[ключ] > эталон[ключ]:
            беды.append(f"{что.capitalize()} стало больше: "
                        f"{эталон[ключ]} → {текущее[ключ]}")
    return беды


async def main() -> None:
    parser = argparse.ArgumentParser(description="Регрессионный прогон Стенографа")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--update", action="store_true",
                        help="записать текущие числа как новый эталон")
    args = parser.parse_args()

    root = args.fixtures
    reference_path = root / "reference.json"
    if not reference_path.exists():
        sys.exit(f"Нет эталонного набора в {root}. "
                 "Сгенерируйте: python scripts/make_regress_fixtures.py")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    print(f"Распознавание ({len(reference['speech'])} фраз):")
    значение_wer = await measure_wer(root, reference)
    print(f"  средний WER: {значение_wer:.3f}\n")

    print(f"Диаризация ({len(reference['voices'])} голосов):")
    профилей, раздвоений, склеек = measure_diarization(root, reference)
    print(f"  профилей создано: {профилей} (ожидается {len(reference['voices'])})")
    print(f"  раздвоений: {раздвоений}, склеек: {склеек}\n")

    текущее = {"wer": round(значение_wer, 3), "profiles": профилей,
               "splits": раздвоений, "merges": склеек}

    baseline_path = root / "baseline.json"
    if args.update or not baseline_path.exists():
        baseline_path.write_text(
            json.dumps(текущее, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Эталон записан в {baseline_path}:\n  {текущее}")
        return

    эталон = json.loads(baseline_path.read_text(encoding="utf-8"))
    беды = compare(текущее, эталон)
    if беды:
        print("РЕГРЕССИЯ:")
        for беда in беды:
            print(f"  - {беда}")
        sys.exit(1)
    print(f"Регрессии нет. Эталон: {эталон}")


if __name__ == "__main__":
    asyncio.run(main())
