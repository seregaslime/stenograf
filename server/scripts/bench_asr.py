"""Сравнение ASR-моделей: качество (WER) и скорость всех движков и размеров.

Синтезирует набор «совещательных» фраз голосом macOS (say -v Milena), прогоняет
через каждую комбинацию движок×модель и печатает таблицу: доля ошибок в словах
(WER), скорость относительно реального времени, время загрузки и память.

Каждая модель тестируется в отдельном процессе — так замеры памяти честные
и модели не мешают друг другу. Синтезированный голос чище живой речи, поэтому
абсолютные WER занижены; сравнивать модели между собой — можно.

Запуск из папки server:  .venv/bin/python scripts/bench_asr.py
"""
import argparse
import asyncio
import json
import resource
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SAMPLE_RATE  # noqa: E402

# Фразы без цифр (whisper пишет числа то словами, то цифрами — WER стал бы нечестным),
# но со сложными местами: имена, термины, длинные слова.
PHRASES = [
    "Коллеги, давайте начнём совещание по проекту Стенограф",
    "Илья Петрович предлагает перенести дедлайн на середину сентября",
    "Бюджет на третий квартал уже согласован с руководством департамента",
    "Нужно интегрировать распознавание речи с базой данных и веб-интерфейсом",
    "Отдел цифровизации рассмотрит предложение о внедрении нейросетей",
    "Протокол встречи будет готов к завтрашнему утру, ответственный Сергей",
    "Качество распознавания зависит от микрофона и фонового шума",
    "Согласовали техническое задание с департаментом информационных технологий",
    "Есть вопросы по диаризации голосов и определению спикеров",
    "Подведём итоги и назначим следующую встречу на конец недели",
]

COMBOS = [
    ("faster_whisper", "tiny"),
    ("faster_whisper", "base"),
    ("faster_whisper", "small"),
    ("mlx", "tiny"),
    ("mlx", "base"),
    ("mlx", "small"),
    ("gigaam", "v3_e2e_ctc"),
    ("gigaam", "v3_e2e_rnnt"),
]


# ------------------------------------------------------------------ WER

def normalize(text: str) -> list[str]:
    text = text.lower().replace("ё", "е")
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
    return cleaned.split()


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """Расстояние Левенштейна по словам → (ошибок, слов в эталоне)."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        current = [i]
        for j, hyp_word in enumerate(hyp, 1):
            current.append(min(
                previous[j] + 1,                                # удаление
                current[j - 1] + 1,                             # вставка
                previous[j - 1] + (ref_word != hyp_word),       # замена
            ))
        previous = current
    return previous[-1], len(ref)


# ------------------------------------------------------------------ дочерний процесс

def run_single(engine: str, model: str, wav_dir: Path) -> None:
    """Меряет одну модель и печатает JSON. Вызывается родителем в подпроцессе."""
    from app.asr.transcriber import Transcriber
    from app.config import Settings

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cfg = Settings(asr_engine=engine, asr_model=model, _env_file=None)
    transcriber = Transcriber(cfg)

    t0 = time.perf_counter()
    transcriber.load()
    load_s = time.perf_counter() - t0

    wavs = sorted(wav_dir.glob("*.wav"))

    def read(path: Path) -> np.ndarray:
        with wave.open(str(path)) as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            return data.astype(np.float32) / 32768.0

    asyncio.run(transcriber.transcribe(read(wavs[0])))  # прогрев не в зачёт

    results, infer_s, audio_s = [], 0.0, 0.0
    for path in wavs:
        audio = read(path)
        t0 = time.perf_counter()
        text = asyncio.run(transcriber.transcribe(audio))
        infer_s += time.perf_counter() - t0
        audio_s += len(audio) / SAMPLE_RATE
        results.append(text)

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({
        "load_s": load_s, "infer_s": infer_s, "audio_s": audio_s,
        "texts": results, "rss_delta_mb": (rss_after - rss_before) / 1e6,
    }))


# ------------------------------------------------------------------ родитель

def synthesize(out_dir: Path) -> None:
    for i, phrase in enumerate(PHRASES):
        path = out_dir / f"{i:02d}.wav"
        subprocess.run(
            ["say", "-v", "Milena", "-o", str(path), "--data-format=LEI16@16000", phrase],
            check=True, capture_output=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк ASR-моделей")
    parser.add_argument("--run-single", nargs=3, metavar=("ENGINE", "MODEL", "WAV_DIR"),
                        help="служебный режим: одна модель в этом процессе")
    args = parser.parse_args()
    if args.run_single:
        run_single(args.run_single[0], args.run_single[1], Path(args.run_single[2]))
        return

    wav_dir = Path(tempfile.mkdtemp(prefix="stenograf_bench_"))
    print(f"Синтез {len(PHRASES)} фраз (Milena) в {wav_dir}...")
    synthesize(wav_dir)

    rows = []
    for engine, model in COMBOS:
        label = f"{engine}/{model}"
        print(f"Тест {label}... ", end="", flush=True)
        proc = subprocess.run(
            [sys.executable, __file__, "--run-single", engine, model, str(wav_dir)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"ошибка:\n{proc.stderr.strip().splitlines()[-1]}")
            continue
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        errors = words = 0
        worst = (0.0, "", "")
        for reference, hypothesis in zip(PHRASES, data["texts"]):
            e, n = word_errors(reference, hypothesis)
            errors += e
            words += n
            if n and e / n > worst[0]:
                worst = (e / n, reference, hypothesis)
        rows.append({
            "label": label, "wer": errors / words, "errors": errors, "words": words,
            "xrt": data["audio_s"] / data["infer_s"], "load_s": data["load_s"],
            "ram_mb": data["rss_delta_mb"], "worst": worst,
        })
        print(f"WER {errors / words:.1%}, {data['audio_s'] / data['infer_s']:.1f}× реального времени")

    print(f"\n{'модель':22s} {'WER':>7s} {'ошибки':>8s} {'скорость':>10s} {'загрузка':>9s} {'RAM':>9s}")
    for r in sorted(rows, key=lambda r: r["wer"]):
        print(f"{r['label']:22s} {r['wer']:7.1%} {r['errors']:4d}/{r['words']:<3d} "
              f"{r['xrt']:8.1f}×RT {r['load_s']:8.1f}с {r['ram_mb']:7.0f}МБ")

    print("\nХарактерные ошибки (худшая фраза каждой модели):")
    for r in sorted(rows, key=lambda r: -r["worst"][0]):
        share, reference, hypothesis = r["worst"]
        if share == 0:
            continue
        print(f"  {r['label']}:\n    эталон:     {reference}\n    распознано: {hypothesis}")
    print("\nWER — доля ошибочных слов (меньше — лучше); скорость — во сколько раз"
          "\nбыстрее реального времени обрабатывается речь; RAM — прирост памяти"
          "\nпроцесса после загрузки модели и прогона.")


if __name__ == "__main__":
    main()
