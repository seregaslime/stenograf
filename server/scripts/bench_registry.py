"""Как растёт время опознания голоса с ростом библиотеки спикеров.

Это единственное место в конвейере с алгоритмической сложностью: match_all
(diarization/registry.py) перебирает ВСЕХ спикеров и ВСЕ их отпечатки, считая
косинусную близость на каждый. Сложность O(спикеры × отпечатки), то есть на
сервере организации, копящем голоса сотрудников, каждая реплика со временем
обрабатывается всё дольше.

Важно: докупкой памяти это не лечится — библиотека на 1000 человек весит 3.7 МБ,
упирается процессор.

Замер (Apple M3, 1000 спикеров × 5 отпечатков): рост строго линейный, но
константа крошечная — 1.8 мс на реплику против ~1400 мс у ASR, то есть 0.1%
стоимости обработки. Чтобы диаризация стала заметной статьёй расхода, нужно
порядка 75 000 спикеров. На горизонте продукта это не узкое место, и бенчмарк
нужен ровно чтобы это утверждение можно было перепроверить, а не принимать на веру.

Бенчмарк синтетический: эмбеддинги случайные, ECAPA не грузится, аудио не
нужно. Меряется чистая стоимость перебора.

Запуск:
    .venv/bin/python scripts/bench_registry.py
    .venv/bin/python scripts/bench_registry.py --sizes 100,500,2000 --json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

from app.config import Settings  # noqa: E402
from app.diarization.registry import SpeakerRegistry, _Print  # noqa: E402

EMBED_DIM = 192  # размерность ECAPA-TDNN


def _unit(rng: np.random.Generator) -> np.ndarray:
    """Случайный единичный вектор: match_all считает скалярное произведение,
    значит векторы должны быть нормированы, как настоящие эмбеддинги."""
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _fill(registry: SpeakerRegistry, speakers: int, prints_each: int, rng) -> None:
    """Набивает библиотеку, минуя БД: нам нужна только стоимость перебора."""
    registry._prints = {
        speaker_id: [
            _Print(id=speaker_id * 100 + k, vector=_unit(rng), count=3)
            for k in range(prints_each)
        ]
        for speaker_id in range(1, speakers + 1)
    }
    registry._self_id = 1


class _StubRow:
    """Заменяет и строку отпечатка (её центроид обновляют при совпадении),
    и строку спикера (у неё спрашивают имя для результата)."""
    centroid = b""
    embedding_count = 0
    name = "Спикер"


class _StubDb:
    """Заглушка сессии: меряем стоимость ПЕРЕБОРА, а не запись в БД.

    При узнавании голоса match_all обновляет центроид отпечатка (_update_print)
    и достаёт имя спикера — для обоих нужен db.get. Это константа на вызов, от
    размера библиотеки не зависит и форму кривой роста не искажает.
    """

    def __init__(self):
        self._row = _StubRow()

    def get(self, model, pk):
        return self._row


def _probe_matching(registry: SpeakerRegistry, rng) -> np.ndarray:
    """Проба, похожая на уже известный голос.

    Со случайным вектором match_all не нашла бы совпадения и полезла бы
    заводить нового спикера — то есть в БД, которой в бенчмарке нет. Берём
    существующий отпечаток с лёгким шумом: близость высокая → узнавание без записи,
    а полный перебор всех отпечатков всё равно выполняется, его и меряем.
    """
    speaker_ids = list(registry._prints)
    donor = registry._prints[speaker_ids[rng.integers(len(speaker_ids))]][0]
    noisy = donor.vector + 0.01 * rng.standard_normal(EMBED_DIM).astype(np.float32)
    return noisy / np.linalg.norm(noisy)


def measure(speakers: int, prints_each: int, samples: int, rng) -> dict:
    cfg = Settings(_env_file=None)
    registry = SpeakerRegistry(cfg)
    _fill(registry, speakers, prints_each, rng)
    db = _StubDb()

    # Прогрев: первый вызов включает ленивые аллокации numpy
    registry.match_all(db=db, embedding=_probe_matching(registry, rng), mic_dominant=False)

    timings = []
    for _ in range(samples):
        probe = _probe_matching(registry, rng)
        started = time.perf_counter()
        registry.match_all(db=db, embedding=probe, mic_dominant=False)
        timings.append((time.perf_counter() - started) * 1000)

    total_prints = speakers * prints_each
    return {
        "speakers": speakers,
        "prints_each": prints_each,
        "total_prints": total_prints,
        "p50_ms": round(statistics.median(timings), 3),
        "p95_ms": round(sorted(timings)[max(0, int(len(timings) * 0.95) - 1)], 3),
        "max_ms": round(max(timings), 3),
        "us_per_print": round(statistics.median(timings) * 1000 / total_prints, 2),
        "library_kb": round(total_prints * EMBED_DIM * 4 / 1024, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Рост времени опознания голоса")
    ap.add_argument("--sizes", default="10,50,100,250,500,1000",
                    help="через запятую: сколько спикеров в библиотеке")
    ap.add_argument("--prints", type=int, default=5,
                    help="отпечатков на спикера (по умолчанию потолок из конфига)")
    ap.add_argument("--samples", type=int, default=200, help="замеров на размер")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(20260731)  # фиксируем — прогоны сравнимы между собой
    sizes = [int(s) for s in args.sizes.split(",")]
    rows = [measure(n, args.prints, args.samples, rng) for n in sizes]

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    print(f"\nОпознание одного голоса, отпечатков на спикера: {args.prints}")
    print(f"{'спикеров':>9} {'отпечатков':>11} {'p50':>9} {'p95':>9} "
          f"{'мкс/отпечаток':>14} {'память':>9}")
    print("-" * 68)
    for r in rows:
        print(f"{r['speakers']:>9} {r['total_prints']:>11} {r['p50_ms']:>8.3f}м "
              f"{r['p95_ms']:>8.3f}м {r['us_per_print']:>14.2f} {r['library_kb']:>7.1f}К")

    first, last = rows[0], rows[-1]
    grew_prints = last["total_prints"] / first["total_prints"]
    grew_time = last["p50_ms"] / first["p50_ms"] if first["p50_ms"] else 0
    print(f"\nБиблиотека выросла в {grew_prints:.0f}×, время опознания — в {grew_time:.0f}×.")
    print(f"Память библиотеки при {last['speakers']} спикерах: "
          f"{last['library_kb'] / 1024:.1f} МБ — упирается процессор, не RAM.")


if __name__ == "__main__":
    main()
