"""Бенчмарк роста библиотеки голосов (scripts/bench_registry.py).

Тесты не меряют производительность (это дело самого бенчмарка), а проверяют,
что он меряет ТО, что заявляет: полный перебор отпечатков без обращения к БД.
Иначе бенчмарк тихо перестанет отражать реальность, и вывод «диаризация не
узкое место» окажется основан на сломанном замере.
"""
import sys
from pathlib import Path

import numpy as np

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR / "scripts"))

import bench_registry  # noqa: E402


def test_fill_creates_requested_library():
    rng = np.random.default_rng(1)
    registry = bench_registry.SpeakerRegistry(bench_registry.Settings(_env_file=None))
    bench_registry._fill(registry, speakers=7, prints_each=3, rng=rng)
    assert len(registry._prints) == 7
    assert all(len(p) == 3 for p in registry._prints.values())


def test_embeddings_are_unit_vectors():
    """match_all считает скалярное произведение как косинусную близость —
    значит векторы обязаны быть нормированы, иначе пороги теряют смысл."""
    rng = np.random.default_rng(2)
    for _ in range(20):
        assert abs(np.linalg.norm(bench_registry._unit(rng)) - 1.0) < 1e-5


def test_probe_matches_existing_speaker():
    """Проба должна УЗНАВАТЬСЯ: со случайным вектором match_all пошла бы
    заводить нового спикера, и замер мерил бы не перебор, а запись в БД."""
    rng = np.random.default_rng(3)
    registry = bench_registry.SpeakerRegistry(bench_registry.Settings(_env_file=None))
    bench_registry._fill(registry, speakers=5, prints_each=2, rng=rng)
    probe = bench_registry._probe_matching(registry, rng)

    best = max(
        float(np.dot(pr.vector, probe))
        for prints in registry._prints.values() for pr in prints
    )
    assert best > 0.9  # заведомо выше рабочего порога 0.35


def test_measure_reports_linear_growth():
    """Главная проверка: время растёт пропорционально размеру библиотеки.

    Если кто-то заменит перебор на индекс, тест упадёт — и это будет
    правильный сигнал обновить выводы в LOAD_REPORT.md, а не молча тащить старые.
    """
    rng = np.random.default_rng(4)
    small = bench_registry.measure(20, 5, samples=40, rng=rng)
    large = bench_registry.measure(200, 5, samples=40, rng=rng)

    assert large["total_prints"] == 10 * small["total_prints"]
    assert large["p50_ms"] > small["p50_ms"]
    # Стоимость одного отпечатка примерно постоянна — признак линейности
    assert 0.3 < large["us_per_print"] / small["us_per_print"] < 3.0


def test_measure_does_not_touch_real_db():
    """Заглушка БД обязана перехватывать все обращения: если бенчмарк однажды
    начнёт писать в рабочую базу, это заметят слишком поздно."""
    db = bench_registry._StubDb()
    row = db.get(object, 123)
    assert row.name and hasattr(row, "centroid") and hasattr(row, "embedding_count")
