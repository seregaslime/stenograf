"""Метрики нагрузочного харнесса (scripts/loadtest.py).

Быстрые юнит-тесты чистых функций: ни сервера, ни моделей, ни Docker.
Появились после того, как в отчёте обнаружился p95 МЕНЬШЕ медианы — наивная
формула индекса на малых выборках брала минимум вместо максимума.
"""
import sys
from pathlib import Path

import pytest

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR / "scripts"))

import loadtest  # noqa: E402


@pytest.mark.parametrize("n", range(1, 40))
def test_p95_never_below_median(n):
    """Главная гарантия: 95-й перцентиль не может оказаться ниже медианы."""
    values = [float(i) for i in range(n)]
    p95 = loadtest._percentile(values, 0.95)
    median = values[n // 2]
    assert p95 >= median


def test_p95_on_tiny_sample_takes_worst():
    """Регрессия: при двух замерах брался минимум, из-за чего p95 < p50."""
    assert loadtest._percentile([4.5, 5.2], 0.95) == 5.2
    assert loadtest._percentile([7.0], 0.95) == 7.0


def test_percentile_bounds():
    """Индекс не вылезает за границы списка ни при каком квантиле."""
    values = [1.0, 2.0, 3.0, 4.0]
    assert loadtest._percentile(values, 0.0) == 1.0
    assert loadtest._percentile(values, 1.0) == 4.0
    assert loadtest._percentile([], 0.95) == 0.0  # пустая выборка не роняет


def test_degraded_flags_container_death():
    """Перезапуск контейнера и OOM считаются отказом, даже если ошибок встреч нет."""
    healthy = {"meetings": 4, "errors": [], "total_segments": 4, "duration_p95_s": 10.0,
               "container_status": "running", "container_restarted": False,
               "container_oom": False}
    assert loadtest._degraded(healthy, limit_p95=60.0) is None

    assert loadtest._degraded({**healthy, "container_oom": True}, 60.0)
    assert loadtest._degraded({**healthy, "container_restarted": True}, 60.0)
    assert loadtest._degraded({**healthy, "container_status": "exited"}, 60.0)


def test_degraded_flags_errors_and_latency():
    """Ошибки встреч, отсутствие распознавания и превышение p95 — тоже отказ."""
    base = {"meetings": 4, "errors": [], "total_segments": 4, "duration_p95_s": 10.0,
            "container_status": "running", "container_restarted": False,
            "container_oom": False}
    assert "ошибки" in loadtest._degraded({**base, "errors": ["boom"]}, 60.0)
    assert "ни одной реплики" in loadtest._degraded({**base, "total_segments": 0}, 60.0)
    assert "p95" in loadtest._degraded({**base, "duration_p95_s": 99.0}, 60.0)


def test_degraded_catches_silent_transcript_loss():
    """Самый коварный отказ: встречи завершились штатно, но часть без транскрипта.

    На замерах явные ошибки плавали (18 отказов и 0 в повторе той же ступени),
    а недостача сегментов воспроизводилась стабильно — значит ловить надо её.
    """
    base = {"meetings": 130, "errors": [], "duration_p95_s": 10.0,
            "container_status": "running", "container_restarted": False,
            "container_oom": False}
    problem = loadtest._degraded({**base, "total_segments": 100}, 60.0)
    assert problem and "потеряны транскрипты" in problem and "30" in problem
    # ровно по сегменту на встречу — это норма
    assert loadtest._degraded({**base, "total_segments": 130}, 60.0) is None
    # сегментов больше, чем встреч (длинная речь режется на части) — тоже норма
    assert loadtest._degraded({**base, "total_segments": 190}, 60.0) is None


def test_degraded_without_container_metrics():
    """Прогон против своего сервера: полей контейнера нет — не должно падать."""
    metrics = {"meetings": 2, "errors": [], "total_segments": 2, "duration_p95_s": 5.0}
    assert loadtest._degraded(metrics, limit_p95=60.0) is None
