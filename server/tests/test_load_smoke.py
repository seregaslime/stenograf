"""Дымовой нагрузочный тест (-m load, исключён из дефолтного прогона): две
параллельные встречи проходят целиком без ошибок. Полноценный прогон — скриптом
scripts/loadtest.py. Требует кэш моделей server/data/models."""
import asyncio
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.load

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR / "scripts"))


def test_two_concurrent_meetings_survive():
    """Две одновременные встречи проходят целиком без ошибок, память измерима — дымовая проверка перед полноценной нагрузкой.
    """
    if not (SERVER_DIR / "data" / "models").exists():
        pytest.skip("нет кэша моделей server/data/models — сначала запустите сервер")
    import loadtest

    metrics = asyncio.run(loadtest.run_load(meetings=2, seconds=4, port=8771))
    assert metrics["errors"] == [], metrics["errors"]
    assert metrics["peak_rss_mb"] > 0
    assert metrics["total_segments"] >= 1  # обе встречи что-то распознали
