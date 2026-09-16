"""Интеграционный тест: настоящий silero-VAD на синтезированной речи.

Запуск (грузит модель VAD, нужен `say` — только macOS):
    .venv/bin/python -m pytest -m integration

Закрепляет порог тишины vad_min_silence_ms с двух сторон: первый тест падает,
если порог поднять (двое снова склеятся), второй — если опустить (монолог
порвётся). Вместе они держат число там, где его поставил замер.

Функции взяты из scripts/eval_split.py, а не повторены: тест и замер должны
мерить одно и то же. Иначе число в комментарии к порогу и зелёный тест
однажды разойдутся, и никто этого не заметит.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))
import eval_split  # noqa: E402

from app.config import Settings  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def голоса(tmp_path_factory):
    if shutil.which("say") is None:
        pytest.skip("нет команды say (не macOS)")
    папка = tmp_path_factory.mktemp("split")
    return {
        "первый": eval_split.синтез(папка, *eval_split.ПЕРВЫЙ),
        "второй": eval_split.синтез(папка, *eval_split.ВТОРОЙ),
        "монолог": eval_split.синтез(папка, *eval_split.МОНОЛОГ),
    }


@pytest.fixture(scope="module")
def порог() -> int:
    return Settings(_env_file=None).vad_min_silence_ms


def test_двое_с_паузой_200_мс_становятся_двумя_репликами(голоса, порог):
    """Смена говорящего с паузой 200 мс режется.

    При прежних 300 мс эта пауза концом фразы не считалась: второй человек
    приклеивался к первому, и реплику целиком забирал тот, кто говорил дольше.
    """
    смесь = np.concatenate([голоса["первый"], eval_split.тишина(200), голоса["второй"]])
    assert eval_split.реплик(смесь, порог) == 2


def test_монолог_одного_человека_не_рвётся(голоса, порог):
    """Паузы между предложениями внутри монолога — не конец реплики.

    При 150 мс монолог из четырёх предложений разваливается на четыре куска, а
    короткий кусок даёт худший отпечаток голоса: лечили бы склейку, ухудшая
    узнавание.
    """
    assert eval_split.реплик(голоса["монолог"], порог) == 1
