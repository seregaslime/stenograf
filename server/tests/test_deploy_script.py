"""Скрипты в deploy/ обязаны работать в dash, а не только в bash.

Поймано 04.09.2026 на живом сервере: update.sh упал первой же строкой с
«КАТАЛОГ=/opt/stenograf: not found». На Ubuntu /bin/sh — это dash, и имена
переменных там только ASCII; на маке /bin/sh — bash, который кириллицу терпит,
поэтому локальная проверка `sh -n` прошла и дала ложную уверенность.

Тест живёт среди серверных, потому что другого набора в проекте нет, а
проверять надо: скрипт запускается ровно один раз в жизни каждой версии — на
сервере, руками, и обнаруживается сломанным в самый неподходящий момент.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parent.parent.parent / "deploy"
СКРИПТЫ = sorted(DEPLOY.glob("*.sh"))

# Имя переменной по POSIX: латиница, цифры, подчёркивание, не с цифры.
ИМЯ = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
ПРИСВАИВАНИЕ = re.compile(r"(?:^|\s|\()([^\s=]+)=")


def test_скрипты_нашлись():
    """Если каталог переименуют, тест должен упасть, а не тихо проверять пустоту."""
    assert СКРИПТЫ, f"в {DEPLOY} нет ни одного .sh — проверять нечего"


@pytest.mark.parametrize("скрипт", СКРИПТЫ, ids=lambda p: p.name)
def test_имена_переменных_ascii(скрипт: Path):
    нарушения = []
    for номер, строка in enumerate(скрипт.read_text(encoding="utf-8").splitlines(), 1):
        for имя in ПРИСВАИВАНИЕ.findall(строка.split("#")[0]):
            if not ИМЯ.fullmatch(имя):
                нарушения.append(f"{скрипт.name}:{номер} — {имя!r}")
    assert not нарушения, (
        "имена переменных вне ASCII не работают в dash (/bin/sh на Ubuntu): "
        + "; ".join(нарушения)
    )


@pytest.mark.parametrize("скрипт", СКРИПТЫ, ids=lambda p: p.name)
def test_синтаксис_в_dash(скрипт: Path):
    """Настоящая проверка тем самым интерпретатором. Пропускается, если dash
    не установлен — тогда остаётся проверка имён выше, которая ловит тот же
    класс ошибок без него."""
    dash = shutil.which("dash")
    if dash is None:
        pytest.skip("dash не установлен")
    результат = subprocess.run([dash, "-n", str(скрипт)], capture_output=True, text=True)
    assert результат.returncode == 0, результат.stderr
