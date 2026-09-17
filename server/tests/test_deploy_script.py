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


# --- смена образа базы: update.sh не подменяет его молча, move-db.sh переносит ---
#
# Docker подменён скриптом, который пишет каждый вызов в журнал и отвечает то,
# что задано в окружении. Настоящий перенос на Docker в CI не прогнать, а
# проверить надо главное: в каком порядке идут необратимые шаги и что они не
# случаются, когда не должны.

ПОДДЕЛЬНЫЙ_DOCKER = r"""#!/bin/sh
echo "$*" >> "$DOCKER_LOG"
case "$*" in
  "compose config") printf 'services:\n  db:\n    environment:\n      POSTGRES_DB: stenograf\n    image: %s\n  server:\n    image: ghcr.io/seregaslime/stenograf-server:latest\n' "$WANT_DB" ;;
  "compose ps -q db") [ -n "$HAVE_DB" ] && echo db-container ;;
  inspect*Config.Image*) echo "$HAVE_DB" ;;
  inspect*Mounts*) echo stenograf_stenograf-db ;;
  *pg_dump*) printf '%s' "$DUMP_TEXT" ;;
  *psql*) cat > /dev/null ;;
esac
exit 0
"""


def прогнать(tmp_path: Path, скрипт: str, have: str, want: str, дамп: str = "-- дамп\n"):
    """Скрипт из deploy/ в отдельном каталоге: он пишет backups/ рядом с собой."""
    (tmp_path / "deploy").mkdir()
    shutil.copy(DEPLOY / скрипт, tmp_path / "deploy" / скрипт)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for имя, текст in {"docker": ПОДДЕЛЬНЫЙ_DOCKER, "git": "#!/bin/sh\nexit 0\n",
                       "curl": "#!/bin/sh\necho '{\"status\":\"ok\"}'\n"}.items():
        (bin_ / имя).write_text(текст)
        (bin_ / имя).chmod(0o755)
    журнал = tmp_path / "docker.log"
    журнал.touch()
    оболочка = shutil.which("dash") or "sh"
    результат = subprocess.run(
        [оболочка, str(tmp_path / "deploy" / скрипт)], capture_output=True, text=True,
        env={"PATH": f"{bin_}:/usr/bin:/bin", "DOCKER_LOG": str(журнал),
             "HAVE_DB": have, "WANT_DB": want, "DUMP_TEXT": дамп},
    )
    return результат, журнал.read_text().splitlines()


def test_update_не_ставит_новый_образ_базы_поверх_старого_тома(tmp_path):
    результат, вызовы = прогнать(tmp_path, "update.sh", "postgres:17-alpine", "pgvector/pgvector:pg17")
    assert результат.returncode == 1
    assert "move-db.sh" in результат.stdout
    assert not any(в.startswith("compose up") or в == "compose pull server" for в in вызовы)


def test_update_идёт_как_обычно_когда_образ_базы_тот_же(tmp_path):
    образ = "pgvector/pgvector:pg17"
    результат, вызовы = прогнать(tmp_path, "update.sh", образ, образ)
    assert результат.returncode == 0, результат.stdout + результат.stderr
    assert "compose up -d --remove-orphans server" in вызовы


def test_перенос_сначала_дамп_потом_необратимое(tmp_path):
    """Порядок — вся суть скрипта: том удаляется только после дампа и копии,
    дамп льётся только в новую базу."""
    результат, вызовы = прогнать(tmp_path, "move-db.sh", "postgres:17-alpine", "pgvector/pgvector:pg17")
    assert результат.returncode == 0, результат.stdout + результат.stderr

    def где(начало: str) -> int:
        return next(i for i, в in enumerate(вызовы) if в.startswith(начало))

    assert (где("compose exec -T db pg_dump") < где("compose stop server db")
            < где("volume create stenograf_stenograf-db-before-")
            < где("volume rm stenograf_stenograf-db") < где("compose up -d db")
            < где("compose exec -T db psql"))
    [дамп] = (tmp_path / "backups").glob("move-*.sql")
    assert дамп.read_text() == "-- дамп\n"


def test_пустой_дамп_останавливает_перенос_до_удаления_тома(tmp_path):
    результат, вызовы = прогнать(tmp_path, "move-db.sh", "postgres:17-alpine",
                                 "pgvector/pgvector:pg17", дамп="")
    assert результат.returncode == 1
    assert not any(в.startswith(("compose stop", "compose rm", "volume rm")) for в in вызовы)


def test_перенос_ничего_не_трогает_если_образ_уже_новый(tmp_path):
    образ = "pgvector/pgvector:pg17"
    результат, вызовы = прогнать(tmp_path, "move-db.sh", образ, образ)
    assert результат.returncode == 0
    assert not any("pg_dump" in в or в.startswith(("compose stop", "volume")) for в in вызовы)
