"""Генератор отчёта о тестировании: числа считаются, а не вписываются руками.

Зачем: раньше покрытие и число тестов жили в TEST_REPORT.md как текст. Проверить
их было нельзя — «красивое число для отчёта». Теперь отчёт целиком генерируется
этим скриптом из машинных артефактов:

    JUnit XML   -> сколько тестов, сколько прошло/упало/пропущено, сколько длилось
    coverage    -> покрытие строк и ВЕТОК, по каждому модулю и по каждому уровню

Каждый уровень тестов гоняется отдельным прогоном в свою базу покрытия, поэтому
видно вклад уровня. Потом базы складываются (coverage combine) — так модельный
код, который исполняется только e2e-тестами, попадает в общий процент.

Запуск:
    .venv/bin/python scripts/test_report.py              # быстрые уровни
    .venv/bin/python scripts/test_report.py --all        # + integration/e2e/load (минуты)
    .venv/bin/python scripts/test_report.py --levels unit,api
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = SERVER_DIR.parent
REPORTS = SERVER_DIR / "reports"


@dataclass(frozen=True)
class Level:
    key: str
    title: str
    what: str            # что этот уровень проверяет — идёт в отчёт
    args: list[str]      # аргументы pytest (пути и/или -m)
    slow: bool = False   # требует моделей/живого сервера — не в быстром наборе


# Уровни намеренно описаны честно: test_api.py гоняется через FastAPI TestClient
# внутри процесса и засеивает данные через crud, поэтому это компонентные тесты
# API-слоя, а не функциональные «чёрным ящиком». Функциональную проверку даёт e2e.
LEVELS: list[Level] = [
    Level(
        "unit", "Юнит",
        "одна функция или класс в изоляции, зависимости замоканы",
        ["tests", "--ignore=tests/test_api.py",
         "--ignore=tests/test_migrations.py",
         "--ignore=tests/test_diarization_integration.py",
         "-m", "not integration and not e2e and not load"],
    ),
    Level(
        "api", "Компонентные (API-слой)",
        "REST-эндпоинты через FastAPI TestClient в одном процессе: коды ответов и форма данных",
        ["tests/test_api.py", "-m", "not integration and not e2e and not load"],
    ),
    Level(
        "migrations", "Интеграционные (БД)",
        "миграции схемы на реальном файле SQLite: старая база → апгрейд без потери данных",
        ["tests/test_migrations.py", "-m", "not integration and not e2e and not load"],
    ),
    Level(
        "integration", "Интеграционные (модели)",
        "реальная ECAPA на синтезированных голосах macOS",
        ["tests/test_diarization_integration.py", "-m", "integration"],
        slow=True,
    ),
    Level(
        "e2e", "E2E (системные)",
        "живой uvicorn + WebSocket-клиент: встреча целиком от аудио до REST",
        ["tests/test_e2e_live.py", "-m", "e2e"],
        slow=True,
    ),
    Level(
        "load", "Нагрузочные",
        "параллельные встречи, замер задержки и памяти",
        ["tests/test_load_smoke.py", "-m", "load"],
        slow=True,
    ),
]


@dataclass
class Result:
    level: Level
    ran: bool = False
    skipped_reason: str = ""
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration_s: float = 0.0
    coverage: float | None = None       # вклад уровня в покрытие, %
    cases: list[tuple[str, str, str]] = field(default_factory=list)  # (файл, тест, статус)

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.errors == 0


def _run_level(level: Level, verbose: bool) -> Result:
    """Гоняет один уровень: свой JUnit XML и своя база покрытия."""
    result = Result(level=level)
    junit = REPORTS / "junit" / f"{level.key}.xml"
    cov_dir = REPORTS / "cov" / level.key
    shutil.rmtree(cov_dir, ignore_errors=True)
    cov_dir.mkdir(parents=True, exist_ok=True)
    junit.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ | {
        "COVERAGE_FILE": str(cov_dir / ".coverage"),
        "COVERAGE_PROCESS_START": str(SERVER_DIR / ".coveragerc"),
    }
    cmd = [
        sys.executable, "-m", "pytest", *level.args,
        f"--junitxml={junit}",
        "--cov=app", "--cov-report=", "-q",
    ]
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=SERVER_DIR, env=env,
                          capture_output=not verbose, text=True)
    result.duration_s = time.perf_counter() - started
    result.ran = True

    if not junit.exists():
        result.ran = False
        tail = (proc.stdout or "")[-400:] if not verbose else ""
        result.skipped_reason = f"pytest не создал отчёт (код {proc.returncode}). {tail}"
        return result

    _parse_junit(junit, result)
    result.coverage = _coverage_percent(cov_dir)
    return result


def _parse_junit(path: Path, result: Result) -> None:
    """Числа берём из машинного отчёта pytest, а не из его текстового вывода."""
    root = ET.parse(path).getroot()
    suites = root.findall("testsuite") or [root]
    for suite in suites:
        result.total += int(suite.get("tests", 0))
        result.failed += int(suite.get("failures", 0))
        result.errors += int(suite.get("errors", 0))
        result.skipped += int(suite.get("skipped", 0))
        for case in suite.iter("testcase"):
            if case.find("failure") is not None:
                status = "упал"
            elif case.find("error") is not None:
                status = "ошибка"
            elif case.find("skipped") is not None:
                status = "пропущен"
            else:
                status = "прошёл"
            result.cases.append((case.get("file", ""), case.get("name", ""), status))
    result.passed = result.total - result.failed - result.errors - result.skipped


def _coverage_percent(cov_dir: Path) -> float | None:
    """Покрытие одного уровня: combine внутри его папки + coverage json."""
    data_files = list(cov_dir.glob(".coverage*"))
    if not data_files:
        return None
    env = os.environ | {"COVERAGE_FILE": str(cov_dir / ".coverage")}
    subprocess.run([sys.executable, "-m", "coverage", "combine"],
                   cwd=SERVER_DIR, env=env, capture_output=True, text=True)
    out = cov_dir / "coverage.json"
    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "json", "-o", str(out), "-q"],
        cwd=SERVER_DIR, env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out.exists():
        return None
    return json.loads(out.read_text())["totals"]["percent_covered"]


def _combine_all(results: list[Result]) -> dict | None:
    """Складывает базы всех уровней в одну: так модельный код, который трогают
    только e2e-тесты, попадает в общий процент."""
    all_dir = REPORTS / "cov" / "_combined"
    shutil.rmtree(all_dir, ignore_errors=True)
    all_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for result in results:
        source = REPORTS / "cov" / result.level.key / ".coverage"
        if source.exists():
            shutil.copy(source, all_dir / f".coverage.{result.level.key}")
            copied += 1
    if not copied:
        return None

    env = os.environ | {"COVERAGE_FILE": str(all_dir / ".coverage")}
    for args in (
        ["combine"],
        ["html", "-d", str(REPORTS / "htmlcov")],
        ["xml", "-o", str(REPORTS / "coverage.xml")],
        ["json", "-o", str(all_dir / "coverage.json")],
    ):
        subprocess.run([sys.executable, "-m", "coverage", *args],
                       cwd=SERVER_DIR, env=env, capture_output=True, text=True)
    data = all_dir / "coverage.json"
    return json.loads(data.read_text()) if data.exists() else None


def _client_tests() -> dict | None:
    """Клиентские тесты (Vitest) — если установлен node_modules."""
    client = REPO_DIR / "client"
    if not (client / "node_modules").exists():
        return None
    proc = subprocess.run(
        ["npm", "run", "--silent", "test", "--", "--reporter=json"],
        cwd=client, capture_output=True, text=True,
    )
    raw = proc.stdout[proc.stdout.find("{"):] if "{" in proc.stdout else ""
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return {
        "total": data.get("numTotalTests", 0),
        "passed": data.get("numPassedTests", 0),
        "failed": data.get("numFailedTests", 0),
    }


def _bar(percent: float, width: int = 22) -> str:
    filled = round(percent / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _render(results: list[Result], totals: dict | None, client: dict | None,
            full: bool) -> str:
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    ran = [r for r in results if r.ran]
    total = sum(r.total for r in ran)
    passed = sum(r.passed for r in ran)
    failed = sum(r.failed + r.errors for r in ran)
    skipped = sum(r.skipped for r in ran)
    duration = sum(r.duration_s for r in ran)

    lines = [
        "# Отчёт о тестировании — Стенограф",
        "",
        "> ⚙️ **Файл сгенерирован автоматически, руками не редактируется.**",
        f"> Сгенерирован: {now} · команда: `python scripts/test_report.py"
        f"{' --all' if full else ''}`",
        f"> Окружение: Python {platform.python_version()}, {platform.system()} "
        f"{platform.machine()}",
        "",
        "Все числа ниже посчитаны из машинных артефактов, а не вписаны вручную:",
        "",
        "| Что | Откуда берётся | Артефакт |",
        "|---|---|---|",
        "| Число тестов, прошло/упало | JUnit XML от pytest | `server/reports/junit/*.xml` |",
        "| Покрытие строк и ветвей | coverage.py по `.coveragerc` | `server/reports/htmlcov/index.html` |",
        "| Покрытие для CI | Cobertura XML | `server/reports/coverage.xml` |",
        "",
        "## Итог",
        "",
        f"**{total} тестов · прошло {passed} · упало {failed} · пропущено {skipped}** "
        f"(за {duration:.1f} c)",
        "",
    ]
    if totals:
        pct = totals["totals"]["percent_covered"]
        t = totals["totals"]
        lines += [
            f"**Покрытие серверного кода: {pct:.1f}%** `{_bar(pct)}`",
            "",
            f"- строк: {t['covered_lines']} из {t['num_statements']} "
            f"(не покрыто {t['missing_lines']})",
            f"- ветвей: {t['covered_branches']} из {t['num_branches']} "
            f"(частично пройдено {t['num_partial_branches']})",
            "",
        ]
    if not full:
        lines += [
            "> ⚠️ Прогон быстрый: интеграционные, e2e и нагрузочные уровни не запускались,"
            " их вклад в покрытие здесь не учтён. Полный прогон: `--all`.",
            "",
        ]

    lines += [
        "## По уровням",
        "",
        "| Уровень | Что проверяет | Тестов | Прошло | Упало | Время | Вклад в покрытие |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if not r.ran:
            lines.append(
                f"| **{r.level.title}** | {r.level.what} | — | — | — | — | "
                f"не запускался |"
            )
            continue
        cov = f"{r.coverage:.1f}%" if r.coverage is not None else "—"
        mark = "" if r.ok else " ⚠️"
        lines.append(
            f"| **{r.level.title}**{mark} | {r.level.what} | {r.total} | {r.passed} | "
            f"{r.failed + r.errors} | {r.duration_s:.1f} c | {cov} |"
        )
    if client:
        lines.append(
            f"| **Клиент (Vitest)** | чистая логика фронтенда в jsdom | "
            f"{client['total']} | {client['passed']} | {client['failed']} | — | не мерится |"
        )
    lines += [
        "",
        "«Вклад в покрытие» — сколько даёт **только этот** уровень, если запустить его "
        "одного. Уровни перекрываются, поэтому сумма столбца больше итога — итог "
        "считается объединением баз (`coverage combine`), а не сложением процентов.",
        "",
    ]

    if totals:
        lines += ["## Покрытие по модулям", "",
                  "| Модуль | Строк | Не покрыто | Ветвей | Покрытие |",
                  "|---|---:|---:|---:|---:|"]
        files = sorted(totals["files"].items(),
                       key=lambda kv: kv[1]["summary"]["percent_covered"])
        for name, info in files:
            s = info["summary"]
            if not s["num_statements"]:
                continue
            lines.append(
                f"| `{name}` | {s['num_statements']} | {s['missing_lines']} | "
                f"{s['num_branches']} | {s['percent_covered']:.1f}% |"
            )
        lines.append("")

    lines += [
        "## Как это считается",
        "",
        "**Покрытие ветвей, а не только строк.** В `.coveragerc` включён `branch = True`. "
        "Это принципиально: объявления полей (`Settings`, модели БД) исполняются при "
        "импорте модуля и раньше засчитывались как «покрытые», раздувая процент. "
        "Ветвей у объявлений нет, поэтому число стало честнее.",
        "",
        "**Каждый уровень меряется отдельно, итог — объединение.** Уровень гоняется "
        "в свою базу покрытия (`reports/cov/<уровень>/`), потом базы складываются "
        "`coverage combine`. Так код, который исполняется только e2e-тестами "
        "(конвейер ASR, WebSocket-сессия), попадает в общий процент — при замере "
        "одним быстрым набором он выглядел бы непокрытым.",
        "",
        "**Что исключено из замера** (`.coveragerc`): `app/__init__.py`, заглушки "
        "`...` у Protocol-классов, ветки `if TYPE_CHECKING`, строки с "
        "`pragma: no cover`. Всё остальное считается.",
        "",
        "**Чего покрытие не показывает.** Это метрика исполненного кода, а не "
        "качества проверок: строку можно исполнить, ничего не утверждая о результате. "
        "Соответствие требованиям измеряется отдельно — см. каталог тестов "
        "[TESTS.md](TESTS.md), где у каждого теста написано, что именно он проверяет.",
        "",
        "## Как воспроизвести",
        "",
        "```bash",
        "cd server",
        ".venv/bin/python scripts/test_report.py --all",
        "```",
        "",
        "Открыть построчный отчёт: `server/reports/htmlcov/index.html` — там видно, "
        "какие именно строки и ветки не покрыты.",
        "",
        "Ручной чек-лист для того, что автоматика проверить не может (микрофон, эхо "
        "колонок, упакованное приложение) — [TESTING.md](TESTING.md).",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Отчёт о тестировании Стенографа")
    ap.add_argument("--all", action="store_true",
                    help="включая медленные уровни (модели, живой сервер, нагрузка)")
    ap.add_argument("--levels", help="через запятую: " + ",".join(l.key for l in LEVELS))
    ap.add_argument("--verbose", action="store_true", help="показывать вывод pytest")
    ap.add_argument("--out", default=str(REPO_DIR / "TEST_REPORT.md"))
    args = ap.parse_args()

    chosen = LEVELS
    if args.levels:
        keys = {k.strip() for k in args.levels.split(",")}
        chosen = [l for l in LEVELS if l.key in keys]
    elif not args.all:
        chosen = [l for l in LEVELS if not l.slow]

    REPORTS.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []
    for level in LEVELS:
        if level not in chosen:
            results.append(Result(level=level, skipped_reason="не выбран"))
            continue
        print(f"→ {level.title}…", flush=True)
        result = _run_level(level, args.verbose)
        status = "ok" if result.ok else "ЕСТЬ ПАДЕНИЯ"
        print(f"  {result.total} тестов, прошло {result.passed}, "
              f"упало {result.failed + result.errors} — {status}")
        results.append(result)

    totals = _combine_all(results)
    print("→ клиентские тесты…", flush=True)
    client = _client_tests()

    report = _render(results, totals, client, full=args.all)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"\nОтчёт: {args.out}")
    print(f"Покрытие (HTML): {REPORTS / 'htmlcov' / 'index.html'}")
    if totals:
        print(f"Итого покрытие: {totals['totals']['percent_covered']:.1f}%")

    if any(r.ran and not r.ok for r in results):
        sys.exit(1)  # падения тестов — ненулевой код возврата (пригодится в CI)


if __name__ == "__main__":
    main()
