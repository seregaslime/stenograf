"""Каталог тестов: что где лежит и что именно проверяет.

Обходит тестовые файлы через AST и достаёт у каждого теста его описание
(докстринг, а если его нет — комментарий над функцией или сама формулировка
имени). Результат — TESTS.md, справочник на все тесты проекта.

Смысл в том, чтобы каталог нельзя было «забыть обновить»: он генерируется из
кода, а не пишется рядом с ним.

Запуск:
    .venv/bin/python scripts/test_inventory.py
"""
from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = SERVER_DIR.parent

# Уровень определяется по файлу — те же группы, что в scripts/test_report.py
LEVELS: dict[str, tuple[str, str]] = {
    "test_e2e_live.py": ("E2E (системные)",
                         "живой uvicorn + WebSocket: встреча целиком, как у пользователя"),
    "test_load_smoke.py": ("Нагрузочные", "параллельные встречи, задержка и память"),
    "test_diarization_integration.py": ("Интеграционные (модели)",
                                        "реальная ECAPA на синтезированных голосах"),
    "test_migrations.py": ("Интеграционные (БД)",
                           "миграции схемы на реальном файле SQLite"),
    "test_api.py": ("Компонентные (API-слой)",
                    "REST через FastAPI TestClient в одном процессе"),
}
UNIT = ("Юнит", "функция или класс в изоляции, зависимости замоканы")

# Что проверяет каждый файл — одной строкой для оглавления
FILE_ABOUT: dict[str, str] = {
    "test_registry.py": "диаризация: сопоставление голосов, приоры, слияние профилей",
    "test_mixer.py": "микшер каналов: склейка mic/system и определение доминанты",
    "test_vad.py": "нарезка речи по паузам, отбрасывание коротких фрагментов",
    "test_transcriber_junk.py": "фильтр галлюцинаций ASR на тишине",
    "test_crud.py": "операции с БД: встречи, спикеры, сегменты",
    "test_config.py": "конфигурация и персист выбора движка ASR",
    "test_auth.py": "токены: хеширование, заголовок, закрытые пути, заведение людей",
    "test_isolation.py": "разделение по владельцам: чужая встреча не видна и не правится",
    "test_summary_intake.py": "приём готового протокола от приложения и ошибки генерации",
    "test_search_vectors.py": "векторы от клиента: что индексировать, хранение, подбор ближайших",
    "test_merge_live.py": "слияние профилей во время встречи и уведомление владельцу",
    "test_deploy_script.py": "скрипт обновления сервера: POSIX-совместимость",
    "test_device.py": "выбор устройства для моделей: cuda/mps/cpu и откат",
    "test_bench_registry.py": "замер диаризации на эталоне голосов",
    "test_loadtest_metrics.py": "разбор метрик нагрузочного прогона",
    "test_live_session.py": "живая сессия: разрез сегментов по смене говорящего",
    "test_api.py": "REST-эндпоинты: коды ответов, форма данных, отказы",
    "test_migrations.py": "апгрейд старых баз без потери данных",
    "test_diarization_integration.py": "реальная модель ECAPA на синтезе голосов macOS",
    "test_e2e_live.py": "сквозные сценарии встречи через WebSocket",
    "test_load_smoke.py": "дымовой нагрузочный прогон",
}


@dataclass
class TestCase:
    name: str
    description: str
    params: int  # сколько кейсов даёт parametrize (0 — не параметризован)


@dataclass
class TestFile:
    path: Path
    level: str
    level_about: str
    module_doc: str
    cases: list[TestCase]


def _clean(text: str) -> str:
    """Докстринг в одну строку."""
    return re.sub(r"\s+", " ", text).strip()


def _humanize(name: str) -> str:
    """Из имени теста делаем человеческую формулировку — на случай отсутствия
    докстринга: test_skip_is_not_sent_to_client → «skip is not sent to client»."""
    return name.removeprefix("test_").replace("_", " ")


def _param_count(node: ast.FunctionDef) -> int:
    """Сколько кейсов добавляет @pytest.mark.parametrize."""
    total = 0
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        target = decorator.func
        if isinstance(target, ast.Attribute) and target.attr == "parametrize":
            for arg in decorator.args:
                if isinstance(arg, (ast.List, ast.Tuple)):
                    total += len(arg.elts)
    return total


def _collect(path: Path) -> TestFile:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_doc = _clean(ast.get_docstring(tree) or "")
    cases: list[TestCase] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        doc = _clean(ast.get_docstring(node) or "")
        cases.append(TestCase(node.name, doc or _humanize(node.name), _param_count(node)))
    cases.sort(key=lambda c: c.name)
    level, about = LEVELS.get(path.name, UNIT)
    return TestFile(path, level, about, module_doc, cases)


def _render(files: list[TestFile], client_files: list[TestFile]) -> str:
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    total = sum(len(f.cases) for f in files)
    total_cases = sum(
        sum(max(c.params, 1) for c in f.cases) for f in files
    )
    client_total = sum(len(f.cases) for f in client_files)

    lines = [
        "# Каталог тестов — Стенограф",
        "",
        "> ⚙️ **Сгенерирован автоматически** из кода тестов "
        "(`python scripts/test_inventory.py`), руками не редактируется.",
        f"> Обновлён: {now}",
        "",
        f"Всего тестовых функций на сервере: **{total}** "
        f"(с учётом параметризации — {total_cases} проверок). "
        f"Клиент: **{client_total}**.",
        "",
        "Здесь написано, **что именно** проверяет каждый тест. Покрытие кода "
        "(сколько строк исполнилось) — отдельная метрика, она в "
        "[TEST_REPORT.md](TEST_REPORT.md).",
        "",
        "## Оглавление",
        "",
    ]

    by_level: dict[str, list[TestFile]] = {}
    for f in files:
        by_level.setdefault(f.level, []).append(f)
    order = ["Юнит", "Компонентные (API-слой)", "Интеграционные (БД)",
             "Интеграционные (модели)", "E2E (системные)", "Нагрузочные"]
    ordered = [lvl for lvl in order if lvl in by_level]

    for level in ordered:
        group = sorted(by_level[level], key=lambda f: f.path.name)
        count = sum(len(f.cases) for f in group)
        lines.append(f"- **{level}** — {count} тестов")
        for f in group:
            about = FILE_ABOUT.get(f.path.name, "")
            anchor = f.path.name.replace(".", "").replace("_", "-")
            lines.append(f"  - [`{f.path.name}`](#{anchor}) — {about} ({len(f.cases)})")
    if client_files:
        lines.append(f"- **Клиент (Vitest)** — {client_total} тестов")
        for f in client_files:
            anchor = f.path.name.replace(".", "").replace("_", "-")
            lines.append(f"  - [`{f.path.name}`](#{anchor})")
    lines.append("")

    for level in ordered:
        group = sorted(by_level[level], key=lambda f: f.path.name)
        about = group[0].level_about
        lines += [f"## {level}", "", f"*{about}*", ""]
        for f in group:
            lines += _render_file(f)

    if client_files:
        lines += ["## Клиент (Vitest)", "",
                  "*чистая логика фронтенда в jsdom, без браузера*", ""]
        for f in client_files:
            lines += _render_file(f)

    lines += [
        "## Чего здесь нет",
        "",
        "UI-страницы (`LivePage`, `SettingsPage`, `SpeakersPage`) юнит-тестами не "
        "покрыты — они проверяются `tsc --noEmit`, сборкой `vite build` и ручным "
        "чек-листом [TESTING.md](TESTING.md). Там же то, что автоматика проверить "
        "не может в принципе: реальный микрофон, эхо колонок, упакованное приложение.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _render_file(f: TestFile) -> list[str]:
    rel = f.path.relative_to(REPO_DIR)
    lines = [f"### `{f.path.name}`", "", f"Путь: `{rel}`  "]
    if f.module_doc:
        lines.append(f"{f.module_doc}")
    lines += ["", "| Тест | Что проверяет |", "|---|---|"]
    for case in f.cases:
        suffix = f" _(×{case.params} кейсов)_" if case.params else ""
        lines.append(f"| `{case.name}` | {case.description}{suffix} |")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Каталог тестов Стенографа")
    ap.add_argument("--out", default=str(REPO_DIR / "TESTS.md"))
    args = ap.parse_args()

    server_files = [
        _collect(p) for p in sorted((SERVER_DIR / "tests").glob("test_*.py"))
    ]
    client_dir = REPO_DIR / "client" / "src"
    client_files = []
    if client_dir.exists():
        for p in sorted(client_dir.rglob("*.test.ts")):
            tf = _collect_ts(p)
            if tf.cases:
                client_files.append(tf)

    report = _render(server_files, client_files)
    Path(args.out).write_text(report, encoding="utf-8")
    total = sum(len(f.cases) for f in server_files) + sum(len(f.cases) for f in client_files)
    print(f"Каталог: {args.out}  ({total} тестов)")


_TS_TEST = re.compile(r"""\b(?:it|test)\s*\(\s*["'`](.+?)["'`]""", re.S)
_TS_DESCRIBE = re.compile(r"""\bdescribe\s*\(\s*["'`](.+?)["'`]""", re.S)


def _collect_ts(path: Path) -> TestFile:
    """Vitest-тесты: у них описание — это сама строка в it("...")."""
    text = path.read_text(encoding="utf-8")
    groups = _TS_DESCRIBE.findall(text)
    cases = [
        TestCase(name=_clean(m), description=_clean(m), params=0)
        for m in _TS_TEST.findall(text)
    ]
    return TestFile(path, "Клиент (Vitest)", "Vitest + jsdom",
                    "; ".join(groups), cases)


if __name__ == "__main__":
    main()
