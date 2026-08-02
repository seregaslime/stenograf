"""Обстрел REST-API параллельными запросами: сколько держит сервер.

Дополняет loadtest.py. Тот грузит тяжёлый путь (аудио → ASR), а этот — лёгкий:
много одновременных HTTP-запросов к эндпоинтам. Проверяется, что сервер под
шквалом отвечает корректно, не отдаёт 5xx и не отваливается по таймауту.

Бить лучше по контейнеру (`--url`), тогда рабочая машина остаётся свободной,
а лимит ресурсов задан в docker-compose.yml.

Запуск:
    docker compose up -d --no-deps server
    .venv/bin/python scripts/spamtest.py --url http://127.0.0.1:8765 --ramp

⚠ Эндпоинты, которые ходят во внешний LLM-API (summarize, /api/llm/models),
намеренно НЕ обстреливаются по умолчанию: это платные запросы к провайдеру и
верный способ упереться в его rate-limit. Для проверки лимитов провайдера есть
отдельный флаг --include-llm, включать осознанно.
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import httpx

# Только читающие и дешёвые эндпоинты: обстрел не должен портить данные.
SAFE_ENDPOINTS = [
    ("GET", "/api/health"),
    ("GET", "/api/asr"),
    ("GET", "/api/llm"),
    ("GET", "/api/meetings"),
    ("GET", "/api/speakers"),
]
# Ходят во внешний платный API — только по явному флагу.
LLM_ENDPOINTS = [
    ("POST", "/api/llm/models"),
]


async def _one_request(client: httpx.AsyncClient, method: str, path: str,
                       timeout: float) -> dict:
    started = time.perf_counter()
    try:
        if method == "GET":
            response = await client.get(path, timeout=timeout)
        else:
            response = await client.post(path, json={"api_base_url": ""}, timeout=timeout)
        return {"status": response.status_code, "ms": (time.perf_counter() - started) * 1000}
    except Exception as exc:  # noqa: BLE001 — любая сетевая ошибка это отказ
        return {"status": 0, "ms": (time.perf_counter() - started) * 1000,
                "error": f"{type(exc).__name__}: {exc}"}


async def run_spam(url: str, *, concurrency: int, total: int, timeout: float,
                   include_llm: bool) -> dict:
    """Шлёт `total` запросов, держа `concurrency` одновременно."""
    endpoints = SAFE_ENDPOINTS + (LLM_ENDPOINTS if include_llm else [])
    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 10)

    async with httpx.AsyncClient(base_url=url.rstrip("/"), limits=limits) as client:
        async def guarded(i: int) -> dict:
            async with semaphore:
                method, path = endpoints[i % len(endpoints)]
                return await _one_request(client, method, path, timeout)

        started = time.perf_counter()
        results = await asyncio.gather(*(guarded(i) for i in range(total)))
        wall = time.perf_counter() - started

    codes = Counter(r["status"] for r in results)
    latencies = sorted(r["ms"] for r in results)
    ok = sum(n for code, n in codes.items() if 200 <= code < 300)
    return {
        "concurrency": concurrency,
        "total": total,
        "wall_s": round(wall, 2),
        "rps": round(total / wall, 1) if wall else 0.0,
        "ok": ok,
        "failed": total - ok,
        "rate_limited": codes.get(429, 0),
        "server_errors": sum(n for code, n in codes.items() if code >= 500),
        "network_errors": codes.get(0, 0),
        "codes": dict(sorted(codes.items())),
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(latencies[max(0, int(len(latencies) * 0.95) - 1)], 1),
        "max_ms": round(max(latencies), 1),
    }


def _degraded(m: dict, limit_p95_ms: float) -> str | None:
    if m["network_errors"]:
        return f"сетевые отказы: {m['network_errors']} (сервер не отвечает)"
    if m["server_errors"]:
        return f"ответы 5xx: {m['server_errors']}"
    if m["rate_limited"]:
        return f"rate-limit 429: {m['rate_limited']}"
    if m["p95_ms"] > limit_p95_ms:
        return f"p95 {m['p95_ms']} мс > порога {limit_p95_ms} мс"
    return None


def _render_report(steps: list[dict], reason: str | None, last_ok: int,
                   url: str, limit_p95_ms: float) -> str:
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    lines = [
        "# Обстрел REST-API — Стенограф",
        "",
        "> ⚙️ Сгенерирован автоматически `scripts/spamtest.py`, руками не редактируется.",
        f"> {now} · цель: `{url}`",
        "",
        "## Итог",
        "",
    ]
    if reason:
        lines += [f"**Держит {last_ok} одновременных запросов.** Дальше — {reason}.", ""]
    else:
        lines += [f"**Предел не достигнут:** выдержал все ступени до "
                  f"{steps[-1]['concurrency']} одновременных запросов.", ""]
    lines += [
        "| Одновременно | Запросов | Время | RPS | Успешно | Отказов | 429 | 5xx | p50 | p95 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in steps:
        lines.append(
            f"| {m['concurrency']} | {m['total']} | {m['wall_s']} c | {m['rps']} | "
            f"{m['ok']} | {m['failed']} | {m['rate_limited']} | {m['server_errors']} | "
            f"{m['p50_ms']} мс | {m['p95_ms']} мс |"
        )
    lines += [
        "",
        "Обстреливаются только читающие эндпоинты (`/api/health`, `/api/asr`, `/api/llm`, "
        "`/api/meetings`, `/api/speakers`) — нагрузка не портит данные. Эндпоинты, "
        "ходящие во внешний LLM-API, исключены: это платные запросы к провайдеру.",
        "",
        f"Порог деградации: p95 > {limit_p95_ms} мс.",
        "",
        "## Как повторить",
        "",
        "```bash",
        "docker compose up -d --no-deps server",
        f"cd server && .venv/bin/python scripts/spamtest.py --url {url} --ramp",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Обстрел REST-API Стенографа")
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--total", type=int, default=500)
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--ramp", action="store_true", help="поднимать нагрузку до отказа")
    ap.add_argument("--start", type=int, default=10)
    ap.add_argument("--step", type=int, default=40)
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--limit-p95-ms", type=float, default=3000.0)
    ap.add_argument("--include-llm", action="store_true",
                    help="обстреливать и эндпоинты внешнего LLM (платные запросы!)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--report")
    args = ap.parse_args()

    try:
        httpx.get(f"{args.url.rstrip('/')}/api/health", timeout=10).raise_for_status()
    except Exception as exc:
        sys.exit(f"Сервер {args.url} недоступен: {exc}")

    if not args.ramp:
        metrics = asyncio.run(run_spam(
            args.url, concurrency=args.concurrency, total=args.total,
            timeout=args.timeout, include_llm=args.include_llm))
        print(json.dumps(metrics, ensure_ascii=False, indent=2) if args.json
              else f"{metrics['total']} запросов при {metrics['concurrency']} одновременных: "
                   f"{metrics['rps']} rps, успешно {metrics['ok']}, отказов {metrics['failed']}, "
                   f"p95 {metrics['p95_ms']} мс")
        return

    steps, reason, last_ok = [], None, 0
    concurrency = args.start
    while concurrency <= args.max:
        total = max(args.total, concurrency * 5)
        print(f"\n▶ ступень: {concurrency} одновременных, {total} запросов…", flush=True)
        metrics = asyncio.run(run_spam(
            args.url, concurrency=concurrency, total=total,
            timeout=args.timeout, include_llm=args.include_llm))
        steps.append(metrics)
        print(f"   {metrics['rps']} rps · p50 {metrics['p50_ms']} мс · "
              f"p95 {metrics['p95_ms']} мс · отказов {metrics['failed']} · "
              f"коды {metrics['codes']}")
        problem = _degraded(metrics, args.limit_p95_ms)
        if problem:
            reason = problem
            print(f"   ✗ предел: {problem}")
            break
        last_ok = concurrency
        print("   ✓ держит")
        concurrency += args.step

    print(f"\n{'=' * 60}")
    print(f"Держит {last_ok} одновременных запросов" if reason else "Предел не достигнут")
    if reason:
        print(f"Причина остановки: {reason}")
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_render_report(steps, reason, last_ok, args.url,
                                       args.limit_p95_ms), encoding="utf-8")
        print(f"Отчёт: {path}")


if __name__ == "__main__":
    main()
