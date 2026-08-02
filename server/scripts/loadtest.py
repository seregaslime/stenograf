"""Нагрузочный тест: параллельные встречи по WebSocket.

Два режима:

1. **Свой сервер** (как раньше) — поднимает временный uvicorn на этой машине.
   Годится для дымовой проверки, но нагружает рабочий ноутбук.

2. **Против контейнера** (`--url`) — бьёт по уже поднятому серверу, например в
   Docker. Это основной сценарий: контейнер играет роль отдельной машины, ему
   можно выдать жёсткий лимит памяти и валить его, не трогая свой компьютер.
   Память меряется через `docker stats`, падение/перезапуск ловится по
   `docker inspect` (RestartCount растёт при OOM-kill).

Рампа (`--ramp`) поднимает число одновременных встреч, пока сервер не начнёт
деградировать: появляются ошибки, растёт задержка или контейнер перезапускается.

Узкое место — глобальный лок ASR (`Transcriber._infer_lock`): все встречи ждут
один транскрайбер, поэтому сквозная задержка растёт с числом параллельных встреч.

Запуск:
    # свой сервер (как раньше)
    .venv/bin/python scripts/loadtest.py --meetings 5 --seconds 30

    # против контейнера, рампа до отказа
    docker compose up -d --no-deps server
    .venv/bin/python scripts/loadtest.py --url http://127.0.0.1:8765 \
        --container practika-server-1 --ramp --report reports/loadtest.md
"""
import argparse
import asyncio
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
import wave
from collections import Counter
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import websockets

SR = 16_000
SERVER_DIR = Path(__file__).resolve().parent.parent


def synth_phrase() -> np.ndarray:
    """Одна русская фраза 16 кГц float32 через macOS `say` (или тон, если say нет)."""
    if shutil.which("say") is None:
        t = np.linspace(0, 3, 3 * SR, dtype=np.float32)  # 3 c тона — грубый запас
        return 0.2 * np.sin(2 * np.pi * 180 * t).astype(np.float32)
    out = Path(SERVER_DIR / "data" / "_loadtest_phrase.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["say", "-v", "Milena", "-o", str(out), f"--data-format=LEI16@{SR}",
         "Добрый день, коллеги, начинаем совещание по проекту, обсудим план и сроки"],
        check=True, capture_output=True,
    )
    with wave.open(str(out)) as w:
        pcm16 = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    out.unlink(missing_ok=True)
    return pcm16.astype(np.float32) / 32768.0


def _pcm(chunk: np.ndarray) -> bytes:
    return np.clip(chunk * 32767, -32768, 32767).astype("<i2").tobytes()


def _percentile(values_sorted: list[float], q: float) -> float:
    """Ближайший ранг: индекс = ceil(q * n) - 1.

    Наивное `int(n * q) - 1` на малых выборках даёт минимум вместо максимума
    (при n=2 это индекс 0), из-за чего p95 оказывался МЕНЬШЕ медианы.
    """
    if not values_sorted:
        return 0.0
    index = math.ceil(q * len(values_sorted)) - 1
    return values_sorted[min(len(values_sorted) - 1, max(0, index))]


def boot_server(port: int, data_dir: Path) -> subprocess.Popen:
    (data_dir / "models").symlink_to(SERVER_DIR / "data" / "models")
    env = os.environ | {
        "STENOGRAF_DATA_DIR": str(data_dir),
        "STENOGRAF_OLLAMA_URL": "http://127.0.0.1:1",  # резюме не нужно — пусть падает тихо
        "STENOGRAF_PRELOAD_ASR": "true",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
        cwd=SERVER_DIR, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/api/asr", timeout=2).json().get("loaded"):
                return proc
        except Exception:
            pass
        time.sleep(2)
    proc.terminate()
    raise TimeoutError("сервер не поднял ASR за 180 c")


# ------------------------------------------------------------------ метрики контейнера

def _docker_mem_mb(container: str) -> float:
    """Текущее потребление памяти контейнером в МБ (0, если недоступен)."""
    try:
        raw = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
    except Exception:
        return 0.0
    used = raw.split("/")[0].strip()  # «1.847GiB / 3.827GiB»
    for suffix, factor in (("GiB", 1024), ("MiB", 1), ("KiB", 1 / 1024), ("B", 1 / 1024 ** 2)):
        if used.endswith(suffix):
            try:
                return float(used[: -len(suffix)]) * factor
            except ValueError:
                return 0.0
    return 0.0


def container_state(container: str) -> dict:
    """Статус и счётчик перезапусков: рост RestartCount = контейнер убили (OOM)."""
    try:
        raw = subprocess.run(
            ["docker", "inspect", "-f",
             "{{.State.Status}} {{.RestartCount}} {{.State.OOMKilled}}", container],
            capture_output=True, text=True, timeout=15,
        ).stdout.split()
    except Exception:
        return {"status": "unknown", "restarts": 0, "oom": False}
    if len(raw) < 3:
        return {"status": "unknown", "restarts": 0, "oom": False}
    return {"status": raw[0], "restarts": int(raw[1]), "oom": raw[2] == "true"}


def _sample_peak(stop: threading.Event, peak: list, pid: int | None, container: str | None):
    """Пиковая память: у контейнера через docker stats, у своего сервера через ps."""
    while not stop.is_set():
        if container:
            peak[0] = max(peak[0], _docker_mem_mb(container))
        elif pid:
            try:
                rss_kb = int(subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(pid)],
                    capture_output=True, text=True).stdout.strip() or 0)
                peak[0] = max(peak[0], rss_kb / 1024)
            except Exception:
                pass
        stop.wait(1.5)


# ------------------------------------------------------------------ одна встреча

FRAME_S = 0.1  # клиент шлёт кадры по 100 мс


async def _one_meeting(
    ws_url: str, title: str, audio: np.ndarray, timeout: float,
    *, pace: float = 0.005, record_audio: bool = False,
) -> dict:
    """Одна встреча.

    pace — пауза между кадрами по 100 мс:
      0.005 (по умолчанию) — «свалить аудио разом», в 20 раз быстрее живого
        клиента. Меряет пропускную способность и очередь.
      0.1 — реальное время, как настоящий клиент. Меряет, сколько ЖИВЫХ встреч
        сервер тянет, не отставая.

    lag_s — сколько сервер доделывал уже ПОСЛЕ конца аудио. Для режима
    реального времени это главный показатель: растёт — значит не успевает.
    """
    segments, error = 0, None
    started = audio_done = None
    try:
        async with websockets.connect(ws_url, max_size=None, open_timeout=30) as ws:
            await ws.send(json.dumps({"type": "start", "title": title,
                                      "record_audio": record_audio,
                                      "summarize": False, "hints": False}))
            assert json.loads(await ws.recv())["type"] == "ready"
            started = time.perf_counter()
            step = SR // 10
            for i in range(0, len(audio), step):
                await ws.send(b"\x00" + _pcm(audio[i:i + step]))
                await asyncio.sleep(pace)
            audio_done = time.perf_counter()
            await ws.send(json.dumps({"type": "stop"}))
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
                if msg["type"] == "segment":
                    segments += 1
                elif msg["type"] == "stopped":
                    break
    except Exception as exc:  # noqa: BLE001 — фиксируем любую ошибку встречи
        error = f"{type(exc).__name__}: {exc}"
    finish = time.perf_counter()
    duration = (finish - started) if started else 0.0
    lag = (finish - audio_done) if audio_done else 0.0
    return {"segments": segments, "duration_s": duration, "lag_s": lag, "error": error}


# ------------------------------------------------------------------ один замер

async def run_load(
    meetings: int,
    seconds: int,
    port: int = 8770,
    *,
    url: str | None = None,
    container: str | None = None,
    timeout: float = 300.0,
    realtime: bool = False,
    record_audio: bool = False,
) -> dict:
    """Гоняет `meetings` параллельных встреч по `seconds` речи, возвращает метрики.

    url=None — поднимает свой временный сервер (нужен кэш server/data/models).
    url задан — бьёт по уже запущенному серверу (например, по контейнеру).
    """
    phrase = synth_phrase()
    reps = max(1, int(np.ceil(seconds * SR / len(phrase))))
    audio = np.tile(phrase, reps)[: seconds * SR]

    proc, data_dir = None, None
    if url:
        base = url.rstrip("/")
        ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + "/ws/live"
    else:
        if not (SERVER_DIR / "data" / "models").exists():
            raise RuntimeError("нет кэша моделей server/data/models — сначала запустите сервер")
        data_dir = Path(SERVER_DIR / "data" / f"_loadtest_{port}")
        if data_dir.exists():
            shutil.rmtree(data_dir)
        data_dir.mkdir(parents=True)
        proc = boot_server(port, data_dir)
        ws_url = f"ws://127.0.0.1:{port}/ws/live"

    before = container_state(container) if container else None
    peak, stop = [0.0], threading.Event()
    sampler = threading.Thread(
        target=_sample_peak, args=(stop, peak, proc.pid if proc else None, container),
        daemon=True,
    )
    sampler.start()

    wall_start = time.perf_counter()
    pace = FRAME_S if realtime else 0.005
    try:
        results = await asyncio.gather(
            *(_one_meeting(ws_url, f"load-{i}", audio, timeout,
                           pace=pace, record_audio=record_audio)
              for i in range(meetings))
        )
    finally:
        stop.set()
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=15)
        if data_dir is not None:
            shutil.rmtree(data_dir, ignore_errors=True)

    after = container_state(container) if container else None
    wall = time.perf_counter() - wall_start
    durations = sorted(r["duration_s"] for r in results)
    lags = sorted(r["lag_s"] for r in results)
    errors = [r["error"] for r in results if r["error"]]
    audio_total = meetings * seconds
    metrics = {
        "meetings": meetings,
        "seconds_each": seconds,
        "realtime": realtime,
        # Во сколько раз быстрее реального времени обработано всё аудио.
        # Ключевая метрика для сайзинга: столько живых встреч тянет один процесс.
        "speedup_x": round(audio_total / wall, 2) if wall else 0.0,
        # Отставание: сколько сервер доделывал уже после конца речи.
        "lag_p50_s": round(statistics.median(lags), 1),
        "lag_p95_s": round(_percentile(lags, 0.95), 1),
        "wall_time_s": round(wall, 1),
        "duration_p50_s": round(statistics.median(durations), 1),
        "duration_p95_s": round(_percentile(durations, 0.95), 1),
        "duration_max_s": round(max(durations), 1),
        "total_segments": sum(r["segments"] for r in results),
        "peak_rss_mb": round(peak[0], 1),
        "errors": errors,
    }
    if container:
        metrics["container_status"] = after["status"]
        metrics["container_restarted"] = after["restarts"] > before["restarts"]
        metrics["container_oom"] = after["oom"]
    return metrics


# ------------------------------------------------------------------ рампа до отказа

def _degraded(m: dict, limit_p95: float) -> str | None:
    """Признак, что сервер перестал справляться. None — всё ещё держит."""
    if m.get("container_restarted") or m.get("container_oom"):
        return "контейнер перезапустился (скорее всего OOM)"
    if m.get("container_status") not in (None, "running"):
        return f"контейнер в состоянии «{m['container_status']}»"
    if m["errors"]:
        return f"ошибки встреч: {len(m['errors'])} из {m['meetings']}"
    if m["total_segments"] == 0:
        return "сервер не распознал ни одной реплики"
    # Более надёжный признак перегрузки, чем ошибки: встреча завершается штатно,
    # но её речь не распозналась. Клиент сбоя не видит — расшифровки просто нет.
    # На замерах явные ошибки плавали (18 отказов и 0 в повторе той же ступени),
    # а недостача сегментов воспроизводилась стабильно.
    if m["total_segments"] < m["meetings"]:
        lost = m["meetings"] - m["total_segments"]
        return (f"потеряны транскрипты: {lost} из {m['meetings']} встреч завершились "
                f"без единого сегмента")
    if m["duration_p95_s"] > limit_p95:
        return f"p95 сквозной задержки {m['duration_p95_s']} c > порога {limit_p95} c"
    return None


async def _measure(
    concurrency: int, seconds: int, *, url, container, timeout, limit_p95: float,
    repeats: int, realtime: bool = False, record_audio: bool = False,
) -> tuple[dict, str | None]:
    """Замер одной ступени. repeats>1 — повторяем и берём ХУДШИЙ результат.

    Предел плавает: на одной и той же ступени наблюдались 18 отказов в первом
    прогоне и 0 в повторе. Один замер — это число одного везения, поэтому для
    честного ответа ступень нужно повторять.
    """
    worst, worst_problem = None, None
    for attempt in range(repeats):
        metrics = await run_load(
            concurrency, seconds, url=url, container=container, timeout=timeout,
            realtime=realtime, record_audio=record_audio,
        )
        problem = _degraded(metrics, limit_p95)
        if repeats > 1:
            print(f"     попытка {attempt + 1}/{repeats}: "
                  f"сегментов {metrics['total_segments']}/{concurrency}, "
                  f"ошибок {len(metrics['errors'])}"
                  f"{' — ' + problem if problem else ' — ok'}")
        # Худший = первый, где нашлась проблема; иначе последний
        if worst is None or (problem and not worst_problem):
            worst, worst_problem = metrics, problem
    worst["repeats"] = repeats
    return worst, worst_problem


async def run_refine(
    *, low: int, high: int, seconds: int, url, container, limit_p95: float,
    timeout: float, repeats: int, realtime: bool = False, record_audio: bool = False,
) -> tuple[list[dict], int]:
    """Бинарный поиск предела в вилке (low держит, high — нет).

    Грубая рампа с шагом 20 даёт ответ «между 120 и 140». Бинарный поиск
    сужает вилку до 1 встречи за ~log2(20) ≈ 5 прогонов вместо 20.
    """
    steps = []
    print(f"\n🔍 уточнение предела в вилке {low}…{high}", flush=True)
    while high - low > 1:
        mid = (low + high) // 2
        print(f"\n▶ проверяем {mid} (вилка {low}…{high})…", flush=True)
        metrics, problem = await _measure(
            mid, seconds, url=url, container=container, timeout=timeout,
            limit_p95=limit_p95, repeats=repeats, realtime=realtime,
            record_audio=record_audio,
        )
        steps.append(metrics)
        if problem:
            print(f"   ✗ {mid} не держит: {problem}")
            high = mid
        else:
            print(f"   ✓ {mid} держит")
            low = mid
    print(f"\n   предел уточнён: {low} держит, {high} — уже нет")
    return steps, low


async def run_ramp(
    *, start: int, step: int, max_meetings: int, seconds: int, url: str | None,
    container: str | None, limit_p95: float, timeout: float, repeats: int = 1,
    realtime: bool = False, record_audio: bool = False,
) -> tuple[list[dict], str | None, int, int]:
    """Поднимает нагрузку ступенями, пока сервер не начнёт деградировать."""
    steps, reason, last_ok = [], None, 0
    concurrency = start
    while concurrency <= max_meetings:
        print(f"\n▶ ступень: {concurrency} параллельных встреч × {seconds} c речи…", flush=True)
        metrics, problem = await _measure(
            concurrency, seconds, url=url, container=container, timeout=timeout,
            limit_p95=limit_p95, repeats=repeats, realtime=realtime,
            record_audio=record_audio,
        )
        steps.append(metrics)
        print(f"   p50 {metrics['duration_p50_s']} c · p95 {metrics['duration_p95_s']} c · "
              f"сегментов {metrics['total_segments']} · память {metrics['peak_rss_mb']} МБ · "
              f"ошибок {len(metrics['errors'])}")
        if problem:
            reason = problem
            print(f"   ✗ предел: {problem}")
            break
        last_ok = concurrency
        print("   ✓ держит")
        concurrency += step
    return steps, reason, last_ok, concurrency if reason else 0


# ------------------------------------------------------------------ отчёт

def _render_report(steps: list[dict], reason: str | None, last_ok: int,
                   target: str, seconds: int, limit_p95: float,
                   *, step: int = 0, refined: bool = False, repeats: int = 1) -> str:
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    lines = [
        "# Нагрузочный прогон — Стенограф",
        "",
        "> ⚙️ Сгенерирован автоматически `scripts/loadtest.py`, руками не редактируется.",
        f"> {now} · цель: `{target}` · речи в каждой встрече: {seconds} c",
        "",
        "## Итог",
        "",
    ]
    if reason:
        # Честно про точность: без уточнения предел известен лишь с точностью до шага
        if refined:
            precision = "предел уточнён бинарным поиском, точность — 1 встреча"
        elif step > 1:
            precision = (f"шаг рампы {step}, поэтому истинный предел — где-то между "
                         f"{last_ok} и {last_ok + step}; для точного числа нужен `--refine`")
        else:
            precision = "шаг рампы 1 — предел точный"
        lines += [
            f"**Предел: {last_ok} параллельных встреч.** На следующей ступени — {reason}.",
            "",
            f"*Точность: {precision}.*",
            "",
        ]
        if repeats > 1:
            lines += [f"*Каждая ступень прогонялась {repeats} раза, в таблице — худший "
                      f"результат.*", ""]
        else:
            lines += ["*Каждая ступень прогонялась один раз. Предел плавает "
                      "(на одной и той же ступени наблюдались 18 отказов и 0 в повторе), "
                      "поэтому для устойчивого числа нужен `--repeats 3`.*", ""]
    else:
        lines += [
            f"**Предел не достигнут:** сервер выдержал все ступени до "
            f"{steps[-1]['meetings']} встреч включительно.",
            "",
        ]
    lines += [
        "| Встреч | Общее время | p50 | p95 | max | Сегментов | Пик памяти | Ошибок |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in steps:
        lines.append(
            f"| {m['meetings']} | {m['wall_time_s']} c | {m['duration_p50_s']} c | "
            f"{m['duration_p95_s']} c | {m['duration_max_s']} c | {m['total_segments']} | "
            f"{m['peak_rss_mb']} МБ | {len(m['errors'])} |"
        )
    lines += [
        "",
        "«Сквозная задержка» — от готовности сессии до события `stopped`, то есть "
        "полный путь: приём аудио → микшер → VAD → ASR → диаризация → БД.",
        "",
    ]

    failing = [m for m in steps if m["errors"]]
    if failing:
        lines += ["## Чем именно отказывает", ""]
        for m in failing:
            kinds = Counter(e.split(":")[0] for e in m["errors"])
            listed = ", ".join(f"`{kind}` × {n}" for kind, n in kinds.most_common())
            lines.append(f"- **{m['meetings']} встреч** — {len(m['errors'])} отказов: {listed}")
            lines.append(f"  - пример: `{m['errors'][0][:160]}`")
        lines.append("")

    lines += [
        "## Что здесь узкое место",
        "",
        "Транскрайбер один на процесс и защищён локом (`Transcriber._infer_lock`): "
        "распознавание не параллелится, встречи выстраиваются в очередь. Поэтому "
        "задержка растёт примерно линейно с числом одновременных встреч, а память — "
        "заметно медленнее (модель в памяти одна, на встречу приходятся только буферы).",
        "",
        f"Порог деградации в этом прогоне: p95 > {limit_p95} c.",
        "",
        "## Как повторить",
        "",
        "```bash",
        "docker compose up -d --no-deps server",
        "cd server && .venv/bin/python scripts/loadtest.py \\",
        f"    --url {target} --container practika-server-1 --ramp",
        "```",
        "",
        "Нагрузка идёт по контейнеру — рабочая машина остаётся свободной, а лимит "
        "памяти контейнера задаётся в `docker-compose.yml` (`mem_limit`).",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Нагрузочный тест Стенографа")
    ap.add_argument("--meetings", type=int, default=5, help="сколько встреч одновременно")
    ap.add_argument("--seconds", type=int, default=30, help="сколько секунд речи в каждой")
    ap.add_argument("--port", type=int, default=8770, help="порт своего временного сервера")
    ap.add_argument("--url", help="бить по уже запущенному серверу, напр. http://127.0.0.1:8765")
    ap.add_argument("--container", help="имя Docker-контейнера для замера памяти и падений")
    ap.add_argument("--ramp", action="store_true", help="поднимать нагрузку до отказа")
    ap.add_argument("--start", type=int, default=2, help="рампа: с какого числа встреч начать")
    ap.add_argument("--step", type=int, default=2, help="рампа: шаг увеличения")
    ap.add_argument("--max", type=int, default=32, help="рампа: потолок")
    ap.add_argument("--limit-p95", type=float, default=60.0,
                    help="рампа: p95 выше этого считается деградацией")
    ap.add_argument("--realtime", action="store_true",
                    help="слать аудио в реальном времени, как живой клиент (по умолчанию "
                         "в 20 раз быстрее — это меряет пропускную способность, а не "
                         "число живых встреч)")
    ap.add_argument("--record-audio", action="store_true",
                    help="включить запись аудио встречи (нагрузка на диск)")
    ap.add_argument("--refine", action="store_true",
                    help="после рампы уточнить предел бинарным поиском в найденной вилке")
    ap.add_argument("--repeats", type=int, default=1,
                    help="сколько раз повторять каждую ступень (берётся худший результат); "
                         "предел плавает, одного замера мало")
    ap.add_argument("--timeout", type=float, default=300.0, help="таймаут ожидания встречи")
    ap.add_argument("--json", action="store_true", help="вывести метрики как JSON")
    ap.add_argument("--report", help="записать отчёт в markdown-файл")
    args = ap.parse_args()

    if args.url:
        try:
            health = httpx.get(f"{args.url.rstrip('/')}/api/health", timeout=10).json()
        except Exception as exc:
            sys.exit(f"Сервер {args.url} недоступен: {exc}")
        if not health.get("asr", {}).get("loaded"):
            print("⚠ модель ASR ещё грузится — первые встречи будут медленнее")

    if args.ramp:
        steps, reason, last_ok, failed_at = asyncio.run(run_ramp(
            start=args.start, step=args.step, max_meetings=args.max, seconds=args.seconds,
            url=args.url, container=args.container, limit_p95=args.limit_p95,
            timeout=args.timeout, repeats=args.repeats, realtime=args.realtime,
            record_audio=args.record_audio,
        ))
        # Грубая рампа даёт вилку («между last_ok и failed_at»). Бинарный поиск
        # сужает её до одной встречи за ~log2(шаг) прогонов вместо шага целиком.
        if args.refine and reason and failed_at - last_ok > 1:
            refined, last_ok = asyncio.run(run_refine(
                low=last_ok, high=failed_at, seconds=args.seconds, url=args.url,
                container=args.container, limit_p95=args.limit_p95,
                timeout=args.timeout, repeats=args.repeats, realtime=args.realtime,
                record_audio=args.record_audio,
            ))
            steps += refined
            steps.sort(key=lambda m: m["meetings"])
        if args.json:
            print(json.dumps({"steps": steps, "limit_reason": reason,
                              "max_ok_meetings": last_ok}, ensure_ascii=False))
        else:
            print(f"\n{'=' * 60}")
            print(f"Предел: {last_ok} параллельных встреч" if reason
                  else "Предел не достигнут")
            if reason:
                print(f"Причина остановки: {reason}")
        if args.report:
            path = Path(args.report)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_render_report(
                steps, reason, last_ok, args.url or f"свой сервер :{args.port}",
                args.seconds, args.limit_p95,
                step=args.step, refined=args.refine, repeats=args.repeats,
            ), encoding="utf-8")
            print(f"Отчёт: {path}")
        return

    metrics = asyncio.run(run_load(
        args.meetings, args.seconds, args.port,
        url=args.url, container=args.container, timeout=args.timeout,
        realtime=args.realtime, record_audio=args.record_audio,
    ))
    if args.json:
        print(json.dumps(metrics, ensure_ascii=False))
        return
    print(f"\nНагрузка: {metrics['meetings']} встреч × {metrics['seconds_each']} c речи")
    print(f"  общее время (wall):     {metrics['wall_time_s']} c")
    print(f"  сквозная встреча p50:   {metrics['duration_p50_s']} c")
    print(f"  сквозная встреча p95:   {metrics['duration_p95_s']} c")
    print(f"  сквозная встреча max:   {metrics['duration_max_s']} c")
    print(f"  распознано сегментов:   {metrics['total_segments']}")
    print(f"  пиковая память:         {metrics['peak_rss_mb']} МБ")
    print(f"  ошибок:                 {len(metrics['errors'])}")
    for err in metrics["errors"]:
        print(f"    ! {err}")


if __name__ == "__main__":
    main()
