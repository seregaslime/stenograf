"""Долгая встреча: ищем утечки памяти и деградацию во времени.

Остальные нагрузочные прогоны длятся десятки секунд — за такое время утечка не
видна. Здесь наоборот: нагрузка небольшая и постоянная, зато часами. Ловим то,
что не лечится докупкой памяти: если сервер течёт, больший объём лишь отодвигает
падение, а не отменяет его.

Что меряется:
- память контейнера во времени (линейная регрессия → МБ/час роста);
- скорость поступления сегментов — не начал ли сервер отставать к концу;
- рост памяти на 1000 распознанных реплик.

Подозреваемые в ws.py: дека _recent, счётчик _participants (растёт вечно),
_recent_speakers, накопление сегментов в БД.

Запуск:
    docker compose up -d --no-deps server
    .venv/bin/python scripts/soaktest.py --url http://127.0.0.1:8765 \
        --container practika-server-1 --minutes 45 --report reports/soaktest.md
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import websockets

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR / "scripts"))

from loadtest import (  # noqa: E402
    FRAME_S,
    SR,
    _docker_mem_mb,
    _pcm,
    container_state,
    synth_phrase,
)


async def _sampler(container: str, samples: list, stop: asyncio.Event,
                   started: float, every: float) -> None:
    """Раз в `every` секунд записывает (секунда прогона, память МБ)."""
    while not stop.is_set():
        mem = await asyncio.to_thread(_docker_mem_mb, container)
        if mem:
            samples.append((time.perf_counter() - started, mem))
        try:
            await asyncio.wait_for(stop.wait(), timeout=every)
        except asyncio.TimeoutError:
            pass


async def _talk(ws_url: str, phrase: np.ndarray, minutes: float,
                marks: list, stop: asyncio.Event) -> dict:
    """Одна длинная встреча: гоняем фразу по кругу в реальном времени."""
    segments, error = 0, None
    deadline = time.perf_counter() + minutes * 60
    step = SR // 10
    try:
        async with websockets.connect(ws_url, max_size=None, open_timeout=30) as ws:
            await ws.send(json.dumps({"type": "start", "title": "soak",
                                      "record_audio": False, "summarize": False,
                                      "hints": False}))
            assert json.loads(await ws.recv())["type"] == "ready"

            async def drain():
                """Считаем сегменты и отмечаем момент прихода каждого."""
                nonlocal segments
                while True:
                    msg = json.loads(await ws.recv())
                    if msg["type"] == "segment":
                        segments += 1
                        marks.append(time.perf_counter())
                    elif msg["type"] == "stopped":
                        return

            reader = asyncio.create_task(drain())
            while time.perf_counter() < deadline:
                for i in range(0, len(phrase), step):
                    await ws.send(b"\x00" + _pcm(phrase[i:i + step]))
                    await asyncio.sleep(FRAME_S)  # реальное время, как живой клиент
                    if time.perf_counter() >= deadline:
                        break
            await ws.send(json.dumps({"type": "stop"}))
            await asyncio.wait_for(reader, timeout=180)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        stop.set()
    return {"segments": segments, "error": error}


def _slope_mb_per_hour(samples: list) -> float:
    """Наклон прямой памяти во времени. Считаем со второй половины прогона:
    первые минуты память ещё скачет из-за прогрева и кэшей."""
    tail = samples[len(samples) // 2:] or samples
    if len(tail) < 3:
        return 0.0
    xs = np.array([t for t, _ in tail])
    ys = np.array([m for _, m in tail])
    slope, _ = np.polyfit(xs, ys, 1)  # МБ в секунду
    return float(slope * 3600)


def _segment_rate(marks: list, window: float = 300.0) -> tuple[float, float]:
    """Сегментов в минуту в начале и в конце прогона — не отстаёт ли сервер."""
    if len(marks) < 4:
        return 0.0, 0.0
    first, last = marks[0], marks[-1]
    early = [m for m in marks if m - first <= window]
    late = [m for m in marks if last - m <= window]
    span_early = max(early[-1] - early[0], 1e-6)
    span_late = max(late[-1] - late[0], 1e-6)
    return len(early) / span_early * 60, len(late) / span_late * 60


def _render(result: dict) -> str:
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    s = result
    verdict = s["verdict"]
    lines = [
        "# Долгая встреча (soak) — Стенограф",
        "",
        "> ⚙️ Сгенерирован автоматически `scripts/soaktest.py`, руками не редактируется.",
        f"> {now} · цель: `{s['target']}` · длительность: {s['minutes']} мин",
        "",
        "## Итог",
        "",
        f"**{verdict}**",
        "",
        "| Показатель | Значение |",
        "|---|---:|",
        f"| Длительность прогона | {s['minutes']} мин |",
        f"| Распознано реплик | {s['segments']} |",
        f"| Память в начале | {s['mem_start_mb']} МБ |",
        f"| Память в конце | {s['mem_end_mb']} МБ |",
        f"| Пик памяти | {s['mem_peak_mb']} МБ |",
        f"| **Рост памяти** | **{s['slope_mb_per_hour']:+.1f} МБ/час** |",
        f"| Рост на 1000 реплик | {s['mb_per_1000_segments']:+.1f} МБ |",
        f"| Реплик/мин в начале | {s['rate_early_per_min']:.1f} |",
        f"| Реплик/мин в конце | {s['rate_late_per_min']:.1f} |",
        f"| Состояние контейнера | {s['container_status']} |",
        f"| Перезапусков / OOM | {s['restarted']} / {s['oom']} |",
        "",
        "## Как читать",
        "",
        "**Рост памяти** — наклон прямой по второй половине прогона (первые минуты "
        "память скачет из-за прогрева, их отбрасываем). Несколько МБ/час — это "
        "нормальные кэши и фрагментация. Десятки МБ/час на постоянной нагрузке — "
        "утечка: она не лечится докупкой памяти, больший объём лишь отодвигает "
        "падение.",
        "",
        "**Реплик в минуту** в начале и в конце должны совпадать. Если к концу "
        "заметно меньше — сервер накапливает отставание и не разгребает очередь.",
        "",
        "## Как повторить",
        "",
        "```bash",
        "docker compose up -d --no-deps server",
        f"cd server && .venv/bin/python scripts/soaktest.py --url {s['target']} \\",
        f"    --container {s['container']} --minutes {s['minutes']}",
        "```",
    ]
    return "\n".join(lines) + "\n"


async def run(url: str, container: str, minutes: float, every: float) -> dict:
    base = url.rstrip("/")
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + "/ws/live"
    phrase = synth_phrase()

    before = container_state(container)
    samples, marks = [], []
    stop = asyncio.Event()
    started = time.perf_counter()

    sampler = asyncio.create_task(_sampler(container, samples, stop, started, every))
    talk = await _talk(ws_url, phrase, minutes, marks, stop)
    await sampler

    after = container_state(container)
    mem = [m for _, m in samples] or [0.0]
    early, late = _segment_rate(marks)
    slope = _slope_mb_per_hour(samples)
    grew = mem[-1] - mem[0]

    # Короткий прогон ловит прогрев, а не утечку: за первые минуты память растёт
    # на кэшах и аллокаторе, и экстраполяция даёт абсурдные гигабайты в час.
    MIN_MINUTES_FOR_VERDICT = 10.0

    if after["oom"] or after["restarts"] > before["restarts"]:
        verdict = "❌ Контейнер перезапустился — память кончилась"
    elif minutes < MIN_MINUTES_FOR_VERDICT:
        verdict = (f"◻️ Прогон слишком короткий ({minutes:g} мин) для вывода об утечке: "
                   f"наклон {slope:+.1f} МБ/час — это в основном прогрев. "
                   f"Нужно от {MIN_MINUTES_FOR_VERDICT:g} минут")
    elif slope > 50:
        verdict = f"❌ Похоже на утечку: {slope:+.1f} МБ/час на постоянной нагрузке"
    elif slope > 15:
        verdict = f"⚠️ Память растёт на {slope:+.1f} МБ/час — стоит прогнать дольше"
    elif late < early * 0.7 and early > 0:
        verdict = "⚠️ Сервер к концу отстаёт: реплик в минуту стало заметно меньше"
    else:
        verdict = f"✅ Утечки не видно: {slope:+.1f} МБ/час, скорость обработки стабильна"

    return {
        "target": base, "container": container, "minutes": minutes,
        "segments": talk["segments"], "error": talk["error"],
        "mem_start_mb": round(mem[0], 1), "mem_end_mb": round(mem[-1], 1),
        "mem_peak_mb": round(max(mem), 1),
        "slope_mb_per_hour": round(slope, 1),
        "mb_per_1000_segments": round(grew / talk["segments"] * 1000, 1)
        if talk["segments"] else 0.0,
        "rate_early_per_min": round(early, 1), "rate_late_per_min": round(late, 1),
        "container_status": after["status"], "restarted": after["restarts"] - before["restarts"],
        "oom": after["oom"], "samples": [(round(t, 1), round(m, 1)) for t, m in samples],
        "verdict": verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Долгая встреча: поиск утечек")
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--container", default="practika-server-1")
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--sample-every", type=float, default=15.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--report")
    args = ap.parse_args()

    try:
        httpx.get(f"{args.url.rstrip('/')}/api/health", timeout=10).raise_for_status()
    except Exception as exc:
        sys.exit(f"Сервер {args.url} недоступен: {exc}")

    print(f"Долгая встреча: {args.minutes} мин, замер памяти каждые "
          f"{args.sample_every} c. Это надолго…", flush=True)
    result = asyncio.run(run(args.url, args.container, args.minutes, args.sample_every))

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"\n{result['verdict']}")
        print(f"  реплик: {result['segments']}")
        print(f"  память: {result['mem_start_mb']} → {result['mem_end_mb']} МБ "
              f"(пик {result['mem_peak_mb']})")
        print(f"  рост: {result['slope_mb_per_hour']:+.1f} МБ/час")
        print(f"  реплик/мин: {result['rate_early_per_min']} в начале, "
              f"{result['rate_late_per_min']} в конце")
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_render(result), encoding="utf-8")
        print(f"Отчёт: {path}")


if __name__ == "__main__":
    main()
