"""Нагрузочный тест: N параллельных встреч по WebSocket против живого сервера.

Поднимает свой uvicorn (временный data_dir, симлинк моделей — как e2e-тест),
синтезирует одну фразу через macOS `say`, гоняет N встреч одновременно, каждая
льёт `seconds` секунд речи. Мерит: сквозное время встречи (p50/p95), число
распознанных сегментов, пиковый RSS сервера, ошибки.

Узкое место — глобальный лок ASR (`Transcriber._infer_lock`): все встречи ждут
один транскрайбер, поэтому сквозная задержка растёт с числом параллельных встреч.

Запуск:
    .venv/bin/python scripts/loadtest.py --meetings 5 --seconds 30
    .venv/bin/python scripts/loadtest.py --meetings 3 --seconds 15 --json
"""
import argparse
import asyncio
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
import wave
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


def _sample_rss(pid: int, stop: threading.Event, peak: list) -> None:
    while not stop.is_set():
        try:
            rss_kb = int(subprocess.run(
                ["ps", "-o", "rss=", "-p", str(pid)],
                capture_output=True, text=True).stdout.strip() or 0)
            peak[0] = max(peak[0], rss_kb)
        except Exception:
            pass
        time.sleep(0.5)


async def _one_meeting(ws_url: str, title: str, audio: np.ndarray) -> dict:
    segments, error = 0, None
    started = None
    try:
        async with websockets.connect(ws_url, max_size=None) as ws:
            await ws.send(json.dumps({"type": "start", "title": title,
                                      "record_audio": False, "summarize": False, "hints": False}))
            assert json.loads(await ws.recv())["type"] == "ready"
            started = time.perf_counter()
            step = SR // 10  # кадры по 100 мс, как с клиента
            for i in range(0, len(audio), step):
                await ws.send(b"\x00" + _pcm(audio[i:i + step]))
                await asyncio.sleep(0.005)
            await ws.send(json.dumps({"type": "stop"}))
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=300))
                if msg["type"] == "segment":
                    segments += 1
                elif msg["type"] == "stopped":
                    break
    except Exception as exc:  # noqa: BLE001 — фиксируем любую ошибку встречи
        error = f"{type(exc).__name__}: {exc}"
    duration = (time.perf_counter() - started) if started else 0.0
    return {"segments": segments, "duration_s": duration, "error": error}


async def run_load(meetings: int, seconds: int, port: int = 8770) -> dict:
    """Поднимает сервер, гоняет `meetings` параллельных встреч по `seconds` речи,
    возвращает метрики. Требует кэш моделей server/data/models."""
    if not (SERVER_DIR / "data" / "models").exists():
        raise RuntimeError("нет кэша моделей server/data/models — сначала запустите сервер")
    phrase = synth_phrase()
    reps = max(1, int(np.ceil(seconds * SR / len(phrase))))
    audio = np.tile(phrase, reps)[: seconds * SR]

    data_dir = Path(SERVER_DIR / "data" / f"_loadtest_{port}")
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)
    proc = boot_server(port, data_dir)
    peak_rss, stop = [0], threading.Event()
    sampler = threading.Thread(target=_sample_rss, args=(proc.pid, stop, peak_rss), daemon=True)
    sampler.start()

    ws_url = f"ws://127.0.0.1:{port}/ws/live"
    wall_start = time.perf_counter()
    try:
        results = await asyncio.gather(
            *(_one_meeting(ws_url, f"load-{i}", audio) for i in range(meetings))
        )
    finally:
        stop.set()
        proc.terminate()
        proc.wait(timeout=15)
        shutil.rmtree(data_dir, ignore_errors=True)

    durations = sorted(r["duration_s"] for r in results)
    errors = [r["error"] for r in results if r["error"]]
    return {
        "meetings": meetings,
        "seconds_each": seconds,
        "wall_time_s": round(time.perf_counter() - wall_start, 1),
        "duration_p50_s": round(statistics.median(durations), 1),
        "duration_p95_s": round(durations[int(len(durations) * 0.95) - 1], 1),
        "duration_max_s": round(max(durations), 1),
        "total_segments": sum(r["segments"] for r in results),
        "peak_rss_mb": round(peak_rss[0] / 1024, 1),
        "errors": errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Нагрузочный тест Стенографа")
    ap.add_argument("--meetings", type=int, default=5, help="сколько встреч одновременно")
    ap.add_argument("--seconds", type=int, default=30, help="сколько секунд речи в каждой")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--json", action="store_true", help="вывести метрики как JSON")
    args = ap.parse_args()

    metrics = asyncio.run(run_load(args.meetings, args.seconds, args.port))
    if args.json:
        print(json.dumps(metrics, ensure_ascii=False))
        return
    print(f"\nНагрузка: {metrics['meetings']} встреч × {metrics['seconds_each']} c речи")
    print(f"  общее время (wall):     {metrics['wall_time_s']} c")
    print(f"  сквозная встреча p50:   {metrics['duration_p50_s']} c")
    print(f"  сквозная встреча p95:   {metrics['duration_p95_s']} c")
    print(f"  сквозная встреча max:   {metrics['duration_max_s']} c")
    print(f"  распознано сегментов:   {metrics['total_segments']}")
    print(f"  пиковый RSS сервера:    {metrics['peak_rss_mb']} МБ")
    print(f"  ошибок:                 {len(metrics['errors'])}")
    for err in metrics["errors"]:
        print(f"    ! {err}")


if __name__ == "__main__":
    main()
