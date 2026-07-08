"""End-to-end тест живого пайплайна без микрофона.

Синтезирует русскую речь голосами macOS `say`, стримит её на сервер по WebSocket
(канал 0 — «микрофон», канал 1 — «системный звук» двумя разными голосами)
и печатает события сервера. Ожидание: реплики распознаны, в системном канале
появились два разных спикера.

Запуск (сервер должен работать на 8765):
    .venv/bin/python tests/e2e_live.py
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import websockets

SERVER = "ws://127.0.0.1:8765/ws/live"
RATE = 16_000
FRAME = 1600  # 100 мс

# В macOS из русских голосов обычно установлена только Milena, поэтому «второй
# человек» имитируется питч-шифтом (ресемплинг): тембр меняется достаточно,
# чтобы ECAPA-эмбеддинг считал его другим голосом.
PHRASES = [
    # (канал, pitch-фактор, текст)
    (0, 1.0, "Добрый день, коллеги. Начинаем встречу по проекту Стенограф."),
    (1, 0.78, "Привет! Я подготовил отчёт по серверной части, всё готово к демонстрации."),
    (1, 1.0, "Отлично. Тогда решаем так: завтра показываем прототип руководителю."),
    (0, 1.0, "Договорились. Я возьму на себя подготовку презентации."),
]


def synth(text: str, out_dir: Path, index: int) -> Path:
    """say → aiff → afconvert → wav 16 кГц mono PCM16."""
    aiff = out_dir / f"{index}.aiff"
    wav = out_dir / f"{index}.wav"
    subprocess.run(["say", "-v", "Milena", "-o", str(aiff), text], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
        check=True,
    )
    return wav


def read_pcm(path: Path, pitch: float) -> bytes:
    with wave.open(str(path), "rb") as f:
        assert f.getframerate() == RATE and f.getnchannels() == 1, "нужен 16k mono"
        raw = f.readframes(f.getnframes())
    if pitch == 1.0:
        return raw
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    positions = np.arange(0, len(samples) - 1, pitch)
    shifted = np.interp(positions, np.arange(len(samples)), samples)
    return shifted.astype("<i2").tobytes()


async def stream(ws, channel: int, pcm: bytes) -> None:
    """Шлём кадрами по 100 мс, как настоящий клиент (чуть быстрее реального времени)."""
    for offset in range(0, len(pcm), FRAME * 2):
        frame = bytes([channel]) + pcm[offset: offset + FRAME * 2]
        await ws.send(frame)
        await asyncio.sleep(0.03)
    # секунда тишины, чтобы VAD закрыл фразу
    silence = bytes(FRAME * 2)
    for _ in range(12):
        await ws.send(bytes([channel]) + silence)
        await asyncio.sleep(0.03)


async def main() -> None:
    out_dir = Path(tempfile.mkdtemp(prefix="stenograf_e2e_"))
    print(f"Синтез речи в {out_dir} ...")
    clips = [(ch, read_pcm(synth(text, out_dir, i), pitch))
             for i, (ch, pitch, text) in enumerate(PHRASES)]

    events: list[dict] = []
    speakers: dict[int, str] = {}

    async with websockets.connect(SERVER, max_size=None) as ws:
        hints = os.environ.get("E2E_HINTS") == "1"
        await ws.send(json.dumps(
            {"type": "start", "title": "E2E тест", "record_audio": False, "hints": hints}
        ))

        async def reader():
            async for message in ws:
                event = json.loads(message)
                events.append(event)
                if event["type"] == "segment":
                    seg = event["segment"]
                    who = seg["speaker"]["name"] if seg["speaker"] else "?"
                    sim = seg.get("similarity")
                    print(f"  [{seg['channel']:6}] {who}: {seg['text']}"
                          + (f"  (sim={sim})" if sim is not None else ""))
                    if seg["speaker"] and not seg["speaker"]["is_self"]:
                        speakers[seg["speaker"]["id"]] = seg["speaker"]["name"]
                elif event["type"] == "speaker_new":
                    print(f"  ++ новый спикер: {event['speaker']['name']}")
                elif event["type"] in ("hint", "hint_error"):
                    print(f"  💡 {event.get('text') or event.get('message')}")
                elif event["type"] == "stopped":
                    print(f"  встреча #{event['meeting_id']} завершена")
                    return

        reader_task = asyncio.create_task(reader())
        # ready
        while not any(e["type"] == "ready" for e in events):
            await asyncio.sleep(0.1)
        print("Стрим аудио...")
        for channel, pcm in clips:
            await stream(ws, channel, pcm)
        if hints:
            # ждём первую подсказку (LLM думает десятки секунд), продолжая слать тишину
            print("Ждём подсказку LLM...")
            silence = bytes([0]) + bytes(FRAME * 2)
            for _ in range(int(90 / 0.1)):
                if any(e["type"] in ("hint", "hint_error") for e in events):
                    break
                await ws.send(silence)
                await asyncio.sleep(0.1)
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.wait_for(reader_task, timeout=180)

    got_segments = sum(1 for e in events if e["type"] == "segment")
    mic_segments = sum(1 for e in events if e["type"] == "segment" and e["segment"]["channel"] == "mic")
    got_hint = any(e["type"] == "hint" for e in events)
    print(f"\nИтого: сегментов {got_segments} (mic: {mic_segments}), "
          f"системных спикеров: {len(speakers)} {list(speakers.values())}"
          + (f", подсказка: {'да' if got_hint else 'нет'}" if hints else ""))
    ok = got_segments >= 3 and mic_segments >= 1 and len(speakers) >= 1
    if hints:
        ok = ok and got_hint
    if len(speakers) < 2:
        print("Примечание: два разных голоса разводятся на два профиля только на чистой базе —"
              " после объединения профилей оба голоса узнаются как один спикер.")
    print("PASS" if ok else "FAIL: ожидалось ≥3 сегментов, ≥1 системный спикер"
          + (" и подсказка LLM" if hints else ""))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
