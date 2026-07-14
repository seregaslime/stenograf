"""E2E: одноразовый живой сервер + WebSocket-клиент, как настоящее приложение.

Поднимает uvicorn на :8766 с временной папкой данных (модели — симлинком из
server/data/models, чтобы не перекачивать веса; БД и голоса одноразовые —
живая база не затрагивается). Прогоняет встречи целиком: синтезированный звук →
микшер → VAD → ASR → диаризация → события → REST. Ollama намеренно недоступна —
проверяется честная ошибка резюме (живое резюме — в ручном чек-листе TESTING.md).

Запуск (медленный, несколько минут):  .venv/bin/python -m pytest -m e2e
Тесты зависят друг от друга по данным и выполняются по порядку — база
накапливает спикеров и встречи, как при реальном использовании.
"""
import asyncio
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import httpx
import numpy as np
import pytest
import websockets

pytestmark = pytest.mark.e2e

SR = 16_000
PORT = 8766
BASE = f"http://127.0.0.1:{PORT}"
WS_URL = f"ws://127.0.0.1:{PORT}/ws/live"
SERVER_DIR = Path(__file__).resolve().parent.parent

# Три разных «человека»: русский голос + два иностранных (тембр важнее произношения)
VOICES = {"milena": "Milena", "daniel": "Daniel", "anna": "Anna"}


# ------------------------------------------------------------------ фикстуры

@pytest.fixture(scope="module")
def server(tmp_path_factory):
    if not (SERVER_DIR / "data" / "models").exists():
        pytest.skip("нет кэша моделей server/data/models — сначала запустите сервер")
    data_dir = tmp_path_factory.mktemp("e2e_data")
    (data_dir / "models").symlink_to(SERVER_DIR / "data" / "models")
    env = os.environ | {
        "STENOGRAF_DATA_DIR": str(data_dir),
        "STENOGRAF_OLLAMA_URL": "http://127.0.0.1:1",  # резюме должно честно падать
        # Порог теста ≠ рабочему 0.35: синтетические голоса macOS между собой
        # ближе живых (женские пары до ~0.38 — первый прогон честно поймал
        # склейку Alice с Милениным профилем). e2e проверяет ПРАВИЛА
        # сопоставления; граничные значения порога покрыты юнит-тестами,
        # подбор под живые голоса — scripts/eval_voices.py.
        "STENOGRAF_SPEAKER_MATCH_THRESHOLD": "0.45",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)],
        cwd=SERVER_DIR, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_loaded(timeout=180)
        yield {"data_dir": data_dir}
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def _wait_loaded(timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            state = httpx.get(f"{BASE}/api/asr", timeout=2).json()
            if state["loaded"]:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("сервер не поднял ASR за отведённое время")


@pytest.fixture(scope="module")
def voices(tmp_path_factory):
    """Синтезированные фразы: voices['milena'] → float32 16 кГц."""
    import shutil

    if shutil.which("say") is None:
        pytest.skip("нет команды say (не macOS)")
    directory = tmp_path_factory.mktemp("voices")
    phrases = {
        "milena": "Добрый день, коллеги, начинаем совещание по проекту Стенограф",
        "daniel": "Спасибо за приглашение, я подготовил отчёт по интеграции",
        "anna": "У меня есть вопросы по срокам следующего этапа работ",
    }
    out = {}
    for key, text in phrases.items():
        path = directory / f"{key}.wav"
        subprocess.run(
            ["say", "-v", VOICES[key], "-o", str(path), f"--data-format=LEI16@{SR}", text],
            check=True, capture_output=True,
        )
        with wave.open(str(path)) as w:
            pcm16 = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        out[key] = pcm16.astype(np.float32) / 32768.0
    return out


# ------------------------------------------------------------------ helpers

def pcm(chunk: np.ndarray) -> bytes:
    return np.clip(chunk * 32767, -32768, 32767).astype("<i2").tobytes()


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def run_meeting(title, mic=None, system=None, abort=False, during=None):
    """Проигрывает встречу. Возвращает (meeting_id, сегменты).

    during(meeting_id) — колбэк посреди стрима (для проверок «пока live»);
    abort=True — оборвать сокет без команды stop."""

    async def _run():
        segments = []
        async with websockets.connect(WS_URL, max_size=None) as ws:
            await ws.send(json.dumps({"type": "start", "title": title, "record_audio": False}))
            ready = json.loads(await ws.recv())
            assert ready["type"] == "ready", ready
            meeting_id = ready["meeting_id"]

            total = max(len(x) for x in (mic, system) if x is not None)
            step = SR // 10
            called = False
            for i in range(0, total, step):
                if mic is not None and i < len(mic):
                    await ws.send(b"\x00" + pcm(mic[i:i + step]))
                if system is not None and i < len(system):
                    await ws.send(b"\x01" + pcm(system[i:i + step]))
                if during is not None and not called and i >= total // 2:
                    called = True
                    await asyncio.to_thread(during, meeting_id)
                await asyncio.sleep(0.003)

            if abort:
                return meeting_id, segments  # выход из with → сокет рвётся без stop
            await ws.send(json.dumps({"type": "stop"}))
            while True:
                message = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
                if message["type"] == "segment":
                    segments.append(message["segment"])
                if message["type"] == "stopped":
                    return meeting_id, segments

    return asyncio.run(_run())


def echoed(voice: np.ndarray, gain: float = 0.3, delay_s: float = 0.04) -> np.ndarray:
    """Как звучит системный голос, долетевший из колонок до микрофона."""
    return np.concatenate([silence(delay_s), voice * gain])[: len(voice)]


def get_meeting(meeting_id: int) -> dict:
    return httpx.get(f"{BASE}/api/meetings/{meeting_id}", timeout=5).json()


def wait_status(meeting_id: int, expected: str, timeout: float = 60) -> dict:
    deadline = time.time() + timeout
    meeting = {}
    while time.time() < deadline:
        meeting = get_meeting(meeting_id)
        if meeting["status"] == expected:
            return meeting
        time.sleep(1)
    raise TimeoutError(f"встреча #{meeting_id} не перешла в {expected}: {meeting.get('status')}")


# ------------------------------------------------------------------ сценарии

def test_01_speaker_echo_no_duplicates(server, voices):
    """Звонок на колонках: голос собеседника и его эхо в микрофоне → один
    сегмент, приписан НЕ «Вы», канал — system."""
    voice = voices["milena"]
    system = np.concatenate([voice, silence(1.2)])
    mic = np.concatenate([echoed(voice), silence(1.2)])
    _, segments = run_meeting("e2e: эхо колонок", mic=mic, system=system)

    assert len(segments) == 1, [s["text"] for s in segments]
    assert not segments[0]["speaker"]["is_self"]
    assert segments[0]["channel"] == "system"
    assert "коллеги" in segments[0]["text"].lower()


def test_02_self_enrolls_and_is_remembered(server, voices):
    """Голос из микрофона становится «Вы»; на следующей встрече тот же голос
    узнаётся даже из системного канала — база голосов живёт между встречами."""
    voice = voices["daniel"]
    _, segments = run_meeting("e2e: энролл владельца",
                              mic=np.concatenate([voice, silence(1.2)]))
    assert len(segments) >= 1
    assert all(s["speaker"]["is_self"] for s in segments)

    _, segments2 = run_meeting("e2e: память голосов",
                               system=np.concatenate([voice, silence(1.2)]))
    assert len(segments2) >= 1
    assert segments2[0]["speaker"]["is_self"], "владелец не узнан на второй встрече"
    assert segments2[0]["similarity"] is not None


def test_03_two_voices_and_short_reply(server, voices):
    """Два голоса в одной встрече → два разных профиля (Милена — та же, что
    во встрече №1, владелец узнан по голосу); короткая реплика (<0.4 с)
    приклеивается к последнему говорившему."""
    gap = silence(1.0)
    short = voices["daniel"][SR: SR + int(0.3 * SR)]  # кусочек речи 0.3 с
    system = np.concatenate([voices["milena"], gap, voices["daniel"], gap, short, gap])
    _, segments = run_meeting("e2e: несколько голосов", system=system)

    assert len(segments) == 3, [s["text"] for s in segments]
    milena, daniel, tail = segments
    assert milena["speaker"]["id"] != daniel["speaker"]["id"]
    assert not milena["speaker"]["is_self"]
    assert daniel["speaker"]["is_self"], "владелец не узнан по голосу в system-канале"
    assert tail["speaker"]["id"] == daniel["speaker"]["id"], "короткая реплика ушла не тому"
    assert tail["similarity"] is None  # эмбеддинг не считался

    # профили без дублей: «Вы» (Daniel) + Милена (узнана повторно, не дублируется)
    speakers = httpx.get(f"{BASE}/api/speakers", timeout=5).json()
    assert len(speakers) == 2, [s["name"] for s in speakers]


def test_04_guest_gets_new_profile(server, voices):
    """Незнакомый голос получает новый профиль, не примазываясь
    ни к «Вы», ни к Милене."""
    _, segments = run_meeting("e2e: гость",
                              system=np.concatenate([voices["anna"], silence(1.2)]))
    assert len(segments) >= 1
    assert not segments[0]["speaker"]["is_self"]
    speakers = httpx.get(f"{BASE}/api/speakers", timeout=5).json()
    assert len(speakers) == 3, [s["name"] for s in speakers]


def test_05_silent_meeting(server):
    """Тишина всю встречу: ноль сегментов, встреча завершается, резюме честно
    сообщает, что распознавать нечего."""
    meeting_id, segments = run_meeting("e2e: тишина", mic=silence(4), system=silence(4))
    assert segments == []
    meeting = wait_status(meeting_id, "done")
    assert meeting["summary_error"], "нет объяснения, почему резюме не построено"


def test_06_long_monologue_is_split(server, voices):
    """Монолог длиннее лимита сегмента (15 с) режется на части, стоп дожимает
    хвост, встреча завершается."""
    piece = voices["milena"]
    repeats = max(2, int(np.ceil(20 * SR / len(piece))))
    meeting_id, segments = run_meeting(
        "e2e: монолог",
        system=np.concatenate([np.tile(piece, repeats), silence(1.2)]),
    )
    assert len(segments) >= 2, "длинный монолог не нарезался"
    wait_status(meeting_id, "done")


def test_07_ws_abort_finalizes_meeting(server, voices):
    """Клиент оборвался без stop (закрыли крышку ноутбука) — встреча всё равно
    завершается на сервере и не остаётся вечно live."""
    voice = voices["anna"]
    meeting_id, _ = run_meeting("e2e: обрыв связи",
                                system=np.concatenate([voice, silence(1.2)]), abort=True)
    meeting = wait_status(meeting_id, "done", timeout=90)
    assert meeting["status"] == "done"


def test_08_summary_error_is_reported(server):
    """Ollama недоступна → у встречи с речью появляется понятная ошибка резюме
    (полный цикл с живой Ollama — в ручном чек-листе TESTING.md)."""
    meetings = httpx.get(f"{BASE}/api/meetings", timeout=5).json()
    with_speech = [m for m in meetings if m["title"] == "e2e: энролл владельца"]
    assert with_speech
    meeting = get_meeting(with_speech[0]["id"])
    assert meeting["status"] == "done"
    assert meeting["summary"] is None
    assert meeting["summary_error"], "ошибка резюме не показана"


def test_09_rest_operations(server):
    """Переименование, merge, экспорт, удаление — поверх накопленных данных."""
    speakers = httpx.get(f"{BASE}/api/speakers", timeout=5).json()
    others = [s for s in speakers if not s["is_self"]]
    assert len(others) >= 2

    renamed = httpx.patch(f"{BASE}/api/speakers/{others[0]['id']}",
                          json={"name": "Илья Петрович"}, timeout=5).json()
    assert renamed["name"] == "Илья Петрович"

    merged = httpx.post(f"{BASE}/api/speakers/merge",
                        json={"speaker_ids": [others[0]["id"], others[1]["id"]]},
                        timeout=10).json()
    assert merged["name"] == "Илья Петрович"  # человеческое имя побеждает
    after = httpx.get(f"{BASE}/api/speakers", timeout=5).json()
    assert len(after) == len(speakers) - 1

    # у объединённого профиля отпечатки обоих; один можно удалить
    target = next(s for s in after if s["id"] == merged["target_id"])
    assert len(target["voiceprints"]) == 2
    print_id = target["voiceprints"][0]["id"]
    removed = httpx.delete(
        f"{BASE}/api/speakers/{target['id']}/voiceprints/{print_id}", timeout=5)
    assert removed.status_code == 200
    fresh = next(s for s in httpx.get(f"{BASE}/api/speakers", timeout=5).json()
                 if s["id"] == target["id"])
    assert [p["id"] for p in fresh["voiceprints"]] == [target["voiceprints"][1]["id"]]
    assert httpx.delete(
        f"{BASE}/api/speakers/{target['id']}/voiceprints/{print_id}", timeout=5,
    ).status_code == 404  # повторное удаление — честный 404

    meetings = httpx.get(f"{BASE}/api/meetings", timeout=5).json()
    first = min(m["id"] for m in meetings)
    export = httpx.get(f"{BASE}/api/meetings/{first}/export?fmt=md", timeout=5)
    assert export.status_code == 200
    assert "Транскрипт" in export.text and "коллеги" in export.text.lower()

    silent = [m for m in meetings if m["title"] == "e2e: тишина"][0]
    deleted = httpx.delete(f"{BASE}/api/meetings/{silent['id']}", timeout=5)
    assert deleted.json()["deleted"] == silent["id"]


def test_10_asr_switch_rules(server, voices):
    """Переключение модели: 409 во время live-встречи, 400 на невалидную пару,
    выбор персистится в asr.json."""

    def try_switch_during_live(_meeting_id):
        response = httpx.post(f"{BASE}/api/asr",
                              json={"engine": "faster_whisper", "model": "base"}, timeout=10)
        assert response.status_code == 409, response.text

    run_meeting("e2e: переключение при live",
                system=np.concatenate([voices["milena"], silence(1.2)]),
                during=try_switch_during_live)

    bad = httpx.post(f"{BASE}/api/asr",
                     json={"engine": "gigaam", "model": "small"}, timeout=10)
    assert bad.status_code == 400

    ok = httpx.post(f"{BASE}/api/asr",
                    json={"engine": "faster_whisper", "model": "base"}, timeout=60)
    assert ok.status_code == 200
    _wait_loaded(timeout=120)
    saved = json.loads((server["data_dir"] / "asr.json").read_text())
    assert saved == {"engine": "faster_whisper", "model": "base"}

    back = httpx.post(f"{BASE}/api/asr",
                      json={"engine": "gigaam", "model": "v3_e2e_rnnt"}, timeout=60)
    assert back.status_code == 200
    _wait_loaded(timeout=120)
