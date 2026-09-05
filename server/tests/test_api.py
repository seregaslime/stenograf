"""Компонентные тесты REST-API: эндпоинты через FastAPI TestClient.

Это НЕ функциональные тесты «чёрным ящиком», и важно не выдавать их за таковые:
TestClient вызывает приложение внутри процесса (без сети и без живого сервера),
а данные для read-эндпоинтов засеваются прямо в БД через crud — то есть в обход
публичного интерфейса. Проверяются коды ответов и форма данных.

Функциональную проверку — систему целиком глазами пользователя — даёт
tests/test_e2e_live.py: настоящий uvicorn, WebSocket, встреча от аудио до REST.
"""
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.db import crud
from app.db.database import init_db, session_scope
from app.db.models import Meeting


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture()
def done_meeting():
    with session_scope() as db:
        m = crud.create_meeting(db, "Тестовая встреча", False)
        s = crud.get_or_create_self_speaker(db)
        crud.add_segment(db, m.id, s.id, "mic", 0.0, 1.0, "привет коллеги")
        crud.end_meeting(db, m.id, status="done")
        return m.id


@pytest.fixture()
def live_meeting():
    with session_scope() as db:
        return crud.create_meeting(db, "Идёт встреча", False).id  # статус live


def test_startup_closes_meetings_stuck_in_summarizing():
    """Сервер убили посреди резюме — при следующем старте встреча не висит.

    Довести её до конца больше некому: фоновая задача умерла вместе с процессом,
    а клиент опрашивал бы статус «summarizing» до бесконечности.
    """
    init_db()  # тест может идти первым в файле — схемы ещё нет, lifespan не поднимался
    with session_scope() as db:
        meeting = crud.create_meeting(db, "Прерванная", False)
        speaker = crud.get_or_create_self_speaker(db)
        crud.add_segment(db, meeting.id, speaker.id, "mic", 0.0, 1.0, "привет")
        crud.end_meeting(db, meeting.id, status="summarizing")
        meeting_id, ended_at = meeting.id, meeting.ended_at

    with TestClient(main.app):  # вход в контекст прогоняет lifespan
        pass

    with session_scope() as db:
        meeting = db.get(Meeting, meeting_id)
        assert meeting.status == "done"
        assert "перезапустился" in meeting.summary_error
        # Встреча кончилась раньше — время окончания не переписываем.
        # tzinfo снимаем с обеих сторон: SQLite часовой пояс не хранит, и
        # прочитанное из базы значение всегда naive.
        assert meeting.ended_at.replace(tzinfo=None) == ended_at.replace(tzinfo=None)


# ------------------------------------------------------------------ health / asr / llm

def test_health(client):
    """/api/health отдаёт статус сервера, состояние ASR, диаризации и выбранного провайдера LLM.
    """
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert set(body["asr"]) >= {"engine", "model", "loaded"}
    # Про модели языка здоровье больше не рассказывает: их адрес и выбор живут
    # в приложении, сервер о них не знает.
    assert "llm" not in body and "ollama" not in body


def test_asr_state(client):
    """/api/asr сообщает текущий движок, модель и список доступных моделей по движкам."""
    body = client.get("/api/asr").json()
    assert body["engine"] and body["model"]
    assert "faster_whisper" in body["models_by_engine"]


# ------------------------------------------------------------------ meetings

def test_meetings_list_and_get(client, done_meeting):
    """Список и карточка встречи отдают сохранённые данные; несуществующий id даёт 404."""
    listed = client.get("/api/meetings").json()
    assert any(m["id"] == done_meeting for m in listed)

    detail = client.get(f"/api/meetings/{done_meeting}").json()
    assert detail["title"] == "Тестовая встреча"
    assert detail["segments"][0]["text"] == "привет коллеги"

    assert client.get("/api/meetings/999999").status_code == 404


def test_export_md_and_txt(client, done_meeting):
    """Выгрузка встречи работает в обоих форматах с правильным Content-Type и содержит транскрипт.
    """
    md = client.get(f"/api/meetings/{done_meeting}/export?fmt=md")
    assert md.status_code == 200
    assert md.headers["content-type"].startswith("text/markdown")
    assert "Транскрипт" in md.text and "привет коллеги" in md.text

    txt = client.get(f"/api/meetings/{done_meeting}/export?fmt=txt")
    assert txt.status_code == 200
    assert txt.headers["content-type"].startswith("text/plain")
    assert "привет коллеги" in txt.text


def test_delete_meeting(client, done_meeting):
    """Встреча удаляется, повторное удаление даёт 404."""
    assert client.delete(f"/api/meetings/{done_meeting}").json()["deleted"] == done_meeting
    assert client.delete(f"/api/meetings/{done_meeting}").status_code == 404  # повторно


# ------------------------------------------------------------------ speakers

def test_speaker_rename(client):
    """Спикер переименовывается; для несуществующего id — 404."""
    with session_scope() as db:
        sid = crud.create_speaker(db).id
    body = client.patch(f"/api/speakers/{sid}", json={"name": "Илья"}).json()
    assert body["name"] == "Илья"
    assert client.patch("/api/speakers/999999", json={"name": "X"}).status_code == 404


def test_delete_self_speaker_forbidden(client):
    """Профиль владельца «Вы» удалить нельзя — 400."""
    with session_scope() as db:
        self_id = crud.get_or_create_self_speaker(db).id
    assert client.delete(f"/api/speakers/{self_id}").status_code == 400  # «Вы» удалять нельзя


def test_delete_missing_speaker_404(client):
    """Удаление несуществующего спикера даёт 404."""
    assert client.delete("/api/speakers/999999").status_code == 404


def test_merge_needs_two_speakers(client):
    """Слияние профилей требует ровно двух id — иначе 400."""
    assert client.post("/api/speakers/merge", json={"speaker_ids": [1]}).status_code == 400


def test_merge_keeps_richer_profile_and_moves_segments(client):
    """Целевым становится профиль с бо́льшим числом реплик (см. registry.merge),
    реплики второго переезжают к нему, сам он исчезает из списка."""
    with session_scope() as db:
        meeting = crud.create_meeting(db, "M", False)
        rich = crud.create_speaker(db)
        poor = crud.create_speaker(db)
        for i in range(2):
            crud.add_segment(db, meeting.id, rich.id, "mic", float(i), i + 1.0, "много")
        crud.add_segment(db, meeting.id, poor.id, "system", 5.0, 6.0, "мало")
        rich_id, poor_id = rich.id, poor.id

    result = client.post("/api/speakers/merge", json={"speaker_ids": [poor_id, rich_id]})
    assert result.status_code == 200
    body = result.json()
    assert body["target_id"] == rich_id      # победил тот, у кого реплик больше
    assert body["moved_segments"] == 1       # переехала единственная реплика второго

    remaining = {s["id"] for s in client.get("/api/speakers").json()}
    assert rich_id in remaining and poor_id not in remaining


def test_merge_same_speaker_rejected(client):
    """Слить профиль сам с собой нельзя — 400."""
    with session_scope() as db:
        sid = crud.create_speaker(db).id
    assert client.post("/api/speakers/merge", json={"speaker_ids": [sid, sid]}).status_code == 400


def test_voiceprint_audio_404_when_absent(client):
    """Аудио несуществующего отпечатка даёт 404, а не падение с трейсбеком."""
    assert client.get("/api/speakers/1/voiceprints/999999/audio").status_code == 404


def test_delete_missing_voiceprint_404(client):
    """Удаление несуществующего отпечатка даёт 404."""
    assert client.delete("/api/speakers/1/voiceprints/999999").status_code == 404


def test_meeting_detail_reports_mode(client, done_meeting):
    """Карточка встречи отдаёт тип встречи — клиент показывает его чипом."""
    assert client.get(f"/api/meetings/{done_meeting}").json()["meeting_mode"] == "work"


def test_export_unknown_meeting_404(client):
    """Выгрузка несуществующей встречи даёт 404."""
    assert client.get("/api/meetings/999999/export?fmt=md").status_code == 404


# ------------------------------------------------------------------ поиск по встречам

