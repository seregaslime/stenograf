"""Функциональные тесты REST-API (чёрный ящик через FastAPI TestClient).

Проверяем поведение эндпоинтов против требований: форма ответа и коды ошибок.
Данные для read-эндпоинтов засеваем прямо в (временную) БД через crud — REST
создаёт встречи только по WebSocket, что покрыто e2e-тестами.
"""
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app import config
from app.db import crud
from app.db.database import session_scope


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


# ------------------------------------------------------------------ health / asr / llm

def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert set(body["asr"]) >= {"engine", "model", "loaded"}
    assert body["llm"]["provider"] in ("local", "api")
    assert "summary_model" in body and "hints_model" in body


def test_asr_state(client):
    body = client.get("/api/asr").json()
    assert body["engine"] and body["model"]
    assert "faster_whisper" in body["models_by_engine"]


def test_llm_get(client):
    body = client.get("/api/llm").json()
    assert body["provider"] in ("local", "api")
    assert set(body) >= {"provider", "api_configured", "summary_model", "hints_model", "reachable"}


def test_llm_set_local_ok(client):
    body = client.post("/api/llm", json={"provider": "local"}).json()
    assert body["provider"] == "local"


def test_llm_set_unknown_provider_400(client):
    assert client.post("/api/llm", json={"provider": "gguf"}).status_code == 400


def test_llm_set_api_unconfigured_400(client, monkeypatch):
    monkeypatch.setattr(config.settings, "llm_api_base_url", "")
    monkeypatch.setattr(config.settings, "llm_api_key", "")
    assert client.post("/api/llm", json={"provider": "api"}).status_code == 400


# ------------------------------------------------------------------ meetings

def test_meetings_list_and_get(client, done_meeting):
    listed = client.get("/api/meetings").json()
    assert any(m["id"] == done_meeting for m in listed)

    detail = client.get(f"/api/meetings/{done_meeting}").json()
    assert detail["title"] == "Тестовая встреча"
    assert detail["segments"][0]["text"] == "привет коллеги"

    assert client.get("/api/meetings/999999").status_code == 404


def test_export_md_and_txt(client, done_meeting):
    md = client.get(f"/api/meetings/{done_meeting}/export?fmt=md")
    assert md.status_code == 200
    assert md.headers["content-type"].startswith("text/markdown")
    assert "Транскрипт" in md.text and "привет коллеги" in md.text

    txt = client.get(f"/api/meetings/{done_meeting}/export?fmt=txt")
    assert txt.status_code == 200
    assert txt.headers["content-type"].startswith("text/plain")
    assert "привет коллеги" in txt.text


def test_summarize_sets_status(client, done_meeting, monkeypatch):
    monkeypatch.setattr(main, "_schedule_summary", lambda mid: None)  # не гонять LLM
    resp = client.post(f"/api/meetings/{done_meeting}/summarize")
    assert resp.status_code == 200 and resp.json()["status"] == "summarizing"
    assert client.get(f"/api/meetings/{done_meeting}").json()["status"] == "summarizing"
    assert client.post("/api/meetings/999999/summarize").status_code == 404


def test_summarize_live_conflict(client, live_meeting, monkeypatch):
    monkeypatch.setattr(main, "_schedule_summary", lambda mid: None)
    assert client.post(f"/api/meetings/{live_meeting}/summarize").status_code == 409


def test_delete_meeting(client, done_meeting):
    assert client.delete(f"/api/meetings/{done_meeting}").json()["deleted"] == done_meeting
    assert client.delete(f"/api/meetings/{done_meeting}").status_code == 404  # повторно


# ------------------------------------------------------------------ speakers

def test_speaker_rename(client):
    with session_scope() as db:
        sid = crud.create_speaker(db).id
    body = client.patch(f"/api/speakers/{sid}", json={"name": "Илья"}).json()
    assert body["name"] == "Илья"
    assert client.patch("/api/speakers/999999", json={"name": "X"}).status_code == 404


def test_delete_self_speaker_forbidden(client):
    with session_scope() as db:
        self_id = crud.get_or_create_self_speaker(db).id
    assert client.delete(f"/api/speakers/{self_id}").status_code == 400  # «Вы» удалять нельзя


def test_delete_missing_speaker_404(client):
    assert client.delete("/api/speakers/999999").status_code == 404


def test_merge_needs_two_speakers(client):
    assert client.post("/api/speakers/merge", json={"speaker_ids": [1]}).status_code == 400
