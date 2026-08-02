"""Компонентные тесты REST-API: эндпоинты через FastAPI TestClient.

Это НЕ функциональные тесты «чёрным ящиком», и важно не выдавать их за таковые:
TestClient вызывает приложение внутри процесса (без сети и без живого сервера),
а данные для read-эндпоинтов засеваются прямо в БД через crud — то есть в обход
публичного интерфейса. Проверяются коды ответов и форма данных.

Функциональную проверку — систему целиком глазами пользователя — даёт
tests/test_e2e_live.py: настоящий uvicorn, WebSocket, встреча от аудио до REST.
"""
import httpx
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app import config
from app.db import crud
from app.db.database import session_scope
from app.llm import openai_client as openai_mod


def _mock_openai(monkeypatch, handler):
    """Подменяет httpx у OpenAI-клиента на MockTransport (без сети)."""
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(openai_mod.httpx, "AsyncClient", factory)


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
    """/api/health отдаёт статус сервера, состояние ASR, диаризации и выбранного провайдера LLM.
    """
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert set(body["asr"]) >= {"engine", "model", "loaded"}
    assert body["llm"]["provider"] in ("local", "api")
    assert "summary_model" in body and "hints_model" in body


def test_asr_state(client):
    """/api/asr сообщает текущий движок, модель и список доступных моделей по движкам."""
    body = client.get("/api/asr").json()
    assert body["engine"] and body["model"]
    assert "faster_whisper" in body["models_by_engine"]


def test_llm_get(client):
    """/api/llm отдаёт состояние провайдера LLM — и намеренно не содержит API-ключ."""
    body = client.get("/api/llm").json()
    assert body["provider"] in ("local", "api")
    assert set(body) >= {"provider", "api_configured", "summary_model", "hints_model", "reachable"}


def test_llm_set_local_ok(client):
    """Переключение на локальную модель принимается всегда: ей не нужны ни адрес, ни ключ."""
    body = client.post("/api/llm", json={"provider": "local"}).json()
    assert body["provider"] == "local"


def test_llm_set_unknown_provider_400(client):
    """Неизвестный провайдер отклоняется с кодом 400, а не сохраняется молча."""
    assert client.post("/api/llm", json={"provider": "gguf"}).status_code == 400


def test_llm_set_api_unconfigured_400(client, monkeypatch):
    """Включить внешний API без адреса и ключа нельзя — 400 с объяснением."""
    monkeypatch.setattr(config.settings, "llm_api_base_url", "")
    monkeypatch.setattr(config.settings, "llm_api_key", "")
    assert client.post("/api/llm", json={"provider": "api"}).status_code == 400


def test_llm_probe_models(client, monkeypatch):
    """/api/llm/models запрашивает список моделей по введённым (ещё не сохранённым) кредам — для выпадающего списка в настройках.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        assert request.headers.get("authorization") == "Bearer probe-key"
        return httpx.Response(200, json={"data": [{"id": "m1"}, {"id": "m2"}]})

    _mock_openai(monkeypatch, handler)
    body = client.post(
        "/api/llm/models",
        json={"api_base_url": "http://api.local/v1", "api_key": "probe-key"},
    ).json()
    assert body["reachable"] is True
    assert body["models"] == ["m1", "m2"]


def test_llm_set_api_with_creds(client, monkeypatch):
    # креды приходят из настроек приложения (не из .env)
    """Адрес, ключ и модели принимаются из настроек приложения; ключ в ответе не возвращается.
    """
    monkeypatch.setattr(config.settings, "llm_api_base_url", "")
    monkeypatch.setattr(config.settings, "llm_api_key", "")
    monkeypatch.setattr(config.settings, "llm_provider", "local")
    _mock_openai(monkeypatch, lambda r: httpx.Response(200, json={"data": []}))
    try:
        body = client.post("/api/llm", json={
            "provider": "api", "api_base_url": "http://api.local/v1",
            "api_key": "secret", "summary_model": "m1", "hints_model": "m2",
        }).json()
        assert body["provider"] == "api"
        assert body["api_base_url"] == "http://api.local/v1"
        assert body["summary_model"] == "m1" and body["hints_model"] == "m2"
        assert "api_key" not in body  # ключ клиенту не отдаём
    finally:
        config.settings.llm_provider = "local"  # не тащить api в остальные тесты


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


def test_summarize_sets_status(client, done_meeting, monkeypatch):
    """Запрос протокола переводит встречу в статус summarizing; для несуществующей встречи — 404.
    """
    monkeypatch.setattr(main, "_schedule_summary", lambda mid: None)  # не гонять LLM
    resp = client.post(f"/api/meetings/{done_meeting}/summarize")
    assert resp.status_code == 200 and resp.json()["status"] == "summarizing"
    assert client.get(f"/api/meetings/{done_meeting}").json()["status"] == "summarizing"
    assert client.post("/api/meetings/999999/summarize").status_code == 404


def test_summarize_live_conflict(client, live_meeting, monkeypatch):
    """Составить протокол по ещё идущей встрече нельзя — 409."""
    monkeypatch.setattr(main, "_schedule_summary", lambda mid: None)
    assert client.post(f"/api/meetings/{live_meeting}/summarize").status_code == 409


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
