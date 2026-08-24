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
from app.db.database import init_db, session_scope
from app.db.models import Meeting
from app.llm import ollama_client as ollama_mod
from app.llm import openai_client as openai_mod
from app.llm.base import LlmError

# Поддерживается только Groq: остальные провайдеры не сообщают контекст модели
GROQ = "https://api.groq.com/openai/v1"


def _mock_openai(monkeypatch, handler):
    """Подменяет httpx у OpenAI-клиента на MockTransport (без сети)."""
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(openai_mod.httpx, "AsyncClient", factory)


def _mock_ollama(monkeypatch, handler):
    """То же для клиента Ollama: /api/tags отвечает мок, а не живой демон."""
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(ollama_mod.httpx, "AsyncClient", factory)


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
        json={"api_base_url": GROQ, "api_key": "probe-key"},
    ).json()
    assert body["reachable"] is True
    assert body["models"] == ["m1", "m2"]


def test_llm_get_includes_local_settings(client):
    """/api/llm отдаёт настройки Ollama — форма показывает их и когда активен api."""
    body = client.get("/api/llm").json()
    assert set(body) >= {"ollama_url", "local_summary_model", "local_hints_model"}
    assert body["ollama_url"].startswith("http")


def test_llm_probe_ollama_models(client, monkeypatch):
    """/api/llm/ollama/models спрашивает список моделей по ещё не сохранённому адресу."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        assert request.url.host == "ollama.corp"
        return httpx.Response(200, json={"models": [{"name": "qwen3:4b"}, {"name": "qwen3:1.7b"}]})

    _mock_ollama(monkeypatch, handler)
    body = client.post(
        "/api/llm/ollama/models", json={"ollama_url": "http://ollama.corp:11434"},
    ).json()
    assert body["reachable"] is True
    assert body["models"] == ["qwen3:4b", "qwen3:1.7b"]


def test_llm_probe_ollama_bad_url_400(client):
    """Адрес без схемы http(s) отклоняется до запроса, а не молча превращается в ошибку связи."""
    response = client.post("/api/llm/ollama/models", json={"ollama_url": "file:///etc/passwd"})
    assert response.status_code == 400


def test_llm_set_local_with_settings(client, monkeypatch):
    """Адрес Ollama и её модели принимаются из настроек приложения и возвращаются обратно."""
    monkeypatch.setattr(config.settings, "ollama_url", "http://127.0.0.1:11434")
    body = client.post("/api/llm", json={
        "provider": "local",
        "ollama_url": "http://ollama.corp:11434",
        "local_summary_model": "qwen3:8b",
        "local_hints_model": "qwen3:4b",
    }).json()
    assert body["ollama_url"] == "http://ollama.corp:11434"
    assert body["local_summary_model"] == "qwen3:8b"


def test_llm_set_local_bad_ollama_url_400(client):
    """Мусорный адрес Ollama не сохраняется — 400 с объяснением."""
    response = client.post(
        "/api/llm", json={"provider": "local", "ollama_url": "просто текст"},
    )
    assert response.status_code == 400


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
            "provider": "api", "api_base_url": GROQ,
            "api_key": "secret", "summary_model": "m1", "hints_model": "m2",
        }).json()
        assert body["provider"] == "api"
        assert body["api_base_url"] == GROQ
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


def test_llm_probe_rejects_unsupported_host(client):
    """Чужой провайдер не сообщает размер контекста — принять его значит
    пустить пользователя выбирать модель вслепую."""
    resp = client.post("/api/llm/models",
                       json={"api_base_url": "https://api.openai.com/v1", "api_key": "k"})
    assert resp.status_code == 400
    assert "Groq" in resp.json()["detail"]


def test_llm_probe_filters_models(client, monkeypatch):
    """В списке выбора остаются только пригодные модели, негодные посчитаны."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [
            {"id": "good", "context_window": 131072,
             "input_modalities": ["text"], "output_modalities": ["text"]},
            {"id": "tiny-context", "context_window": 512,
             "input_modalities": ["text"], "output_modalities": ["text"]},
            {"id": "speech-out", "context_window": 131072,
             "input_modalities": ["text"], "output_modalities": ["speech"]},
        ]})

    _mock_openai(monkeypatch, handler)
    body = client.post("/api/llm/models",
                       json={"api_base_url": GROQ, "api_key": "k"}).json()
    assert body["models"] == ["good"]
    assert body["models_rejected"] == 2
    assert body["models_info"][0]["context_window"] == 131072


# ------------------------------------------------------------------ поиск по встречам

def test_search_indexes_and_finds(client, done_meeting, monkeypatch):
    """/api/search сам индексирует прошедшие встречи и возвращает цитату со ссылкой.

    Индексация ленивая: встречи могли пройти до появления поиска, и требовать
    от пользователя «нажмите переиндексировать» — значит гарантировать, что
    поиск у него не заработает.
    """
    # Вектор задаётся по содержимому: иначе у всех кусков в общей тестовой базе
    # он одинаковый, близость у всех единица, и в выдачу попадают случайные пять.
    async def embed(self, model, texts):
        return [[1.0, 0.0] if "привет коллеги" in т else [0.0, 1.0] for т in texts]

    monkeypatch.setattr(ollama_mod.OllamaClient, "embed", embed)
    body = client.get("/api/search", params={"q": "привет коллеги, о чём говорили"}).json()
    найдено = [r for r in body["results"] if r["meeting_id"] == done_meeting]
    assert найдено and найдено[0]["text"] == "привет коллеги"
    assert найдено[0]["meeting_title"] == "Тестовая встреча"


def test_search_empty_query_returns_empty(client, monkeypatch):
    """Пустой запрос не будит модель эмбеддингов — она грузится в память секундами."""
    async def embed(self, model, texts):
        raise AssertionError("к модели ходить не должны")

    monkeypatch.setattr(ollama_mod.OllamaClient, "embed", embed)
    assert client.get("/api/search", params={"q": "  "}).json()["results"] == []


@pytest.mark.parametrize("limit", [0, -3, 999])
def test_search_rejects_absurd_limit(client, limit):
    """limit вне разумных границ отклоняется на входе.

    Отрицательный особенно коварен: он не падает, а превращает срез в «все
    результаты, кроме последних трёх» — на большом архиве это мегабайты ответа
    вместо пяти цитат.
    """
    assert client.get("/api/search", params={"q": "бюджет", "limit": limit}).status_code == 422


def test_search_without_model_answers_503(client, done_meeting, monkeypatch):
    """Модель эмбеддингов не скачана — 503 с командой, которая это чинит,
    а не пустая выдача: пустая выглядит как «ничего не нашлось»."""
    async def embed(self, model, texts):
        raise LlmError("Модель «bge-m3» не найдена в Ollama. Скачайте её: `ollama pull bge-m3`.")

    monkeypatch.setattr(ollama_mod.OllamaClient, "embed", embed)
    response = client.get("/api/search", params={"q": "бюджет"})
    assert response.status_code == 503
    assert "ollama pull" in response.json()["detail"]


def test_llm_state_offers_default_base_url(client):
    """Клиент подставляет адрес в поле из ответа сервера, а не хранит свою
    копию — иначе она однажды разъедется со списком разрешённых хостов."""
    from app.config import LLM_API_DEFAULT_BASE_URL, api_host_supported

    body = client.get("/api/llm").json()
    assert body["api_base_url_default"] == LLM_API_DEFAULT_BASE_URL
    # предлагаемый адрес обязан проходить собственную же валидацию
    assert api_host_supported(body["api_base_url_default"])
