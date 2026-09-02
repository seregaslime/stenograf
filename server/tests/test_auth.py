"""Токены доступа: кого сервер пускает и что отдаёт чужому.

Тесты чистят за собой людей в finally не из аккуратности, а по необходимости:
база у функциональных тестов одна на весь прогон, и оставленный пользователь
включил бы обязательный токен для всех остальных тестов сразу.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from starlette.websockets import WebSocketDisconnect

from app.ws import LiveSession

import app.main as main
from app import auth
from app.db import crud
from app.db.database import session_scope
from app.db.models import Meeting, User


@pytest.fixture()
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture()
def человек():
    """Заводит человека и отдаёт его токен, после теста — убирает."""
    with session_scope() as db:
        _, token = auth.create_user(db, "Сергей")
    yield token
    with session_scope() as db:
        for user in db.scalars(select(User)):
            db.delete(user)


# --- разбор заголовка ---

@pytest.mark.parametrize("заголовок, ожидание", [
    ("Bearer abc", "abc"),
    ("bearer abc", "abc"),        # схема регистронезависима
    ("Bearer  abc ", "abc"),
    ("Basic abc", None),          # чужая схема — не наш токен
    ("abc", None),                # без схемы
    ("Bearer", None),
    ("", None),
    (None, None),
])
def test_разбор_заголовка(заголовок, ожидание):
    assert auth.token_from_header(заголовок) == ожидание


# --- режим без людей: сервер личный ---

def test_без_людей_пускает_без_токена(client):
    assert client.get("/api/meetings").status_code == 200


# --- режим с людьми: токен обязателен ---

def test_без_токена_401(client, человек):
    ответ = client.get("/api/meetings")
    assert ответ.status_code == 401
    assert "токен" in ответ.json()["detail"].lower()


def test_чужой_токен_401(client, человек):
    # Токен латиницей намеренно: в заголовок HTTP кириллица не помещается
    # (httpx кодирует значения в ascii), а наши токены — urlsafe base64.
    ответ = client.get("/api/meetings", headers={"Authorization": "Bearer wrong-token-42"})
    assert ответ.status_code == 401


def test_свой_токен_пускает(client, человек):
    ответ = client.get("/api/meetings", headers={"Authorization": f"Bearer {человек}"})
    assert ответ.status_code == 200


def test_закрыты_и_настройки(client, человек):
    """Не только чтение встреч: смена провайдера LLM чужому тоже недоступна."""
    assert client.get("/api/llm").status_code == 401
    assert client.get("/api/speakers").status_code == 401


# --- здоровье: открыто, но без подробностей ---

def test_здоровье_без_токена_урезано(client, человек):
    тело = client.get("/api/health").json()
    assert тело["status"] == "ok"
    assert тело["authorized"] is False
    assert "asr" not in тело and "llm" not in тело


def test_здоровье_с_токеном_полное(client, человек):
    тело = client.get("/api/health", headers={"Authorization": f"Bearer {человек}"}).json()
    assert тело["authorized"] is True
    assert "asr" in тело and "llm" in тело


def test_здоровье_со_слэшем_тоже_открыто(client, человек):
    """«/api/health/» — тот же health: мониторинг с привычным слэшем не должен
    видеть упавший сервер вместо живого."""
    ответ = client.get("/api/health/", follow_redirects=False)
    assert ответ.status_code != 401


def test_автодокументация_закрыта(client, человек):
    """Под /api она не попадает, но карту сервера постороннему отдавать незачем."""
    assert client.get("/openapi.json").status_code == 401
    assert client.get("/docs").status_code == 401


# --- хранение ---

def test_токен_хранится_хешем(человек):
    with session_scope() as db:
        user = db.scalar(select(User))
        assert user.token_hash != человек
        assert user.token_hash == auth.hash_token(человек)


def test_токены_разные():
    assert auth.create_token() != auth.create_token()


# --- живая встреча: токен первым кадром ---
#
# Заголовок сюда поставить нельзя: браузерный WebSocket их не умеет, а в адресе
# токен попал бы в журналы сервера. Поэтому первый кадр — auth.

def test_ws_без_людей_пускает_без_токена(client):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "не-такая-команда"})
        assert ws.receive_json()["type"] == "error"  # цикл сессии работает


def test_ws_с_чужим_токеном_закрывается(client, человек):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "auth", "token": "wrong-token"})
        событие = ws.receive_json()
        assert событие["type"] == "error"
        assert "токен" in событие["message"].lower()
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_ws_с_пустым_токеном_отвечает_сразу(client, человек):
    """Клиент шлёт кадр auth даже без токена — чтобы отказ пришёл сразу, а не
    через десять секунд ожидания, пока человек уже говорит."""
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "auth", "token": ""})
        assert ws.receive_json()["type"] == "error"
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_ws_без_кадра_auth_закрывается(client, человек):
    """Первым кадром пришло аудио, а не токен — соединение не начинается."""
    with client.websocket_connect("/ws/live") as ws:
        ws.send_bytes(b"\x00" + b"\x00" * 320)
        assert ws.receive_json()["type"] == "error"
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_ws_молчание_обрывается_по_таймауту(client, человек, monkeypatch):
    """Молчащее соединение не висит вечно: иначе их пачкой занимают сервер,
    не зная токена. Таймаут в тесте укорочен, чтобы не ждать десять секунд."""
    monkeypatch.setattr(LiveSession, "AUTH_TIMEOUT_S", 0.05)
    with client.websocket_connect("/ws/live") as ws:
        assert ws.receive_json()["type"] == "error"
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_ws_со_своим_токеном_работает(client, человек):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "auth", "token": человек})
        ws.send_json({"type": "не-такая-команда"})
        событие = ws.receive_json()
        assert событие["type"] == "error"
        assert "Неизвестная команда" in событие["message"]


def test_ws_повторный_auth_не_ошибка(client, человек):
    """Клиент может прислать auth ещё раз — это не «неизвестная команда»."""
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "auth", "token": человек})
        ws.send_json({"type": "auth", "token": человек})
        ws.send_json({"type": "не-такая-команда"})
        assert "Неизвестная команда" in ws.receive_json()["message"]


# --- разделение данных: у каждого свои встречи ---

def test_первый_человек_получает_ничейные_встречи(db_session):
    """Полгода работы на личном сервере не должны исчезнуть в момент, когда его
    закрывают токеном: записи на месте, но не видны никому — это выглядит как
    потеря архива, а не как защита."""
    старая = crud.create_meeting(db_session, "До токенов", False)
    assert старая.owner_id is None

    сергей, _ = auth.create_user(db_session, "Сергей")
    assert db_session.get(Meeting, старая.id).owner_id == сергей.id


def test_второй_человек_чужого_не_получает(db_session):
    """Усыновление разовое: иначе каждый новый сотрудник забирал бы себе всё,
    что успело осиротеть."""
    сергей, _ = auth.create_user(db_session, "Сергей")
    его = crud.create_meeting(db_session, "Планёрка", False, owner_id=сергей.id)

    куратор, _ = auth.create_user(db_session, "Куратор")
    assert db_session.get(Meeting, его.id).owner_id == сергей.id


def test_список_встреч_только_свои(db_session):
    сергей, _ = auth.create_user(db_session, "Сергей")
    куратор, _ = auth.create_user(db_session, "Куратор")
    crud.create_meeting(db_session, "Моя", False, owner_id=сергей.id)
    crud.create_meeting(db_session, "Чужая", False, owner_id=куратор.id)

    названия = [m["title"] for m in crud.list_meetings(db_session, сергей.id)]
    assert названия == ["Моя"]
    # Без владельца (личный сервер) фильтровать не по чему — видно всё
    assert len(crud.list_meetings(db_session, None)) == 2


def test_чужая_встреча_неотличима_от_несуществующей(db_session):
    """404, а не 403: по разнице между ними перебором узнаётся, сколько встреч
    у соседа и когда они шли."""
    сергей, _ = auth.create_user(db_session, "Сергей")
    куратор, _ = auth.create_user(db_session, "Куратор")
    чужая = crud.create_meeting(db_session, "Чужая", False, owner_id=куратор.id)

    assert crud.meeting_for_owner(db_session, чужая.id, сергей.id) is None
    assert crud.meeting_for_owner(db_session, 9999, сергей.id) is None
    assert crud.meeting_for_owner(db_session, чужая.id, куратор.id) is not None
    # Личный сервер: владельца нет, доступно всё
    assert crud.meeting_for_owner(db_session, чужая.id, None) is not None


def test_двое_через_http_видят_только_своё(client):
    """Тот же запрет, но через реальный конвейер: middleware → эндпоинт → БД."""
    with session_scope() as db:
        сергей, токен_сергея = auth.create_user(db, "Сергей")
        куратор, токен_куратора = auth.create_user(db, "Куратор")
        crud.create_meeting(db, "Планёрка Сергея", False, owner_id=сергей.id)
        crud.create_meeting(db, "Разбор куратора", False, owner_id=куратор.id)
    try:
        свои = client.get("/api/meetings",
                          headers={"Authorization": f"Bearer {токен_сергея}"}).json()
        чужие = client.get("/api/meetings",
                           headers={"Authorization": f"Bearer {токен_куратора}"}).json()
        assert [m["title"] for m in свои] == ["Планёрка Сергея"]
        assert [m["title"] for m in чужие] == ["Разбор куратора"]

        # И поштучно: чужая встреча отвечает «не найдена», а не «нельзя»
        чужой_id = чужие[0]["id"]
        ответ = client.get(f"/api/meetings/{чужой_id}",
                           headers={"Authorization": f"Bearer {токен_сергея}"})
        assert ответ.status_code == 404
    finally:
        with session_scope() as db:
            for встреча in db.scalars(select(Meeting)):
                db.delete(встреча)
            for человек in db.scalars(select(User)):
                db.delete(человек)


def test_повторное_удаление_не_роняет_сервер(client):
    """Проверка прав и удаление — в разных сессиях, между ними встречу могли
    удалить (двойной клик, два клиента). Второй запрос должен ответить, а не
    упасть на None."""
    with session_scope() as db:
        meeting_id = crud.create_meeting(db, "Дважды удалённая", False).id
    первый = client.delete(f"/api/meetings/{meeting_id}")
    второй = client.delete(f"/api/meetings/{meeting_id}")
    assert первый.status_code == 200
    assert второй.status_code in (200, 404)
