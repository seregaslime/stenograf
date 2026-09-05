"""Сервер принимает протокол, составленный клиентом.

Раньше протокол считал сам сервер. Теперь генерация уезжает к человеку — у
каждого свой адрес модели, свой ключ и свой лимит, — а серверу остаётся принять
результат и проверить, что подписывают свою встречу, а не чужую.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

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
def встреча():
    with session_scope() as db:
        m = crud.create_meeting(db, "Планёрка", False)
        s = crud.get_or_create_self_speaker(db)
        crud.add_segment(db, m.id, s.id, "mic", 0.0, 1.0, "давайте по срокам")
        crud.end_meeting(db, m.id, status="done")
        return m.id


@pytest.fixture()
def двое():
    with session_scope() as db:
        сергей, токен_с = auth.create_user(db, "Сергей")
        куратор, токен_к = auth.create_user(db, "Куратор")
        чужая = crud.create_meeting(db, "Чужая", False, owner_id=куратор.id)
        crud.end_meeting(db, чужая.id, status="done")
        данные = {"токен": токен_с, "чужая": чужая.id}
    yield данные
    with session_scope() as db:
        for таблица in (Meeting, User):
            for строка in db.scalars(select(таблица)):
                db.delete(строка)


def test_протокол_сохраняется(client, встреча):
    ответ = client.post(f"/api/meetings/{встреча}/summary",
                        json={"text": "## Краткий итог\nПеренесли демо.", "model": "qwen3:4b"})
    assert ответ.status_code == 200
    assert ответ.json() == {"status": "done", "has_summary": True}

    встреча_целиком = client.get(f"/api/meetings/{встреча}").json()
    assert встреча_целиком["summary"].startswith("## Краткий итог")
    assert встреча_целиком["summary_model"] == "qwen3:4b"
    assert встреча_целиком["summary_error"] is None


def test_успех_затирает_прошлую_ошибку(client, встреча):
    client.post(f"/api/meetings/{встреча}/summary", json={"error": "модель не ответила"})
    assert client.get(f"/api/meetings/{встреча}").json()["summary_error"] == "модель не ответила"

    client.post(f"/api/meetings/{встреча}/summary", json={"text": "## Итог\nГотово."})
    встреча_целиком = client.get(f"/api/meetings/{встреча}").json()
    assert встреча_целиком["summary_error"] is None
    assert встреча_целиком["summary"] == "## Итог\nГотово."


def test_неудача_не_стирает_готовый_протокол(client, встреча):
    """Неудачная попытка пересоздать не должна оставлять человека без того,
    что уже было составлено."""
    client.post(f"/api/meetings/{встреча}/summary", json={"text": "## Итог\nПервый."})
    client.post(f"/api/meetings/{встреча}/summary", json={"error": "лимит токенов"})

    встреча_целиком = client.get(f"/api/meetings/{встреча}").json()
    assert встреча_целиком["summary"] == "## Итог\nПервый."
    assert встреча_целиком["summary_error"] == "лимит токенов"


def test_пустое_тело_отбивается(client, встреча):
    ответ = client.post(f"/api/meetings/{встреча}/summary", json={"text": "   "})
    assert ответ.status_code == 400


def test_идущую_встречу_подписать_нельзя(client):
    with session_scope() as db:
        живая = crud.create_meeting(db, "Идёт", False).id
    assert client.post(f"/api/meetings/{живая}/summary",
                       json={"text": "рано"}).status_code == 409


def test_чужую_встречу_подписать_нельзя(client, двое):
    """Без этой проверки любой, у кого есть токен, подписал бы чужой встрече
    любой текст — а протокол читают те, кого на встрече не было."""
    ответ = client.post(f"/api/meetings/{двое['чужая']}/summary",
                        json={"text": "я тут был"},
                        headers={"Authorization": f"Bearer {двое['токен']}"})
    assert ответ.status_code == 404
