"""Токены доступа: кого сервер пускает и что отдаёт чужому.

Тесты чистят за собой людей в finally не из аккуратности, а по необходимости:
база у функциональных тестов одна на весь прогон, и оставленный пользователь
включил бы обязательный токен для всех остальных тестов сразу.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import app.main as main
from app import auth
from app.db.database import session_scope
from app.db.models import User


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
