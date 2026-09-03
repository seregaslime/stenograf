"""Сводная проверка изоляции: что закрыто токеном и что не видно чужому.

Отдельным файлом от test_auth.py намеренно. Там проверяется механика токена
(разбор заголовка, кадр auth, хранение хешем), здесь — обещание, которое сервер
даёт человеку: «у каждого своё приложение, пересечений нет». Обещание проверять
надо целиком и по всем эндпоинтам сразу, иначе новый эндпоинт добавят, а
проверку к нему — нет.

Главный тест здесь — перебор ВСЕХ маршрутов приложения. Он ловит не то, что
написано сегодня, а то, что допишут завтра: забытый токен у нового эндпоинта
проваливает прогон, и это единственная защита, которая не зависит от того,
вспомнил ли автор про доступ.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from starlette.routing import Route

import app.main as main
from app import auth
from app.db import crud
from app.db.database import session_scope
from app.db.models import Meeting, Speaker, User

# Здоровье открыто намеренно: по нему видно, жив ли сервер, до ввода токена.
ОТКРЫТО_БЕЗ_ТОКЕНА = {"/api/health"}

# Пример значения для параметра пути: нам важен код ответа, а не сама сущность.
ЗАГЛУШКИ = {"meeting_id": "1", "speaker_id": "1", "print_id": "1"}


def подставить(путь: str) -> str:
    for имя, значение in ЗАГЛУШКИ.items():
        путь = путь.replace("{" + имя + "}", значение)
    return путь


def все_маршруты() -> list[tuple[str, str]]:
    """(метод, путь) для всех HTTP-маршрутов приложения."""
    маршруты = []
    for route in main.app.routes:
        if not isinstance(route, Route) or "{" in route.path and "path" in route.path:
            continue
        for метод in sorted(route.methods - {"HEAD", "OPTIONS"}):
            маршруты.append((метод, route.path))
    return маршруты


@pytest.fixture()
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture()
def двое():
    """Двое заведённых людей с данными; после теста база возвращается к пустой."""
    with session_scope() as db:
        сергей, токен_с = auth.create_user(db, "Сергей")
        куратор, токен_к = auth.create_user(db, "Куратор")
        данные = {
            "сергей": токен_с,
            "куратор": токен_к,
            "встреча_куратора": crud.create_meeting(db, "Чужая", False,
                                                    owner_id=куратор.id).id,
            "спикер_куратора": crud.create_speaker(db, owner_id=куратор.id).id,
        }
    yield данные
    with session_scope() as db:
        for таблица in (Meeting, Speaker, User):
            for строка in db.scalars(select(таблица)):
                db.delete(строка)


@pytest.mark.parametrize("метод, путь", все_маршруты())
def test_каждый_маршрут_закрыт_без_токена(client, двое, метод, путь):
    """Ни один маршрут, кроме здоровья, не отвечает без токена.

    Перебор по маршрутам приложения, а не по списку в тесте: список пришлось бы
    дописывать руками, и первый же забытый эндпоинт остался бы открытым молча.
    """
    ответ = client.request(метод, подставить(путь), json={})
    if путь in ОТКРЫТО_БЕЗ_ТОКЕНА:
        assert ответ.status_code == 200, f"{метод} {путь} должен быть открыт"
        assert "asr" not in ответ.json(), "чужому здоровье отдаёт только «жив»"
    else:
        assert ответ.status_code == 401, f"{метод} {путь} отвечает без токена!"


def test_схема_api_тоже_закрыта(client, двое):
    """Схема — это карта сервера. Отдавать её тому, от кого сервер закрывали,
    незачем: она перечисляет все эндпоинты и форму их тел."""
    for путь in ("/openapi.json", "/docs", "/redoc"):
        assert client.get(путь).status_code == 401, путь


def test_личный_сервер_открыт_как_раньше(client):
    """Пока людей не заводили, сервер работает без токена — иначе локальный
    запуск потребовал бы настройки на пустом месте."""
    assert client.get("/api/meetings").status_code == 200
    assert client.get("/openapi.json").status_code == 200


@pytest.mark.parametrize("метод, путь", [
    ("GET", "/api/meetings/{meeting_id}"),
    ("DELETE", "/api/meetings/{meeting_id}"),
    ("POST", "/api/meetings/{meeting_id}/summarize"),
    ("GET", "/api/meetings/{meeting_id}/export"),
])
def test_чужая_встреча_не_найдена(client, двое, метод, путь):
    """404, а не 403: разница между ними — это ответ на вопрос «а есть ли у
    соседа встреча с таким номером», то есть утечка без единого байта текста."""
    адрес = путь.replace("{meeting_id}", str(двое["встреча_куратора"]))
    ответ = client.request(метод, адрес,
                           headers={"Authorization": f"Bearer {двое['сергей']}"})
    assert ответ.status_code == 404


def test_чужой_голос_не_найден(client, двое):
    чужой = двое["спикер_куратора"]
    заголовки = {"Authorization": f"Bearer {двое['сергей']}"}
    assert client.patch(f"/api/speakers/{чужой}", json={"name": "Мой"},
                        headers=заголовки).status_code == 404
    assert client.delete(f"/api/speakers/{чужой}", headers=заголовки).status_code == 404
    assert client.get(f"/api/speakers/{чужой}/voiceprints/1/audio",
                      headers=заголовки).status_code == 404
    assert client.delete(f"/api/speakers/{чужой}/voiceprints/1",
                         headers=заголовки).status_code == 404


def test_поиск_не_видит_чужих_встреч(client, двое, monkeypatch):
    """Поиск идёт по векторам, но фильтр владельца стоит до похода к модели:
    чужие куски не попадают ни в выдачу, ни в индексацию."""
    async def без_модели(db, cfg, texts=None, *args, **kwargs):
        raise AssertionError("до модели дойти не должно: своих встреч нет")

    from app import search as search_mod
    monkeypatch.setattr(search_mod, "index_meeting", без_модели)
    ответ = client.get("/api/search?q=сроки",
                       headers={"Authorization": f"Bearer {двое['сергей']}"})
    assert ответ.status_code == 200
    assert ответ.json()["results"] == []
