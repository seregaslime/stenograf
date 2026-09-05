"""Сервер хранит присланные векторы и ищет по ним, но сам их не считает.

Эмбеддинги уехали в приложение: у каждого своя модель и свой адрес. Сравнение
векторов модели не требует — это скалярное произведение, поэтому поиск остался
на сервере: по сети едет вопрос в килобайтах, а не вся матрица в мегабайтах.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import app.main as main
from app import auth
from app.db import crud
from app.db.database import session_scope
from app.db.models import Chunk, Meeting, User


# Имя модели у каждого теста своё: база одна на весь прогон, и векторы соседних
# встреч иначе попадали бы в выдачу — падало бы не то, что проверяем.
def модель(встреча_id: int) -> str:
    return f"bge-m3-{встреча_id}"


# Короткие векторы: важна арифметика близости, а не размерность настоящей модели.
БЛИЗКИЙ = [1.0, 0.0, 0.0]
ДАЛЁКИЙ = [0.0, 1.0, 0.0]


@pytest.fixture()
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture()
def встреча():
    with session_scope() as db:
        m = crud.create_meeting(db, "Планёрка", False)
        s = crud.get_or_create_self_speaker(db)
        # Реплики намеренно длинные: куски набираются до 600 символов, и на
        # коротких вся встреча уместилась бы в один — сравнивать было бы нечего.
        for i in range(6):
            crud.add_segment(db, m.id, s.id, "mic", i * 10.0, i * 10 + 5.0,
                             f"реплика {i} про сроки и релиз. " + "подробности разговора " * 15)
        crud.end_meeting(db, m.id, status="done")
        return m.id


def куски(client, встреча_id: int) -> list[dict]:
    ждут = client.get(f"/api/search/pending?model={модель(встреча_id)}").json()["meetings"]
    return next(m["chunks"] for m in ждут if m["meeting_id"] == встреча_id)


def test_сервер_отдаёт_куски_а_не_векторы(client, встреча):
    """Нарезка осталась на сервере: она про содержимое встречи, а не про модель.

    Имя модели при этом обязательно приходит от приложения: сервер больше не
    знает, чем считают векторы, и «кого индексировать» зависит именно от неё.
    """
    порции = куски(client, встреча)
    assert порции, "встреча с речью должна ждать индексации"
    assert "text" in порции[0] and "start_s" in порции[0]
    assert "vector" not in порции[0]


def test_принятые_векторы_находятся_поиском(client, встреча):
    порции = куски(client, встреча)
    ответ = client.post("/api/search/index", json={
        "model": модель(встреча),
        "meeting_id": встреча,
        "chunks": [{**к, "vector": БЛИЗКИЙ} for к in порции],
    })
    assert ответ.status_code == 200
    assert ответ.json()["chunks"] == len(порции)

    найдено = client.post("/api/search/query", json={
        "model": модель(встреча), "vector": БЛИЗКИЙ, "limit": 5,
    }).json()["results"]
    assert найдено and найдено[0]["similarity"] == pytest.approx(1.0)
    assert найдено[0]["meeting_id"] == встреча


def test_проиндексированная_встреча_больше_не_ждёт(client, встреча):
    порции = куски(client, встреча)
    client.post("/api/search/index", json={
        "model": модель(встреча), "meeting_id": встреча,
        "chunks": [{**к, "vector": БЛИЗКИЙ} for к in порции],
    })
    адрес = f"/api/search/pending?model={модель(встреча)}"
    ждут = [m["meeting_id"] for m in client.get(адрес).json()["meetings"]]
    assert встреча not in ждут


def test_повторная_индексация_не_плодит_куски(client, встреча):
    """Иначе одна встреча заняла бы всю выдачу одинаковыми цитатами."""
    порции = куски(client, встреча)
    тело = {"model": модель(встреча), "meeting_id": встреча,
            "chunks": [{**к, "vector": БЛИЗКИЙ} for к in порции]}
    client.post("/api/search/index", json=тело)
    client.post("/api/search/index", json=тело)

    with session_scope() as db:
        всего = len(list(db.scalars(select(Chunk).where(Chunk.meeting_id == встреча))))
    assert всего == len(порции)


def test_векторы_чужой_модели_в_выдачу_не_лезут(client, встреча):
    порции = куски(client, встреча)
    client.post("/api/search/index", json={
        "model": модель(встреча), "meeting_id": встреча,
        "chunks": [{**к, "vector": БЛИЗКИЙ} for к in порции],
    })
    найдено = client.post("/api/search/query", json={
        "model": "другая-модель", "vector": БЛИЗКИЙ,
    }).json()["results"]
    assert найдено == []


def test_дальний_вектор_ранжируется_ниже(client, встреча):
    порции = куски(client, встреча)
    смешанные = [{**к, "vector": БЛИЗКИЙ if i == 0 else ДАЛЁКИЙ}
                 for i, к in enumerate(порции)]
    client.post("/api/search/index", json={
        "model": модель(встреча), "meeting_id": встреча, "chunks": смешанные,
    })
    найдено = client.post("/api/search/query", json={
        "model": модель(встреча), "vector": БЛИЗКИЙ,
    }).json()["results"]
    assert найдено[0]["similarity"] > найдено[-1]["similarity"]


def test_чужую_встречу_проиндексировать_нельзя(client):
    """Иначе чужой архив можно было бы наполнить своими векторами и вытянуть
    его цитаты собственным поиском."""
    with session_scope() as db:
        сергей, токен = auth.create_user(db, "Сергей")
        куратор, _ = auth.create_user(db, "Куратор")
        чужая = crud.create_meeting(db, "Чужая", False, owner_id=куратор.id)
        crud.end_meeting(db, чужая.id, status="done")
        чужая_id = чужая.id
    try:
        ответ = client.post("/api/search/index", json={
            "model": "bge-m3", "meeting_id": чужая_id,
            "chunks": [{"first_segment_id": 1, "last_segment_id": 1, "start_s": 0.0,
                        "text": "подделка", "vector": БЛИЗКИЙ}],
        }, headers={"Authorization": f"Bearer {токен}"})
        assert ответ.status_code == 404
    finally:
        with session_scope() as db:
            for таблица in (Meeting, User):
                for строка in db.scalars(select(таблица)):
                    db.delete(строка)
