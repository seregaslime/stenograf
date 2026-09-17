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


def test_пустой_вектор_отбивается_до_базы(client, встреча):
    """У vector размерность не меньше единицы: пустой вектор база не примет, и
    вместо внятного отказа человек получил бы ошибку сервера."""
    порции = куски(client, встреча)
    assert client.post("/api/search/index", json={
        "model": модель(встреча), "meeting_id": встреча,
        "chunks": [{**к, "vector": []} for к in порции],
    }).status_code == 422
    assert client.post("/api/search/query", json={
        "model": модель(встреча), "vector": [],
    }).status_code == 422


# --- документы базы знаний ищутся вместе со встречами ---

@pytest.fixture()
def документ():
    from app.db.models import Document

    with session_scope() as db:
        д = Document(title="Регламент созвонов", text="# Созвоны\n\nПо вторникам в 11, демо по пятницам.")
        db.add(д)
        db.flush()
        return д.id


def test_документ_индексируется_и_находится_рядом_со_встречей(client, встреча, документ):
    """Главное обещание пункта 5а: один поиск по встречам и документам сразу."""
    модель_ = модель(встреча)
    ждут = client.get(f"/api/search/pending?model={модель_}").json()
    [д] = [д for д in ждут["documents"] if д["document_id"] == документ]
    assert д["title"] == "Регламент созвонов"
    assert д["chunks"] and set(д["chunks"][0]) == {"text"}  # у документа нет реплик и времени

    assert client.post("/api/search/index", json={
        "model": модель_, "document_id": документ,
        "chunks": [{**к, "vector": БЛИЗКИЙ} for к in д["chunks"]],
    }).status_code == 200
    client.post("/api/search/index", json={
        "model": модель_, "meeting_id": встреча,
        "chunks": [{**к, "vector": ДАЛЁКИЙ} for к in куски(client, встреча)],
    })

    найдено = client.post("/api/search/query", json={"model": модель_, "vector": БЛИЗКИЙ}).json()["results"]
    assert найдено[0]["document_id"] == документ
    assert найдено[0]["document_title"] == "Регламент созвонов"
    assert найдено[0]["meeting_id"] is None and найдено[0]["start_s"] is None
    assert any(к["meeting_id"] == встреча and к["document_id"] is None for к in найдено)

    ждут = client.get(f"/api/search/pending?model={модель_}").json()
    assert документ not in [д["document_id"] for д in ждут["documents"]]


def test_индексация_требует_ровно_один_источник(client, встреча, документ):
    кусок = {"first_segment_id": 1, "last_segment_id": 1, "start_s": 0.0, "text": "т", "vector": БЛИЗКИЙ}
    for тело in ({}, {"meeting_id": встреча, "document_id": документ}):
        assert client.post("/api/search/index", json={
            "model": "bge-m3", "chunks": [кусок], **тело}).status_code == 400


def test_кусок_встречи_без_реплик_не_принимается(client, встреча):
    """Кусок встречи без реплик и времени в выдаче не открыть на нужном месте."""
    assert client.post("/api/search/index", json={
        "model": модель(встреча), "meeting_id": встреча,
        "chunks": [{"text": "без реплик", "vector": БЛИЗКИЙ}],
    }).status_code == 400


# --- состояние базы знаний для экрана (пункт 5б) ---

def статус(client, модель_: str) -> dict:
    return client.get(f"/api/knowledge/status?model={модель_}").json()


def найти(список: list[dict], id_: int) -> dict:
    return next(и for и in список if и["id"] == id_)


def test_статус_показывает_ожидание_и_индексацию(client, встреча, документ):
    """Главное, что экран должен сказать человеку: найдётся ли его встреча и
    документ, и сколько кусков ещё ждут векторов — по ним оценивается время."""
    модель_ = модель(встреча)
    до = статус(client, модель_)
    assert найти(до["meetings"], встреча)["status"] == "waiting"
    assert найти(до["meetings"], встреча)["chunks_waiting"] == len(куски(client, встреча))
    assert найти(до["documents"], документ)["status"] == "waiting"

    порции = куски(client, встреча)
    client.post("/api/search/index", json={
        "model": модель_, "meeting_id": встреча,
        "chunks": [{**к, "vector": БЛИЗКИЙ} for к in порции],
    })
    после = найти(статус(client, модель_)["meetings"], встреча)
    assert (после["status"], после["chunks"], после["chunks_waiting"]) == ("indexed", len(порции), 0)


def test_другая_модель_видит_чужие_векторы_и_снова_ждёт(client, встреча):
    """Сменили модель в настройках — поиск старые векторы не возьмёт. Экран
    должен это показать, а не рапортовать «проиндексировано»."""
    порции = куски(client, встреча)
    client.post("/api/search/index", json={
        "model": модель(встреча), "meeting_id": встреча,
        "chunks": [{**к, "vector": БЛИЗКИЙ} for к in порции],
    })
    другая = найти(статус(client, "другая-модель")["meetings"], встреча)
    assert (другая["status"], другая["chunks"], другая["chunks_other_models"]) == ("waiting", 0, len(порции))


def test_идущая_и_пустая_встречи_не_ждут_индексации(client):
    with session_scope() as db:
        идёт = crud.create_meeting(db, "Идёт", False).id
        пустая = crud.create_meeting(db, "Тишина", False)
        crud.end_meeting(db, пустая.id, status="done")
        пустая_id = пустая.id
    состояние = статус(client, "bge-m3")
    assert найти(состояние["meetings"], идёт)["status"] == "not_ready"
    assert найти(состояние["meetings"], пустая_id)["status"] == "empty"
