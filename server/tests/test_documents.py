"""Документы базы знаний: приём файла, кодировка, список и удаление (пункт 5а).

Через TestClient, как test_api.py: проверяется то, что увидит приложение, —
коды ответов и форма данных, — а не внутренности.
"""
import base64

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app import documents


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


def загрузить(client, имя: str, байты: bytes):
    return client.post("/api/documents", json={
        "filename": имя, "content_base64": base64.b64encode(байты).decode(),
    })


# ------------------------------------------------------------ кодировка

@pytest.mark.parametrize("байты", [
    "Регламент созвонов".encode("utf-8"),
    "Регламент созвонов".encode("utf-8-sig"),  # Блокнот пишет метку в начало
    "Регламент созвонов".encode("cp1251"),     # txt из русской Windows
    "Регламент созвонов".encode("utf-16"),     # «Юникод» в Блокноте, с меткой
], ids=["utf-8", "utf-8 с меткой", "cp1251", "utf-16"])
def test_текст_читается_в_любой_из_привычных_кодировок(байты):
    assert documents.decode_text(байты) == "Регламент созвонов"


def test_utf16_без_пробелов_не_уходит_в_cp1251():
    """У русских букв в UTF-16 нет нулевых байтов: первая версия проверки на
    двоичность пропускала такой файл, и он читался как cp1251 — кракозябрами."""
    assert documents.decode_text("Регламент".encode("utf-16")) == "Регламент"


def test_переводы_строк_windows_приводятся_к_обычным():
    assert documents.decode_text("раз\r\nдва\rтри".encode()) == "раз\nдва\nтри"


def test_двоичный_файл_отбивается_а_не_сохраняется_мусором():
    """pdf с переименованным расширением: в нём нулевые байты. В cp1251 он
    «прочитался» бы, и мусор потом находился бы в поиске."""
    with pytest.raises(documents.DocumentRejected) as отказ:
        documents.decode_text(b"%PDF-1.7\n\x00\x01\x02 stream")
    assert отказ.value.status == 415


# ------------------------------------------------------------ через API

def test_загруженный_документ_виден_в_списке(client):
    ответ = загрузить(client, "Регламент созвонов.md", "# Созвоны\n\nПо вторникам.".encode("cp1251"))
    assert ответ.status_code == 200, ответ.text
    документ = ответ.json()
    assert документ["title"] == "Регламент созвонов"
    assert документ in client.get("/api/documents").json()


@pytest.mark.parametrize("имя, байты, код", [
    ("договор.pdf", b"%PDF-1.7", 415),               # pdf и docx — позже
    ("заметки.txt", b"   \n\n  ", 400),              # пустой
    ("большой.txt", b"a" * (documents.MAX_BYTES + 1), 413),
], ids=["pdf", "пустой", "больше мегабайта"])
def test_неподходящий_файл_отбивается_с_понятным_кодом(client, имя, байты, код):
    ответ = загрузить(client, имя, байты)
    assert ответ.status_code == код
    assert ответ.json()["detail"]


def test_повреждённое_содержимое_отбивается(client):
    ответ = client.post("/api/documents", json={"filename": "а.txt", "content_base64": "не base64!"})
    assert ответ.status_code == 400


def test_удалённый_документ_пропадает_из_списка(client):
    документ = загрузить(client, "черновик.txt", "текст".encode()).json()
    assert client.delete(f"/api/documents/{документ['id']}").status_code == 200
    assert документ["id"] not in [д["id"] for д in client.get("/api/documents").json()]
    assert client.delete(f"/api/documents/{документ['id']}").status_code == 404
