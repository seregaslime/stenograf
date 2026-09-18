"""Документы базы знаний: приём файла, кодировка, список и удаление (пункт 5а).

Через TestClient, как test_api.py: проверяется то, что увидит приложение, —
коды ответов и форма данных, — а не внутренности.
"""
import base64
import io

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


# ------------------------------------------------------------ docx

def сделать_docx(абзацы: list[str], таблица: list[list[str]] | None = None) -> bytes:
    """Настоящий docx, собранный тут же: двоичный файл в репозитории — это
    файл, который никто не прочитает глазами в ревью."""
    import docx

    документ = docx.Document()
    for абзац in абзацы:
        документ.add_paragraph(абзац)
    if таблица:
        т = документ.add_table(rows=len(таблица), cols=len(таблица[0]))
        for строка, значения in zip(т.rows, таблица):
            for ячейка, значение in zip(строка.cells, значения):
                ячейка.text = значение
    буфер = io.BytesIO()
    документ.save(буфер)
    return буфер.getvalue()


def test_из_docx_берутся_абзацы_и_таблицы():
    """Таблица лежит в документе отдельно от абзацев: без её разбора из
    регламента со сроками в поиск попала бы одна вода вокруг таблицы."""
    текст = documents.docx_text(сделать_docx(
        ["# Регламент", "Планёрка по вторникам."],
        [["Этап", "Срок"], ["Демо", "пятница"]],
    ))
    assert текст.split("\n\n") == [
        "# Регламент", "Планёрка по вторникам.", "Этап | Срок", "Демо | пятница",
    ]


def test_пустые_абзацы_docx_не_плодят_пустоты():
    assert documents.docx_text(сделать_docx(["", "   ", "Есть текст"])) == "Есть текст"


def test_повреждённый_docx_отбивается():
    with pytest.raises(documents.DocumentRejected) as отказ:
        documents.docx_text(b"PK\x03\x04" + "но дальше мусор".encode())
    assert отказ.value.status == 415


def test_zip_бомба_не_распаковывается(monkeypatch):
    """Архив на пару килобайт, который распаковывается в гигабайты, положил бы
    сервер ещё до того, как мы дошли бы до текста."""
    import zipfile

    буфер = io.BytesIO()
    with zipfile.ZipFile(буфер, "w", zipfile.ZIP_DEFLATED) as архив:
        архив.writestr("word/document.xml", b"\0" * 5_000_000)
    monkeypatch.setattr(documents, "MAX_UNPACKED_BYTES", 1_000_000)

    with pytest.raises(documents.DocumentRejected) as отказ:
        documents.docx_text(буфер.getvalue())
    assert отказ.value.status == 413
    assert len(буфер.getvalue()) < 100_000  # сам архив крошечный — на размер файла не поймать


def test_docx_загружается_через_api(client):
    ответ = загрузить(client, "Регламент отдела.docx", сделать_docx(["Планёрка по вторникам в 11."]))
    assert ответ.status_code == 200, ответ.text
    assert ответ.json()["title"] == "Регламент отдела"
    assert ответ.json()["chars"] == len("Планёрка по вторникам в 11.")


def test_docx_без_текста_отбивается(client):
    assert загрузить(client, "пустой.docx", сделать_docx(["", "  "])).status_code == 400


def test_слишком_много_текста_в_файле_отбивается(client, monkeypatch):
    """Предел на текст — про время индексации: миллион символов это ~7 минут."""
    monkeypatch.setattr(documents, "MAX_BYTES", 100)
    ответ = загрузить(client, "длинный.docx", сделать_docx(["а" * 200]))
    assert ответ.status_code == 413
    assert "индексация" in ответ.json()["detail"]
