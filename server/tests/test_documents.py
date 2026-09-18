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
    ("презентация.pptx", b"PK\x03\x04", 415),        # формат, которого мы не умеем
    ("заметки.txt", b"   \n\n  ", 400),              # пустой
    ("большой.txt", b"a" * (documents.MAX_BYTES + 1), 413),
], ids=["чужой формат", "пустой", "больше мегабайта"])
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


# ------------------------------------------------------------ pdf

def сделать_pdf(страницы: list[list[str]]) -> bytes:
    """Настоящий pdf, собранный тут же из минимума объектов: страницы, шрифт и
    таблица /ToUnicode — та самая, по которой читалка узнаёт, какая буква стоит
    за кодом. Без неё кириллица извлеклась бы вопросительными знаками, и тест
    проверял бы не то, что нужно.

    Библиотеки, умеющей рисовать pdf, в проекте нет, а заводить её ради тестов
    дороже двадцати строк здесь.
    """
    код = {c: i + 1 for i, c in enumerate(sorted({c for стр in страницы for s in стр for c in s}))}
    тела: list[bytes] = []

    def добавить(тело: bytes) -> int:
        тела.append(тело)
        return len(тела)

    cmap = ("/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
            "1 begincodespacerange <00> <FF> endcodespacerange\n"
            f"{len(код)} beginbfchar\n"
            + "".join(f"<{к:02X}> <{ord(c):04X}>\n" for c, к in код.items())
            + "endbfchar endcmap CMapName currentdict /CMap defineresource pop end end").encode()
    юникод = добавить(b"<< /Length %d >>\nstream\n" % len(cmap) + cmap + b"\nendstream")
    шрифт = добавить(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica"
                     b" /ToUnicode %d 0 R >>" % юникод)

    номера = []
    for строки in страницы:
        рисование = []
        for строка in строки:
            байты = bytes(код[c] for c in строка)
            экран = байты.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
            рисование.append(b"(" + экран + b") Tj T*")
        поток = b"BT /F1 12 Tf 72 720 Td 14 TL\n" + b"\n".join(рисование) + b"\nET"
        содержимое = добавить(b"<< /Length %d >>\nstream\n" % len(поток) + поток + b"\nendstream")
        номера.append(добавить(
            b"<< /Type /Page /Parent PARENT /MediaBox [0 0 612 792]"
            b" /Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>" % (шрифт, содержимое)))
    дерево = добавить(b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % н for н in номера)
                      + b"] /Count %d >>" % len(номера))
    каталог = добавить(b"<< /Type /Catalog /Pages %d 0 R >>" % дерево)
    тела = [т.replace(b"PARENT", b"%d 0 R" % дерево) for т in тела]

    файл = io.BytesIO()
    файл.write(b"%PDF-1.4\n")
    смещения = []
    for номер, тело in enumerate(тела, start=1):
        смещения.append(файл.tell())
        файл.write(b"%d 0 obj\n" % номер + тело + b"\nendobj\n")
    xref = файл.tell()
    файл.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(тела) + 1))
    for смещение in смещения:
        файл.write(b"%010d 00000 n \n" % смещение)
    файл.write(b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
               % (len(тела) + 1, каталог, xref))
    return файл.getvalue()


def test_из_pdf_берётся_текст_всех_страниц():
    текст = documents.pdf_text(сделать_pdf([
        ["Регламент отдела", "Планёрка по вторникам в 11:00"],
        ["Демо заказчику по пятницам в 16:00"],
    ]))
    assert "Планёрка по вторникам в 11:00" in текст
    assert "Демо заказчику по пятницам в 16:00" in текст
    # Пустая строка между страницами: по ней document_chunks режет на куски
    assert "\n\n" in текст


def test_слово_разорванное_переносом_склеивается():
    """Иначе в вектор уйдут «доку» и «мент», и по слову «документ» не найдётся."""
    текст = documents.pdf_text(сделать_pdf([["Приложен доку-", "мент о сроках"]]))
    assert "документ о сроках" in текст


def test_точки_оглавления_не_занимают_место_в_куске():
    """Строка «Введение .......... 5» наполовину состоит из точек, а кусок для
    вектора ограничен по длине — точки вытесняют из него слова."""
    текст = documents.pdf_text(сделать_pdf([["Общая формулировка ............ 2"]]))
    assert текст == "Общая формулировка 2"


def test_скан_отбивается_с_объяснением():
    """В скане текста нет вовсе — молча сохранить его значит завести документ,
    который никогда ничего не найдёт."""
    with pytest.raises(documents.DocumentRejected) as отказ:
        documents.pdf_text(сделать_pdf([[], []]))
    assert отказ.value.status == 415
    assert "скан" in отказ.value.message


def test_pdf_с_паролем_не_выдаётся_за_повреждённый():
    """Слово «повреждён» отправило бы человека искать вторую копию файла, хотя
    открыть нужно этот же — и пересохранить без пароля."""
    from pypdf import PdfWriter

    писатель = PdfWriter(clone_from=io.BytesIO(сделать_pdf([["Секретный регламент"]])))
    писатель.encrypt("пароль")
    буфер = io.BytesIO()
    писатель.write(буфер)

    with pytest.raises(documents.DocumentRejected) as отказ:
        documents.pdf_text(буфер.getvalue())
    assert отказ.value.status == 415
    assert "паролем" in отказ.value.message


def test_незнакомое_шифрование_тоже_про_пароль(monkeypatch):
    """AES-256 pypdf разбирает только с библиотекой cryptography, а её у нас
    нет: он бросает исключение вместо ответа «не открылось». Без перехвата
    человек получил бы 500 вместо объяснения."""
    from pypdf import PdfReader

    def падает(self, пароль):
        raise Exception("cryptography не установлена")

    monkeypatch.setattr(PdfReader, "decrypt", падает)
    писатель_буфер = io.BytesIO()
    from pypdf import PdfWriter
    писатель = PdfWriter(clone_from=io.BytesIO(сделать_pdf([["Секретный регламент"]])))
    писатель.encrypt("пароль")
    писатель.write(писатель_буфер)

    with pytest.raises(documents.DocumentRejected) as отказ:
        documents.pdf_text(писатель_буфер.getvalue())
    assert отказ.value.status == 415
    assert "паролем" in отказ.value.message


def test_повреждённый_pdf_отбивается():
    with pytest.raises(documents.DocumentRejected) as отказ:
        documents.pdf_text(b"%PDF-1.7\n" + "а дальше мусор".encode())
    assert отказ.value.status == 415


def test_слишком_длинный_pdf_не_разбирается_целиком(monkeypatch):
    monkeypatch.setattr(documents, "MAX_PDF_PAGES", 2)
    with pytest.raises(documents.DocumentRejected) as отказ:
        documents.pdf_text(сделать_pdf([["раз"], ["два"], ["три"]]))
    assert отказ.value.status == 413


def test_много_текста_в_pdf_отбивается_не_дочитав(monkeypatch):
    """Предел на символы проверяется по ходу страниц — иначе документ молча
    сохранился бы обрезанным."""
    monkeypatch.setattr(documents, "MAX_BYTES", 10)
    with pytest.raises(documents.DocumentRejected) as отказ:
        documents.pdf_text(сделать_pdf([["первая страница"], ["вторая страница"]]))
    assert отказ.value.status == 413
    assert "индексация" in отказ.value.message


def test_pdf_загружается_через_api(client):
    ответ = загрузить(client, "Инструкция.pdf", сделать_pdf([["Пароли храним в менеджере."]]))
    assert ответ.status_code == 200, ответ.text
    assert ответ.json()["title"] == "Инструкция"
    assert ответ.json()["chars"] == len("Пароли храним в менеджере.")
