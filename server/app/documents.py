"""Документы базы знаний: приём файла, текст из него, свои и чужие документы.

Пункт 5а. Человек загружает txt или md — регламент, ТЗ, заметки, — и документ
ищется тем же поиском, что и встречи. Храним извлечённый текст, а не файл:
искать нужно по тексту, оригинал у человека и так есть.

pdf и docx не принимаются намеренно (решение Сергея 17.09.2026 — позже):
текст из них достаётся отдельными библиотеками, а у pdf ещё и с потерями —
колонки, колонтитулы и таблицы собираются как получится.
"""
from datetime import datetime
from pathlib import PurePath
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .db.models import Document

ALLOWED_SUFFIXES = (".txt", ".md")
# Предел не про место в базе, а про время индексации: векторы считает модель на
# машине человека, bge-m3 на ПК — 4.4 куска в секунду (замер 02.09.2026).
# Мегабайт текста — около 1700 кусков по 600 символов, то есть ~7 минут, пока
# поиск ждёт индексацию. Больше этого человек сочтёт зависанием.
MAX_BYTES = 1_000_000


class DocumentRejected(ValueError):
    """Файл не принят. status — код ответа, message — что показать человеку."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def decode_text(raw: bytes) -> str:
    """Текст файла: UTF-16 с меткой, UTF-8, а если не читается — cp1251.

    cp1251 — не догадка на всякий случай: Блокнот и Word в русской Windows
    сохраняют txt именно в ней, а у куратора Windows. UTF-8 проверяется раньше,
    потому что байты русского UTF-8 в cp1251 тоже «прочитаются» — кракозябрами.
    UTF-16 — это «Юникод» в Блокноте; узнаётся по метке в начале файла. Первая
    версия его отбивала по нулевым байтам и промахнулась: у русских букв в
    UTF-16 нулевых байтов нет, и текст без пробелов ушёл бы в cp1251 мусором.

    Нулевой байт вне UTF-16 — признак не текста: pdf с переименованным
    расширением. Такое честнее отбить с подсказкой, чем сохранить мусор, который
    потом найдётся в поиске.
    """
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return raw.decode("utf-16").replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError:
            raise DocumentRejected(415, "Не удалось прочитать текст: сохраните файл в кодировке UTF-8.")
    if b"\x00" in raw:
        raise DocumentRejected(415, "Это не текстовый файл.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("cp1251")
        except UnicodeDecodeError:
            raise DocumentRejected(415, "Не удалось прочитать текст: сохраните файл в кодировке UTF-8.")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def title_from_filename(filename: str) -> str:
    """Название — имя файла без расширения: так человек узнает его в выдаче."""
    return PurePath(filename).stem.strip()[:300] or "Документ"


def check_filename(filename: str) -> None:
    if PurePath(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        raise DocumentRejected(415, "Пока принимаются только файлы .txt и .md.")


def create(db: Session, filename: str, raw: bytes, owner_id: Optional[int]) -> Document:
    check_filename(filename)
    if len(raw) > MAX_BYTES:
        raise DocumentRejected(413, "Файл больше мегабайта: его индексация заняла бы больше семи минут.")
    text = decode_text(raw).strip()
    if not text:
        raise DocumentRejected(400, "В файле нет текста.")
    document = Document(owner_id=owner_id, title=title_from_filename(filename), text=text)
    db.add(document)
    db.flush()
    return document


def for_owner(db: Session, document_id: int, owner_id: Optional[int]) -> Optional[Document]:
    """Документ, если он есть и принадлежит этому человеку. Чужой неотличим от
    несуществующего — как у встреч и голосов."""
    document = db.get(Document, document_id)
    if document is None or (owner_id is not None and document.owner_id != owner_id):
        return None
    return document


def list_for_owner(db: Session, owner_id: Optional[int]) -> list[dict]:
    query = select(Document.id, Document.title, Document.created_at, func.length(Document.text))
    if owner_id is not None:
        query = query.where(Document.owner_id == owner_id)
    return [to_dict(id_, title, created_at, chars)
            for id_, title, created_at, chars in db.execute(query.order_by(Document.created_at.desc()))]


def to_dict(id_: int, title: str, created_at: datetime, chars: int) -> dict:
    return {"id": id_, "title": title, "created_at": created_at.isoformat(), "chars": chars}
