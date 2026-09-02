"""Кто пришёл на сервер: токены доступа.

Пока в базе нет ни одного человека, сервер работает как раньше — вообще без
токена. Это не забытая дыра, а переход: запуск на своём ноутбуке остаётся
однопользовательским и не требует настройки, а сервер, смотрящий в интернет,
закрывается ровно в тот момент, когда на нём заводят первого человека. Иначе
обновление сервера отрезало бы от него и владельца: клиент про токен ещё не
знает, а порт уже закрыт.

Токен хранится хешем: восстанавливать его некому — при утере выдаётся новый,
а украденная копия базы входа не даёт.
"""
import hashlib
import hmac
import secrets

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db.models import Meeting, User

# 32 байта случайности — 43 символа в urlsafe-виде. Подбирать нечего.
TOKEN_BYTES = 32


def create_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_from_header(header: str | None) -> str | None:
    """Достаёт токен из «Authorization: Bearer <токен>».

    Отдельной функцией, потому что то же самое понадобится WebSocket-у: там
    заголовок задать нельзя (ограничение браузерного API), токен придёт первым
    кадром — а разбор и сравнение должны остаться одни на оба пути.
    """
    if not header:
        return None
    схема, _, значение = header.partition(" ")
    if схема.lower() != "bearer":
        return None
    return значение.strip() or None


def auth_required(db: Session) -> bool:
    """Заведён ли хоть один человек. Нет — сервер личный, токен не спрашиваем."""
    return db.scalar(select(User.id).limit(1)) is not None


def user_by_token(db: Session, token: str | None) -> User | None:
    """Кому принадлежит токен. Неизвестный или пустой — None.

    Перебор с compare_digest, а не поиск по индексу: сравнение строк в SQLite
    завершается на первом несовпавшем байте, и по времени ответа токен можно
    подбирать посимвольно. Людей на сервере единицы, перебор ничего не стоит.
    """
    if not token:
        return None
    искомый = hash_token(token)
    for user in db.scalars(select(User)):
        if hmac.compare_digest(user.token_hash, искомый):
            return user
    return None


def create_user(db: Session, name: str) -> tuple[User, str]:
    """Заводит человека и возвращает его вместе с токеном.

    Токен показывается ровно один раз — в базе лежит только хеш.

    Первому заведённому достаются все ничейные встречи. Иначе человек, который
    полгода работал на личном сервере и закрыл его токеном, обнаружил бы пустой
    архив: записи на месте, но принадлежат никому и не видны никому.
    """
    первый = not auth_required(db)
    token = create_token()
    user = User(name=name.strip() or "Без имени", token_hash=hash_token(token))
    db.add(user)
    db.flush()
    if первый:
        for встреча in db.scalars(select(Meeting).where(Meeting.owner_id.is_(None))):
            встреча.owner_id = user.id
    return user, token
