"""Выдача токенов доступа — командой на сервере, без экрана регистрации.

Почему так, а не форма с паролем: пользователей заводит тот, у кого есть доступ
к машине, и это ровно один человек. Форма регистрации на открытом в интернет
порту означала бы, что аккаунт себе заводит и сканер тоже.

    python -m app.users add "Сергей"     # завести человека, показать токен
    python -m app.users list             # кто заведён
    python -m app.users remove 2         # убрать доступ

В контейнере:
    docker compose exec server python -m app.users add "Сергей"
"""
import sys

from sqlalchemy import select

from . import auth
from .db.database import init_db, session_scope
from .db.models import User


def _добавить(имя: str) -> None:
    with session_scope() as db:
        user, token = auth.create_user(db, имя)
        print(f"Заведён #{user.id} «{user.name}»")
        print(f"Токен: {token}")
        print("Показывается один раз — в базе только хеш. Потеряется — выдайте новый.")


def _список() -> None:
    with session_scope() as db:
        строки = list(db.scalars(select(User).order_by(User.id)))
        if not строки:
            print("Никого не заведено — сервер работает без токена (личный режим).")
            return
        for user in строки:
            print(f"#{user.id}\t{user.name}\t{user.created_at:%d.%m.%Y}")


def _убрать(id_: int) -> None:
    with session_scope() as db:
        user = db.get(User, id_)
        if user is None:
            print(f"Человека #{id_} нет")
            return
        db.delete(user)
        print(f"Доступ #{id_} «{user.name}» отозван")
        # Предупреждаем намеренно: сервер без людей снова пускает без токена,
        # и удаление последнего — это не «закрыли всем», а «открыли всем».
        if db.scalar(select(User.id).limit(1)) is None:
            print("Это был последний: сервер снова открыт без токена.")


def main(аргументы: list[str]) -> int:
    init_db()
    if not аргументы:
        print(__doc__)
        return 1
    команда, *остальное = аргументы
    if команда == "add" and остальное:
        _добавить(" ".join(остальное))
    elif команда == "list":
        _список()
    elif команда == "remove" and остальное:
        if not остальное[0].isdigit():
            print(f"Нужен номер человека, а не «{остальное[0]}» — посмотрите: list")
            return 1
        _убрать(int(остальное[0]))
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
