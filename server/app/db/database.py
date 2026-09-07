from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..config import settings

settings.data_dir.mkdir(parents=True, exist_ok=True)

# pool_pre_ping: база живёт отдельным контейнером и переживает перезапуски
# независимо от сервера. Без проверки первое же соединение из пула после
# перезапуска базы отдаёт «server closed the connection unexpectedly» — и падает
# не старт, который заметили бы, а случайный запрос посреди встречи.
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    # Пояс сессии прибит к UTC. PostgreSQL иначе отдаёт время в поясе своей
    # сессии, а он у всех разный: на макбуке Europe/Moscow, в контейнере
    # postgres — UTC. Одна и та же встреча выгружалась бы с разным временем в
    # зависимости от того, откуда её скачали, и тест на это молча проходил бы
    # там, где пояс совпал. Наружу время переводится явно (см. МОСКВА в main).
    connect_args={"options": "-c timezone=UTC"},
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    """Создаёт недостающие таблицы.

    Самописные миграции отсюда убраны: они были написаны на PRAGMA и
    перестройке таблиц — командах, которых в PostgreSQL нет вовсе. Схему на
    существующих базах будет доводить Alembic.
    """
    from . import models  # noqa: F401 — регистрирует таблицы

    models.Base.metadata.create_all(engine)


@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
