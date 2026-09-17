from contextlib import contextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..config import settings

_SERVER_DIR = Path(__file__).resolve().parent.parent.parent

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
    #
    # random_page_cost 1.1 вместо 4: четвёрка рассчитана на диск с головкой,
    # где прыжок в случайное место вчетверо дороже чтения подряд. На SSD это не
    # так, а база, веря четвёрке, «штрафует» индекс и выбирает перебор. Замер
    # scripts/bench_search.py, 50 тысяч кусков: с четвёркой база перебирала
    # (120 мс), с 1.1 — брала индекс поиска (36 мс).
    connect_args={"options": "-c timezone=UTC -c random_page_cost=1.1"},
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def alembic_config() -> Config:
    """Настройки Alembic с адресом базы из приложения.

    Отдельной функцией, потому что то же самое нужно тесту, который сверяет
    ревизии с models.py.
    """
    # Путь к ревизиям берётся из самого alembic.ini (%(here)s), поэтому
    # переопределять его здесь не нужно — он посчитан от файла, а не от текущей
    # папки, и работает, откуда бы сервер ни запустили.
    return Config(_SERVER_DIR / "alembic.ini")


def alembic_include_object(obj, name, type_, reflected, compare_to) -> bool:
    """Что Alembic сверяет с models.py.

    Индексы поиска по длине вектора создаёт сервер на лету (search.ensure_index):
    в models.py их нет и быть не может — длины заранее неизвестны. Без этого
    исключения autogenerate на живой базе предложил бы их удалить.
    """
    return not (type_ == "index" and name and name.startswith("chunks_vector_hnsw_"))


def init_db() -> None:
    """Доводит схему до последней ревизии.

    Не create_all: он создаёт только отсутствующие таблицы и НЕ добавляет
    колонки в существующие. Раньше эту дыру закрывали шесть самописных функций
    в этом же файле; теперь её закрывает Alembic, и у базы появляется отметка,
    до какой ревизии она доведена, — вместо опроса «а есть ли уже такая
    колонка» на каждом старте.

    Миграции идут при старте сервера намеренно: на машине деплоя обновление —
    это `docker compose up -d`, и отдельный шаг «не забыть прогнать миграции»
    там некому выполнить.
    """
    command.upgrade(alembic_config(), "head")


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
