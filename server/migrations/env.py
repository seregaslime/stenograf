"""Как Alembic находит базу и модели.

Соединение берётся у приложения, а не собирается заново из alembic.ini: адрес
базы должен быть один на сервер и на миграции. Режим offline (генерация SQL без
подключения) не реализован — им никто не пользуется, а мёртвый код в миграциях
опаснее обычного: его никто не проверяет и не запускает.
"""
from alembic import context

from app.db import models
from app.db.database import engine

target_metadata = models.Base.metadata

with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()
