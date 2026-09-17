"""векторы кусков поиска в pgvector

Revision ID: 5d1c2e7a9b40
Revises: 69052433fe21
Create Date: 2026-09-17 11:00:00.000000

Поиск по встречам переезжает в базу: вместо байтов, которые сервер поднимал в
память и перемножал в numpy, — колонка типа vector, и сравнение делает сама
база. Требование куратора от 17.09.2026.

Старые векторы переносятся, а не пересчитываются: пересчитать их можно только
моделью эмбеддингов, а она у приложения, не у сервера. Байты — это float32
подряд, в новую колонку они ложатся как есть.
"""
from typing import Sequence, Union

import numpy as np
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "5d1c2e7a9b40"
down_revision: Union[str, Sequence[str], None] = "69052433fe21"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _литерал(вектор: np.ndarray) -> str:
    return "[" + ",".join(repr(float(x)) for x in вектор) + "]"


def upgrade() -> None:
    # IF NOT EXISTS — и для прав тоже: там, где роль сервера не суперпользователь,
    # расширение включает администратор заранее, и эта строка его просто видит.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column("chunks", sa.Column("vector_pg", Vector(), nullable=True))

    conn = op.get_bind()
    битые_встречи = set()
    for id_, meeting_id, байты in conn.execute(sa.text("SELECT id, meeting_id, vector FROM chunks")):
        вектор = np.frombuffer(байты, dtype=np.float32) if байты and len(байты) % 4 == 0 else None
        if вектор is None or not np.isfinite(вектор).all():
            битые_встречи.add(meeting_id)
            continue
        conn.execute(sa.text("UPDATE chunks SET vector_pg = CAST(:v AS vector) WHERE id = :id"),
                     {"v": _литерал(вектор), "id": id_})
    # Кусок с битыми байтами перенести не во что. Удаляются ВСЕ куски такой
    # встречи, а не один: встреча без кусков снова попадает в очередь на
    # индексацию, и приложение пересчитает её целиком при следующем поиске.
    # Оставь уцелевшие — встреча считалась бы проиндексированной с дырой.
    if битые_встречи:
        conn.execute(sa.text("DELETE FROM chunks WHERE meeting_id = ANY(:ids)"),
                     {"ids": list(битые_встречи)})

    op.drop_column("chunks", "vector")
    op.alter_column("chunks", "vector_pg", new_column_name="vector", nullable=False)


def downgrade() -> None:
    op.add_column("chunks", sa.Column("vector_bytes", sa.LargeBinary(), nullable=True))
    conn = op.get_bind()
    for id_, текст in conn.execute(sa.text("SELECT id, vector::text FROM chunks")):
        байты = np.asarray(текст.strip("[]").split(","), dtype=np.float32).tobytes()
        conn.execute(sa.text("UPDATE chunks SET vector_bytes = :b WHERE id = :id"),
                     {"b": байты, "id": id_})
    op.drop_column("chunks", "vector")
    op.alter_column("chunks", "vector_bytes", new_column_name="vector", nullable=False)
