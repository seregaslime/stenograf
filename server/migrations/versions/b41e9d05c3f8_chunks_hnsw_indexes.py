"""индексы HNSW для векторов, которые уже лежат в базе

Revision ID: b41e9d05c3f8
Revises: 8e3f41c2d7a6
Create Date: 2026-09-17 16:00:00.000000

Индекс строится на каждую длину вектора отдельно (app/search.py). Для новых
длин его создаёт сервер, когда приходят векторы, — а для уже лежащих его
создать некому, кроме этой ревизии.

Строится обычным CREATE INDEX, а не CONCURRENTLY: CONCURRENTLY нельзя внутри
транзакции миграции. На время сборки запись в chunks ждёт — на сервере деплоя
это сотня кусков и доли секунды; на 50 тысячах сборка занимает около 30 секунд.

SQL здесь повторяет search.ensure_index, а не зовёт его: ревизия должна
делать ровно то, что делала в день написания, даже если код сервера потом
поменяется.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b41e9d05c3f8"
down_revision: Union[str, Sequence[str], None] = "8e3f41c2d7a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX_MAX_DIMS = 2000


def upgrade() -> None:
    conn = op.get_bind()
    for (dims,) in conn.execute(sa.text("SELECT DISTINCT vector_dims(vector) FROM chunks")):
        if 0 < dims <= INDEX_MAX_DIMS:
            op.execute(
                f"CREATE INDEX IF NOT EXISTS chunks_vector_hnsw_{int(dims)} ON chunks "
                f"USING hnsw ((vector::vector({int(dims)})) vector_ip_ops) "
                f"WHERE vector_dims(vector) = {int(dims)}"
            )


def downgrade() -> None:
    conn = op.get_bind()
    for (name,) in conn.execute(sa.text(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'chunks' "
            "AND indexname LIKE 'chunks\\_vector\\_hnsw\\_%'")):
        op.execute(f'DROP INDEX IF EXISTS "{name}"')
