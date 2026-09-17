"""векторы кусков поиска хранятся в строке, а не в TOAST

Revision ID: 8e3f41c2d7a6
Revises: 5d1c2e7a9b40
Create Date: 2026-09-17 15:00:00.000000

Вектор bge-m3 весит 4 КБ, и Postgres по умолчанию выносит такие значения в
отдельное хранилище (TOAST), оставляя в строке ссылку. Отсюда две беды,
найденные замером scripts/bench_search.py на 50 тысячах кусков:
  - перебор при поиске достаёт каждый вектор отдельно — 188 мс;
  - база оценивает перебор по «тонким» строкам со ссылками, считает его
    дешёвым и не берёт индекс, хотя с индексом запрос в разы быстрее.
С векторами в строке перебор на тех же данных — 59 мс (на 10 тысячах — 10 мс
вместо 30), вместе с оценкой диска для SSD база берёт индекс.

MAIN, а не PLAIN: PLAIN запрещает вынос совсем, и вектор длиннее страницы (у
qwen3-embedding:8b это 4096 чисел, 16 КБ) не записался бы вовсе. MAIN держит
значение в строке, пока оно помещается, и выносит только то, что не влезает.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "8e3f41c2d7a6"
down_revision: Union[str, Sequence[str], None] = "5d1c2e7a9b40"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE chunks ALTER COLUMN vector SET STORAGE MAIN")
    # Режим хранения действует только на новые записи. Уже лежащие векторы
    # переписываются, а приведение через текст нужно, чтобы запись была новым
    # значением: одинаковое значение база оставила бы на старом месте в TOAST.
    op.execute("UPDATE chunks SET vector = vector::text::vector")


def downgrade() -> None:
    op.execute("ALTER TABLE chunks ALTER COLUMN vector SET STORAGE EXTENDED")
