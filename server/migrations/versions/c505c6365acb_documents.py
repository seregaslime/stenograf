"""документы в базе знаний: таблица documents и источник у куска поиска

Revision ID: c505c6365acb
Revises: b41e9d05c3f8
Create Date: 2026-09-17 17:09:09.060314

Пункт 5а: свои файлы человека ищутся тем же поиском, что и встречи. Кусок
теперь из встречи или из документа — одна таблица кусков, один индекс HNSW
и один поиск на всё. Поля встречи у куска (встреча, реплики, время) стали
необязательными, а «источник ровно один» держит ограничение в базе.

Написано по автогенерации, но руками поправлено в трёх местах, которые она
не видит: ограничение-проверку Alembic не сравнивает и не создаёт, внешний
ключ без имени нельзя удалить при откате, а откат, делающий meeting_id снова
обязательным, упал бы на кусках документов — их сначала удаляем.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c505c6365acb"
down_revision: Union[str, Sequence[str], None] = "b41e9d05c3f8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("owner_id", sa.Integer(), nullable=True),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_documents_owner_id"), "documents", ["owner_id"], unique=False)

    op.add_column("chunks", sa.Column("document_id", sa.Integer(), nullable=True))
    for колонка, тип in (("meeting_id", sa.INTEGER()), ("first_segment_id", sa.INTEGER()),
                         ("last_segment_id", sa.INTEGER()),
                         ("start_s", sa.DOUBLE_PRECISION(precision=53))):
        op.alter_column("chunks", колонка, existing_type=тип, nullable=True)
    op.create_index(op.f("ix_chunks_document_id"), "chunks", ["document_id"], unique=False)
    op.create_foreign_key("chunks_document_id_fkey", "chunks", "documents",
                          ["document_id"], ["id"], ondelete="CASCADE")
    op.create_check_constraint("chunks_one_source", "chunks",
                               "(meeting_id IS NULL) <> (document_id IS NULL)")


def downgrade() -> None:
    op.execute("DELETE FROM chunks WHERE document_id IS NOT NULL")
    op.drop_constraint("chunks_one_source", "chunks", type_="check")
    op.drop_constraint("chunks_document_id_fkey", "chunks", type_="foreignkey")
    op.drop_index(op.f("ix_chunks_document_id"), table_name="chunks")
    for колонка, тип in (("start_s", sa.DOUBLE_PRECISION(precision=53)),
                         ("last_segment_id", sa.INTEGER()), ("first_segment_id", sa.INTEGER()),
                         ("meeting_id", sa.INTEGER())):
        op.alter_column("chunks", колонка, existing_type=тип, nullable=False)
    op.drop_column("chunks", "document_id")
    op.drop_index(op.f("ix_documents_owner_id"), table_name="documents")
    op.drop_table("documents")
