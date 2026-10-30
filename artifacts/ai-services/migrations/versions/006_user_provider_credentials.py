"""006: user provider credentials + project LLM override.

Additive only (no existing table touched beyond a nullable column):

- user_provider_credentials: per-user encrypted LLM API keys
  (unique per user+provider).
- projects.llm_override: nullable JSONB {"provider", "model"} selecting
  a per-project provider/model. The key still resolves from the user's
  own credential (or the app default), so no secret is stored here.

Offline SQLite degrades honestly (logs a warning, changes nothing);
live Postgres applies the DDL. The upgrade is additive and rerunnable: live
recovery may have applied this schema while the database was still stamped at
004, so an existing credential table is preserved rather than recreated.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "006_user_provider_credentials"
down_revision = "005_contract_drop_legacy_embedding"
branch_labels = None
depends_on = None


def _is_postgres() -> bool:
    try:
        return op.get_bind().dialect.name == "postgresql"
    except Exception:
        return False


def upgrade() -> None:
    if not _is_postgres():
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "006 skipped: live Postgres required (offline SQLite changes nothing)."
        )
        return

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("user_provider_credentials"):
        op.create_table(
            "user_provider_credentials",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("provider", sa.String(32), nullable=False),
            sa.Column("key_encrypted", sa.Text(), nullable=False),
            sa.Column("key_hint", sa.String(8), nullable=False, server_default=""),
            sa.Column("model", sa.String(128), nullable=False, server_default=""),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("user_id", "provider", name="uq_user_provider_credentials_user_provider"),
        )
    else:
        # A live database may have received this additive DDL directly while
        # still stamped at 004. Re-running 006 must not replace that table or
        # its credential rows.
        import logging as _logging

        _logging.getLogger(__name__).info(
            "006 additive schema already present; preserving existing credential rows."
        )

    inspector = sa.inspect(bind)
    if not any(
        column["name"] == "llm_override"
        for column in inspector.get_columns("projects")
    ):
        op.add_column("projects", sa.Column("llm_override", JSONB(), nullable=True))

    # CREATE INDEX IF NOT EXISTS is safe when 006 was applied manually before
    # the Alembic version stamp was repaired.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_user_provider_credentials_user_id "
        "ON user_provider_credentials (user_id)"
    )


def downgrade() -> None:
    if not _is_postgres():
        return
    op.drop_column("projects", "llm_override")
    op.drop_index("ix_user_provider_credentials_user_id", table_name="user_provider_credentials")
    op.drop_table("user_provider_credentials")
