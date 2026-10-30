"""Auth hardening: users.is_active + users.token_version, projects.subject.

Revision ID: 002_auth_hardening
Revises: 001_initial
Create Date: 2026-09-17

- users gains is_active boolean NOT NULL DEFAULT true and token_version
  integer NOT NULL DEFAULT 0 (JWT revocation via the tv claim).
- Email uniqueness enforced case-insensitively via a lower(email) unique
  index (portable alternative to citext that needs no extension).
- projects gains subject varchar(255) NULL (frontend creates projects with
  title/description/subject; title maps to the existing name column).
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "002_auth_hardening"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.add_column(
        "users",
        sa.Column("token_version", sa.Integer(), nullable=False, server_default="0"),
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_users_email_lower "
        "ON users (lower(email))"
    )
    op.add_column(
        "projects",
        sa.Column("subject", sa.String(length=255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("projects", "subject")
    op.execute("DROP INDEX IF EXISTS ix_users_email_lower")
    op.drop_column("users", "token_version")
    op.drop_column("users", "is_active")
