"""007: add Phase-4/Academic output types to the outputtype enum (T-phaseB).

The Python OutputType enum gained mcq/answer/syllabus/pyq/mapping/analysis
but the live Postgres enum (created by 001 with 6 labels) rejects them:
`invalid input value for enum outputtype: "mcq"` 500'd generation live
(Phase B find). Additive only: six ADD VALUE statements, no table touch,
safe to run repeatedly (DO block guards).

Live note: databases stamped at 004 (005 refuses by design while the
legacy BYTEA column is in use) should apply these statements directly;
traversing 006 -> 007 later is a no-op thanks to the guard.
"""

from alembic import op

revision = "007_outputtype_phase4_values"
down_revision = "006_user_provider_credentials"
branch_labels = None
depends_on = None

_NEW_LABELS = ("mcq", "answer", "syllabus", "pyq", "mapping", "analysis")


def upgrade() -> None:
    for label in _NEW_LABELS:
        op.execute(
            "DO $$ BEGIN "
            f"IF NOT EXISTS (SELECT 1 FROM pg_enum e JOIN pg_type t "
            f"ON e.enumtypid = t.oid WHERE t.typname = 'outputtype' "
            f"AND e.enumlabel = '{label}') THEN "
            f"ALTER TYPE outputtype ADD VALUE '{label}'; END IF; "
            "END $$;"
        )


def downgrade() -> None:
    # Postgres cannot drop enum values; downgrade is a documented no-op.
    pass
