"""
Alembic migrations environment configuration.
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Import models and base
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import all models so they get registered
from database.engine import Base
from database.models import (
    User, Project, Document, Chunk,
    FlashcardDeck, Flashcard, Note, Quiz,
    StudySession, GeneratedContentCache,
    Job, Website
)

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Prefer the runtime DATABASE_URL (compose sets it to the `db` host).
# alembic.ini ships a localhost default for local runs; without this
# override `alembic upgrade head` inside the api container tries
# 127.0.0.1:5432 and never migrates the live database.
_db_url_override = os.getenv("DATABASE_URL")
if _db_url_override:
    config.set_main_option("sqlalchemy.url", _db_url_override)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def _ensure_wide_alembic_version(connection) -> None:
    """Keep Alembic's bookkeeping column wide enough for legacy revisions.

    Alembic creates ``alembic_version.version_num`` as ``VARCHAR(32)`` by
    default. Revision 005 predates this repair and is 34 characters, so its
    version stamp fails before a later migration can widen the column. Run
    this preflight before Alembic opens its migration transaction. The CREATE
    is also safe for a brand-new database: Alembic reuses the wide table.
    """
    if connection.dialect.name != "postgresql":
        return

    await connection.execute(
        text(
            "CREATE TABLE IF NOT EXISTS alembic_version ("
            "version_num VARCHAR(255) NOT NULL, "
            "CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)"
            ")"
        )
    )
    await connection.execute(
        text(
            "ALTER TABLE alembic_version "
            "ALTER COLUMN version_num TYPE VARCHAR(255)"
        )
    )
    await connection.commit()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """

    # Use the asyncpg driver for PostgreSQL
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        # For async SQLAlchemy 2.0 with asyncpg
        connect_args={"statement_cache_size": 0},
    )

    async with connectable.connect() as connection:
        await connection.execute(
            # Enable pgvector extension
            text("CREATE EXTENSION IF NOT EXISTS vector")
        )
        
        await connection.commit()

    async with connectable.connect() as connection:
        await _ensure_wide_alembic_version(connection)
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()