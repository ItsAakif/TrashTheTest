"""
Database Engine and Session Management

Provides async PostgreSQL connection with pgvector support.
Uses asyncpg driver for high-performance async operations.
"""

import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

# Database URL - supports all connection string formats
# Examples:
# postgresql+asyncpg://user:pass@localhost:5432/dbname
# postgresql+asyncpg://localhost:5432/dbname
DATABASE_URL = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    # Resolve through settings so the error names the variable.
    try:
        from config.settings import get_settings as _get_settings

        DATABASE_URL = _get_settings().database_url or ""
    except Exception:
        DATABASE_URL = ""

if not DATABASE_URL:
    # Placeholder so module import never crashes on engine construction;
    # the real fail-fast lives in config.settings (JWT_SECRET/DATABASE_URL
    # required at import) and in assert_database_configured() at boot.
    DATABASE_URL = "postgresql+asyncpg://localhost/trashthetest"


def assert_database_configured() -> str:
    """Raise with a clear message when DATABASE_URL was never provided."""
    url = os.getenv("DATABASE_URL", "")
    if not url:
        try:
            from config.settings import get_settings as _get_settings

            url = _get_settings().database_url or ""
        except Exception:
            url = ""
    if not url or "localhost/trashthetest" in url and not os.getenv("DATABASE_URL"):
        raise RuntimeError(
            "Missing required configuration: DATABASE_URL. "
            "Set DATABASE_URL (see .env.example)."
        )
    return url

# Echo SQL queries for debugging (set to False in production)
ECHO_SQL = os.getenv("ECHO_SQL", "false").lower() == "true"

# Create async engine
# Using NullPool for serverless/container environments
# For persistent servers, use regular pool with proper settings
engine = create_async_engine(
    DATABASE_URL,
    echo=ECHO_SQL,
    poolclass=NullPool,  # For container/serverless environments
    # For persistent servers, use:
    # pool_size=10,
    # max_overflow=20,
    # pool_pre_ping=True,
    future=True,
)

# Create async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for SQLAlchemy models
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session.
    
    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Context manager for explicit session handling
class DatabaseSession:
    """Context manager for database sessions"""
    
    def __init__(self):
        self.session: AsyncSession | None = None
    
    async def __aenter__(self) -> AsyncSession:
        self.session = async_session_maker()
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type is not None:
                await self.session.rollback()
            else:
                await self.session.commit()
            await self.session.close()


async def init_db():
    """
    Initialize database - create all tables.
    Use Alembic for production migrations.
    """
    async with engine.begin() as conn:
        # Create all tables (use Alembic for production)
        await conn.run_sync(Base.metadata.create_all)


async def drop_db():
    """Drop all tables - for development only"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)