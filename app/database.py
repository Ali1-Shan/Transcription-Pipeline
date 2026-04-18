"""Database engine and session management.

Provides async SQLAlchemy engine and session factory.
Swap the DATABASE_URL to postgresql+asyncpg:// for production PostgreSQL.
"""

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

# Ensure parent directory exists for SQLite
if settings.database_url.startswith("sqlite"):
    db_path = settings.database_url.split("///")[-1]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_async_engine(
    settings.database_url,
    echo=False,
    # SQLite doesn't support pool_size; only set for non-SQLite
    **({} if "sqlite" in settings.database_url else {"pool_size": 5, "max_overflow": 10}),
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


async def init_db() -> None:
    """Create all tables. Called once at application startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose the engine connection pool. Called at shutdown."""
    await engine.dispose()


async def get_db() -> AsyncSession:  # type: ignore[misc]
    """FastAPI dependency that yields a database session.

    Usage:
        @router.get("/")
        async def endpoint(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_factory() as session:
        yield session
