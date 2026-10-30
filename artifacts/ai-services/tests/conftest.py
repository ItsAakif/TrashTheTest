"""Shared fakes: dummy DB session, in-memory users/projects, app harness."""

import uuid
from datetime import datetime

import pytest
import pytest_asyncio

from database.models.user import User


class DummySession:
    """Stands in for AsyncSession: nothing hits a real database."""

    def add(self, obj):
        return None

    async def commit(self):
        return None

    async def refresh(self, obj):
        return obj

    async def flush(self):
        return None

    async def close(self):
        return None

    async def delete(self, obj):
        return None

    async def rollback(self):
        return None

    async def execute(self, *args, **kwargs):
        """Default empty result (lets read-only routes render empty states)."""

        class _Scalars:
            def all(self):
                return []

            def first(self):
                return None

        class _Result:
            def scalars(self):
                return _Scalars()

            def scalar(self):
                return None

        return _Result()


def make_user(email="a@example.com", password_hash="hashed", name="User A"):
    return User(
        id=uuid.uuid4(),
        email=email,
        password_hash=password_hash,
        name=name,
        is_active=True,
        token_version=0,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@pytest.fixture
def dummy_db():
    return DummySession()


@pytest.fixture
def user_a():
    return make_user("a@example.com")


@pytest.fixture
def user_b():
    return make_user("b@example.com", name="User B")


@pytest.fixture
def app_client():
    """httpx client bound to the single FastAPI app (ASGI, no network)."""
    import httpx
    from main import app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture
def auth_headers(user_a):
    from services.auth.service import issue_token_pair

    pair = issue_token_pair(user_a.id, user_a.token_version)
    return {"Authorization": f"Bearer {pair['access_token']}"}
