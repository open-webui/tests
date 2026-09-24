"""Fixtures for the httpx tests and for the ones that need a degraded instance.

The shared scratch instances, the scripted provider and the account fixtures live in
`harness/fixtures.py`, loaded for the whole suite from the root conftest. `degraded_instance`
is a further one on a fake Redis and a vector DB that refuses every connection.
"""

from __future__ import annotations

from typing import Generator

import httpx
import pytest

from harness.actors import Actor
from harness.instance import LaunchedInstance, launch
from harness.upstream import MockUpstream
from integration.fake_redis import FakeRedis


@pytest.fixture
def api_client(user: Actor) -> Generator[httpx.Client, None, None]:
    """The regular account's client on the shared instance."""
    with user.client() as client:
        yield client


@pytest.fixture(scope="session")
def fake_redis() -> Generator[FakeRedis, None, None]:
    redis = FakeRedis()
    yield redis
    redis.close()


@pytest.fixture(scope="session")
def degraded_instance(
    mock_upstream: MockUpstream, fake_redis: FakeRedis
) -> Generator[LaunchedInstance, None, None]:
    """A scratch instance on the fake Redis and a vector DB that refuses every connection."""
    yield from launch(
        mock_upstream,
        {"REDIS_URL": fake_redis.url, "VECTOR_DB": "qdrant", "QDRANT_URI": "http://127.0.0.1:9"},
    )
