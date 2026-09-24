"""Regression: once Redis went away, every request from a signed-in user failed.

open-webui 0.11.4 fixes `c1615bec2` and `6a85abb3f`: `is_valid_token` read the revocation
markers from Redis with no error handling, so a Redis outage turned every authenticated request
into a 500. It now accepts the token when Redis raises, with one rate-limited warning, so a
sign-out may not take effect until Redis is back. While Redis answers, a signed-out token is
still refused.

Twin of unit/security/test_signin_session_expiry_and_revocation_fallback.py (its Redis half; the
session expiry half stays a unit test).

Discriminates: passes on dev bbfa876af; with c1615bec2 reverted in a copy the session request
after Redis vanished answers 500.
"""

from __future__ import annotations

import socket

import pytest

from harness.actors import create_user
from integration.stateful_redis import StatefulRedis

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]


class VanishingRedis(StatefulRedis):
    """The stateful fake Redis, plus `vanish()`: stop listening and cut every connection."""

    def __init__(self) -> None:
        self.connections: list[socket.socket] = []
        super().__init__()

    def _handler(self):
        base = super()._handler()
        connections = self.connections

        class Handler(base):
            def setup(self) -> None:
                super().setup()
                connections.append(self.connection)

        return Handler

    def vanish(self) -> None:
        self.close()
        for connection in self.connections:
            connection.shutdown(socket.SHUT_RDWR)


@pytest.fixture
def redis():
    redis = VanishingRedis()
    yield redis
    redis.close()


def test_sign_out_revokes_while_redis_answers_and_an_outage_locks_nobody_out(instance_with, redis):
    """Nearby: with Redis up a signed-out token is refused. Narrow: once Redis is gone, a
    signed-in request still answers instead of failing."""
    backend = instance_with({"REDIS_URL": redis.url})
    signed_out = create_user(backend)
    with signed_out.client() as client:
        assert client.post("/api/v1/auths/signout").status_code == 200
        assert client.get("/api/v1/auths/").status_code == 401, "sign-out did not revoke"

    with backend.client() as admin:
        assert admin.get("/api/v1/auths/").status_code == 200
        redis.vanish()
        after_outage = admin.get("/api/v1/auths/")
    assert after_outage.status_code == 200, f"HTTP {after_outage.status_code} {after_outage.text}"
