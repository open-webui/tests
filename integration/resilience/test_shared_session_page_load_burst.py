"""Regression: with session sharing on, a burst of page loads stalled and failed on a full pool.

Issue #34184. With `DATABASE_ENABLE_SESSION_SHARING` on, a request that had only read kept its
transaction, and with it a pooled connection, open until the response was sent. The chat list,
the inbox count and other page-load endpoints then wait for a second connection, so a burst of
them took every connection in the pool while each waited for another one: they stalled until the
pool timeout and answered 500. Behind PgBouncer in transaction mode the same hold kept a server
connection, which turned a short database pause into a lasting outage. The fix ends a read-only
transaction on the shared session as soon as the call that started it finishes.

The instance gets a pool of 5 and a 3 second pool timeout, so the stall is bounded; the SQLite
engine keeps its overflow of 10, which a burst of 80 still runs out of.

Discriminates: fails on dev 782f4b705 (33 of the 80 page loads on SQLite and 61 on Postgres answer
500 or 400 after the pool timeout); passes with the fix 9bd032197 (branch
fix-shared-session-read-transaction). The write test passes on both.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from harness.actors import create_user

pytestmark = [
    pytest.mark.regression,
    pytest.mark.slow,
    pytest.mark.api,
    pytest.mark.requires_source,
]

SHARED_SESSION_SMALL_POOL = {
    "DATABASE_ENABLE_SESSION_SHARING": "true",
    "DATABASE_POOL_SIZE": "5",
    "DATABASE_POOL_MAX_OVERFLOW": "0",
    "DATABASE_POOL_TIMEOUT": "3",
}
PAGE_LOAD_PATHS = [
    "/api/v1/chats/?page=1",
    "/api/v1/chats/pinned",
    "/api/v1/notifications/inbox/count",
    "/api/v1/folders/",
    "/api/v1/users/user/settings",
]
BURST = 80


@pytest.fixture
def shared(instance_with):
    return instance_with(SHARED_SESSION_SMALL_POOL)


def _create_chat(client, title: str) -> str:
    created = client.post(
        "/api/v1/chats/new",
        json={"chat": {"title": title, "messages": [], "history": {"messages": {}}}},
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def test_a_burst_of_page_loads_is_answered_with_session_sharing_on(shared):
    """Narrow: every page-load request of a burst answers, none waits out the pool."""
    reader = create_user(shared)
    with reader.client() as client:
        _create_chat(client, f"burst {uuid.uuid4().hex[:8]}")
        requests = [PAGE_LOAD_PATHS[index % len(PAGE_LOAD_PATHS)] for index in range(BURST)]
        with ThreadPoolExecutor(max_workers=BURST) as pool:
            answers = list(pool.map(lambda path: (path, client.get(path, timeout=30.0)), requests))

    failed = [(path, answer.status_code) for path, answer in answers if answer.status_code != 200]
    assert len(answers) == BURST
    assert not failed, (
        f"{len(failed)} of {BURST} page loads failed with session sharing on: {failed[:5]}; "
        "each request held its read transaction while it waited for a second connection (#34184)"
    )


def test_a_pinned_chat_is_listed_with_session_sharing_on(shared):
    """Nearby: a write on the shared session is kept and read back."""
    owner = create_user(shared)
    with owner.client() as client:
        title = f"pinned {uuid.uuid4().hex[:8]}"
        chat_id = _create_chat(client, title)
        pinned = client.post(f"/api/v1/chats/{chat_id}/pin")
        assert pinned.status_code == 200, pinned.text
        listed = client.get("/api/v1/chats/pinned")

    assert listed.status_code == 200, listed.text
    assert [chat["title"] for chat in listed.json()] == [title], (
        f"the pinned chat was not listed with session sharing on: {listed.json()}"
    )
