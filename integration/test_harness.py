"""The harness itself: a scripted reply reaches the stored chat, and the request is recorded.

Every chat test builds on these two facts. When they fail, the harness or the chat entrypoint
it drives has changed, and the failures of the tests built on it say nothing about regressions.

With `OWUI_TEST_DATABASE=postgres` or `OWUI_TEST_REDIS=1`, the shared instance really keeps its
accounts in Postgres and its model pool in Redis, so a green run in that mode means something.
"""

from __future__ import annotations

import pytest

from harness import backends
from harness import upstream as reply
from harness.chat import ask
from harness.instance import ADMIN_EMAIL
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.api, pytest.mark.requires_source]


def test_a_scripted_reply_is_stored_on_the_chat(user, upstream):
    upstream.queue(reply.text("scripted answer"))
    with user.client() as client:
        _, message = ask(client, "hello")
    assert message["content"] == "scripted answer"


def test_the_provider_records_the_prompt_it_was_sent(user, upstream):
    with user.client() as client:
        ask(client, "a distinctive prompt")
    sent = upstream.chat_requests()[-1]["messages"]
    assert any(entry.get("content") == "a distinctive prompt" for entry in sent)


@pytest.mark.skipif(backends.DATABASE != "postgres", reason="OWUI_TEST_DATABASE is not postgres")
def test_the_shared_instance_keeps_its_accounts_in_postgres(instance, admin):
    import sqlalchemy

    assert backends.on_postgres(instance), instance.database_url
    engine = sqlalchemy.create_engine(instance.database_url)
    try:
        with engine.connect() as connection:
            emails = connection.execute(sqlalchemy.text('SELECT email FROM "user"')).scalars()
            assert ADMIN_EMAIL in set(emails)
    finally:
        engine.dispose()


@pytest.mark.skipif(not backends.REDIS, reason="OWUI_TEST_REDIS is not set")
def test_the_shared_instance_keeps_its_model_pool_in_redis(instance):
    import redis

    assert instance.redis_url, "the shared instance was booted without Redis"
    with redis.Redis.from_url(instance.redis_url, decode_responses=True) as client:
        assert MOCK_MODEL_ID in client.hkeys("open-webui:models")
