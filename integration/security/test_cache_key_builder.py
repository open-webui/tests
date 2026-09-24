"""Regression: the provider model list is cached per user, not once for everyone.

`routers/openai.py` and `routers/ollama.py` shipped `@cached(key=lambda _, user: ...)` on
`get_all_models`. In aiocache 0.12 `key=` is used verbatim as the cache key and never called,
so every caller shared one entry for the whole TTL and one user's permission-filtered model
list could be served to another. The fix switched to `key_builder=` (branch
fix/cached-key-builder-per-user-models).

With a five-minute cache, a second account's first model listing must reach the provider
itself, while a repeat listing by the same account is served from its own cache entry.

Twin of unit/security/test_cache_key_builder.py (its `ast` sweep over every `@cached` stays
there).

Discriminates: passes on dev bbfa876af; with `key_builder=` turned back into `key=` in
routers/openai.py the second account's listing is served from the first account's entry and
the provider is never asked.
"""

from __future__ import annotations

import pytest

from harness.actors import create_user

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]


@pytest.fixture
def cached_models(instance_with):
    return instance_with({"MODELS_CACHE_TTL": "300"})


def _provider_listings_during(launched, client) -> int:
    before = len(launched.upstream.requests_to("/models"))
    client.get("/api/models").raise_for_status()
    return len(launched.upstream.requests_to("/models")) - before


def test_each_account_gets_its_own_cached_model_list(cached_models):
    with cached_models.client() as admin_client:
        _provider_listings_during(cached_models, admin_client)
        assert _provider_listings_during(cached_models, admin_client) == 0, (
            "the model list is not cached at all, so this test proves nothing"
        )

    with create_user(cached_models).client() as user_client:
        user_listings = _provider_listings_during(cached_models, user_client)
        repeat_listings = _provider_listings_during(cached_models, user_client)

    assert user_listings == 1, (
        "a second account's model listing was served from the first account's cache entry, "
        "so one user's filtered model list reaches another (static aiocache key)"
    )
    assert repeat_listings == 0
