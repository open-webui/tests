"""Regression: a blank embedding model bricked the instance on its next start.

open-webui 0.9.6 made `get_embedding_function` raise at construction when the local engine had no
model loaded (commit 55ca719b). The lifespan builds it on every start, so a blank embedding model,
saved through Admin Panel > Settings > Documents or set in the environment, left an instance that
could not boot, and so could not be reached to undo the setting (issues #25634, #25165). The fix
(PR #25683) moves the check into the returned coroutine: building always succeeds and embedding
fails at use with a clear message.

Twin of unit/config/test_embedding_boot_safety.py; the local-model happy path, which needs a
SentenceTransformer loaded, stays there.

Discriminates: passes on bbfa876af, fails with the missing-model check raised at construction
again (the blank instance exits during boot and saving a blank local model answers 500). The
external-engine and unknown-engine tests pass on both.
"""

from __future__ import annotations

import httpx
import pytest

from harness.actors import admin_of
from harness.mock_embeddings import EMBEDDING_SETTINGS

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

BLANK_LOCAL_EMBEDDING = {"RAG_EMBEDDING_ENGINE": "", "RAG_EMBEDDING_MODEL": ""}


@pytest.fixture
def embedding_settings(admin, preserve) -> dict:
    preserve(EMBEDDING_SETTINGS)
    with admin.client() as client:
        current = client.get(EMBEDDING_SETTINGS[0])
    assert current.status_code == 200, current.text
    return current.json()


def _save_embedding(admin, settings: dict, engine: str, model: str) -> httpx.Response:
    with admin.client() as client:
        return client.post(
            EMBEDDING_SETTINGS[1],
            json={**settings, "RAG_EMBEDDING_ENGINE": engine, "RAG_EMBEDDING_MODEL": model},
        )


def test_an_instance_with_a_blank_embedding_model_boots(instance_with):
    blank = instance_with(BLANK_LOCAL_EMBEDDING)

    assert httpx.get(f"{blank.base_url}/health", timeout=30).status_code == 200
    with admin_of(blank).client() as client:
        embedding = client.get(EMBEDDING_SETTINGS[0])
    assert embedding.status_code == 200, embedding.text
    assert embedding.json()["RAG_EMBEDDING_MODEL"] == ""


def test_embedding_without_a_model_fails_at_use_with_a_logged_reason(instance_with):
    blank = instance_with(BLANK_LOCAL_EMBEDDING)
    offset = blank.log_size()

    with admin_of(blank).client() as client:
        processed = client.post(
            "/api/v1/retrieval/process/text", json={"name": "note", "content": "some text"}
        )

    assert processed.status_code == 500, processed.text
    assert "No embedding model is loaded" in blank.log_since(offset)


def test_saving_a_blank_local_embedding_model_is_accepted(admin, embedding_settings):
    saved = _save_embedding(admin, embedding_settings, engine="", model="")

    assert saved.status_code == 200, (
        "saving a blank local embedding model was refused, and the same construction error "
        f"keeps an instance with that setting from booting (#25634): {saved.text}"
    )


@pytest.mark.parametrize("engine", ["ollama", "openai", "azure_openai"])
def test_saving_an_unconfigured_external_engine_is_accepted(admin, embedding_settings, engine):
    assert _save_embedding(admin, embedding_settings, engine=engine, model="").status_code == 200


def test_an_unknown_engine_is_still_refused(instance, admin, embedding_settings):
    offset = instance.log_size()

    refused = _save_embedding(admin, embedding_settings, engine="not-a-real-engine", model="")

    assert refused.status_code == 500, refused.text
    assert "Unknown embedding engine" in instance.log_since(offset)
