"""Point a launched instance's RAG embeddings at the scripted provider.

`launch()` sets `RAG_EMBEDDING_ENGINE=openai` expecting the mock to serve embeddings, but
`open_webui.config` resets `OPENAI_API_BASE_URL` to api.openai.com before
`RAG_OPENAI_API_BASE_URL` defaults to it, so without this every file processed into a knowledge
base is embedded against the real OpenAI API and fails.
"""

from __future__ import annotations

from typing import Callable

from harness.actors import Actor
from harness.upstream import MockUpstream

EMBEDDING_SETTINGS = ("/api/v1/retrieval/embedding", "/api/v1/retrieval/embedding/update")


def embed_through(upstream: MockUpstream, admin: Actor, preserve: Callable[..., None]) -> None:
    """Route embeddings to `upstream` until `preserve` restores the settings after the test."""
    preserve(EMBEDDING_SETTINGS)
    with admin.client() as client:
        current = client.get(EMBEDDING_SETTINGS[0]).json()
        routed = client.post(
            EMBEDDING_SETTINGS[1],
            json={**current, "openai_config": {"url": upstream.base_url, "key": "sk-mock"}},
        )
    assert routed.status_code == 200, routed.text
