"""File Context ticked for every model in the admin's Model Defaults.

Since 0.13.1 (bd6fc15be) a model in native function calling gets attached files and knowledge
bases as references it reads through the file and knowledge tools, unless its File Context
capability is ticked; only then does retrieval put the file text into the prompt. Tests about
that text (top k, the RAG template, full context, rerankers, extraction engines) tick it in the
Model Defaults, which every model without its own setting follows.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import httpx

from harness.actors import Actor

MODELS_CONFIG = ("/api/v1/configs/models", "/api/v1/configs/models")


def _save_models_config(client: httpx.Client, config: dict) -> None:
    saved = client.post(MODELS_CONFIG[1], json=config)
    assert saved.status_code == 200, f"saving the Model Defaults failed: {saved.text}"
    # the panel reloads the model list after saving, which is what applies the defaults
    client.get("/api/models", params={"refresh": "true"}).raise_for_status()


@contextlib.contextmanager
def file_context_ticked(admin: Actor) -> Iterator[None]:
    """File Context ticked in the Model Defaults until the block ends."""
    with admin.client() as client:
        current = client.get(MODELS_CONFIG[0])
        assert current.status_code == 200, current.text
        config = current.json()
        metadata = config.get("DEFAULT_MODEL_METADATA") or {}
        capabilities = {**(metadata.get("capabilities") or {}), "file_context": True}
        ticked = {**metadata, "capabilities": capabilities}
        _save_models_config(client, {**config, "DEFAULT_MODEL_METADATA": ticked})
    try:
        yield
    finally:
        # a new connection: the first one can be closed by the keep-alive timeout meanwhile
        with admin.client() as client:
            _save_models_config(client, config)
