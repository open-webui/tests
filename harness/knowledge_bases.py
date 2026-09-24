"""Knowledge bases filled through the API and a model preset that has one attached.

`knowledge_base(client)` creates a knowledge base and deletes it afterwards; `add_text_file`
uploads a text file into it the way the workspace's upload does. `model_with_knowledge` is a
preset on the scripted model, readable by every account, with that base attached, so a chat on
it is offered the knowledge tools. `kb_exec` is only offered on an instance booted with `KB_EXEC`.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Iterator

import httpx

from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID

KB_EXEC = {"ENABLE_KB_EXEC": "true"}


@contextlib.contextmanager
def knowledge_base(
    client: httpx.Client, name: str = "Notes", access_grants: list[dict] | None = None
) -> Iterator[str]:
    created = client.post(
        "/api/v1/knowledge/create",
        json={"name": name, "description": "", "access_grants": access_grants or []},
    )
    assert created.status_code == 200, f"creating knowledge base {name} failed: {created.text}"
    knowledge_id = created.json()["id"]
    try:
        yield knowledge_id
    finally:
        client.delete(f"/api/v1/knowledge/{knowledge_id}/delete")


def add_text_file(client: httpx.Client, knowledge_id: str, filename: str, text: str) -> str:
    """Upload `text` as `filename` into the knowledge base; returns the file id."""
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (filename, text.encode(), "text/plain")},
    )
    assert uploaded.status_code == 200, f"uploading {filename} failed: {uploaded.text}"
    file_id = uploaded.json()["id"]
    added = client.post(f"/api/v1/knowledge/{knowledge_id}/file/add", json={"file_id": file_id})
    assert added.status_code == 200, f"adding {filename} to the knowledge base failed: {added.text}"
    return file_id


@contextlib.contextmanager
def model_with_knowledge(client: httpx.Client, knowledge_id: str) -> Iterator[str]:
    """A preset every account may chat with, the knowledge base attached; yields its id."""
    model_id = f"knowledge-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": "Knowledge model",
        "meta": {"knowledge": [{"type": "collection", "id": knowledge_id, "name": "Notes"}]},
        "params": {},
        "access_grants": [EVERYONE_READS],
    }
    created = client.post("/api/v1/models/create", json=form)
    assert created.status_code == 200, f"creating the knowledge model failed: {created.text}"
    try:
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        yield model_id
    finally:
        client.post("/api/v1/models/model/delete", json={"id": model_id})
