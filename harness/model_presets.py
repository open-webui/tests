"""Workspace models over the scripted model, the way the model editor saves one.

`preset_with_capabilities(admin, capabilities)` is a preset every account may read, with
`capabilities` as its Capabilities checkboxes and native function calling; it yields the
preset's id and deletes it again afterwards.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Iterator

from harness.actors import Actor
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID


@contextlib.contextmanager
def preset_with_capabilities(admin: Actor, capabilities: dict) -> Iterator[str]:
    model_id = f"capable-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "name": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "meta": {"capabilities": capabilities},
        "params": {"function_calling": "native"},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, f"creating {model_id} failed: {created.text}"
        try:
            client.get("/api/models").raise_for_status()  # registers the preset for chats
            yield model_id
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model_id})
