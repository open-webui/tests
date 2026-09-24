"""Filter functions installed through the admin API, the way the Functions page installs them."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Iterator

import httpx


@contextmanager
def global_filter(client: httpx.Client, source: str) -> Iterator[str]:
    """An active filter that runs on every model until the block exits; yields its id."""
    function_id = f"filter_{uuid.uuid4().hex[:8]}"
    created = client.post(
        "/api/v1/functions/create",
        json={"id": function_id, "name": function_id, "content": source, "meta": {}},
    )
    assert created.status_code == 200, f"installing the filter failed: {created.text}"
    try:
        activated = client.post(f"/api/v1/functions/id/{function_id}/toggle")
        assert activated.json()["is_active"], activated.text
        made_global = client.post(f"/api/v1/functions/id/{function_id}/toggle/global")
        assert made_global.json()["is_global"], made_global.text
        yield function_id
    finally:
        client.delete(f"/api/v1/functions/id/{function_id}/delete")
