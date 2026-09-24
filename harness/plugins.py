"""Install a function (a filter or a pipe) through the admin API for the length of a test.

The admin's Functions page does the same three things: create the function from its source,
switch it on, and optionally make a filter global so it runs on every chat. The function is
deleted again afterwards, so no other test sees it.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Iterator

from harness.actors import Actor


@contextmanager
def installed_function(
    admin: Actor, source: str, *, active: bool = True, is_global: bool = False
) -> Iterator[str]:
    """Yield the id of a new function built from `source`."""
    function_id = f"probe_{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/functions/create",
            json={
                "id": function_id,
                "name": function_id,
                "content": source,
                "meta": {"description": "installed by a regression test"},
            },
        )
        assert created.status_code == 200, f"creating {function_id} failed: {created.text}"
        try:
            if created.json()["is_active"] != active:
                client.post(f"/api/v1/functions/id/{function_id}/toggle").raise_for_status()
            if is_global:
                client.post(f"/api/v1/functions/id/{function_id}/toggle/global").raise_for_status()
            yield function_id
        finally:
            client.delete(f"/api/v1/functions/id/{function_id}/delete")
