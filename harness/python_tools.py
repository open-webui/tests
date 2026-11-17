"""Workspace Python tools, created by the admin the way the tool editor saves one."""

from __future__ import annotations

import contextlib
import uuid
from typing import Iterator

from harness.actors import Actor

EVERYONE_READS = {"principal_type": "user", "principal_id": "*", "permission": "read"}


@contextlib.contextmanager
def python_tool(admin: Actor, source: str, name: str = "Test tools") -> Iterator[str]:
    """Create a tool from `source` that every account may use; yields its id, deletes it after."""
    tool_id = f"tool_{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/tools/create",
            json={
                "id": tool_id,
                "name": name,
                "content": source,
                "meta": {"description": name},
                "access_grants": [EVERYONE_READS],
            },
        )
        assert created.status_code == 200, f"creating tool {name} failed: {created.text}"
        try:
            yield tool_id
        finally:
            # the creating connection can idle out on the server while the test runs
            with admin.client() as fresh:
                fresh.delete(f"/api/v1/tools/id/{tool_id}/delete")
