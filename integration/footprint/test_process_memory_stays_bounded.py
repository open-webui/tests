"""Guard: deleting a plugin gives its memory back to the process.

Measured on the server process: install a tool or function whose source is a few megabytes
of comment, load it once so the module cache and the source cache both hold it, delete it, and
repeat. The comment vanishes at compile time, so the only copy that can survive delete is the
source text kept in `TOOL_CONTENTS` / `FUNCTION_CONTENTS`, and forty rounds of it show up as
well over a hundred megabytes of resident memory. The control runs the same rounds without the
load, so nothing is cached and the same measurement stays flat.

Twin of unit/footprint/test_unbounded_process_state.py (its deleted-plugin cases).
Read on upstream dev at 4948842be (2026-09-09), where both caches kept the source; #29983 made
delete pop it, so the loaded rounds now assert the memory comes back.
Discriminates: passes on dev bbfa876af, fails with 22d522c55 (#29983) reverted in a copy of it
(over 160 MiB retained after forty loaded rounds).
"""

from __future__ import annotations

import httpx
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.api, pytest.mark.requires_source]

ROUNDS = 40
SOURCE_BYTES = 4 * 1024 * 1024
ALLOWED_GROWTH = 64 * 1024 * 1024

TOOL = '''
class Tools:
    def __init__(self):
        pass

    def ping(self) -> str:
        """Answer pong."""
        return "pong"
'''

FILTER = """
class Filter:
    def __init__(self):
        pass

    async def inlet(self, body: dict, __user__=None) -> dict:
        return body
"""


def _install_load_delete(client: httpx.Client, kind: str, source: str, load: bool) -> None:
    for round_number in range(ROUNDS):
        plugin_id = f"bulk_{kind}_{round_number}"
        created = client.post(
            f"/api/v1/{kind}/create",
            json={
                "id": plugin_id,
                "name": plugin_id,
                "content": source + "\n# " + "x" * SOURCE_BYTES + "\n",
                "meta": {"description": "bulk"},
            },
        )
        created.raise_for_status()
        if load:
            client.get(f"/api/v1/{kind}/id/{plugin_id}/valves/spec").raise_for_status()
        client.delete(f"/api/v1/{kind}/id/{plugin_id}/delete").raise_for_status()


def _growth_after(instance, kind: str, source: str, load: bool) -> int:
    with instance.client() as client:
        _install_load_delete(client, kind, source, load=False)  # a load here would fill the cache
        before = instance.rss_bytes()
        _install_load_delete(client, kind, source, load)
        return instance.rss_bytes() - before


@pytest.mark.parametrize("kind, source", [("tools", TOOL), ("functions", FILTER)])
def test_deleting_unloaded_plugins_keeps_memory_flat(launched_instance, kind, source):
    """Control: without a load the source cache is never filled, so nothing lasts."""
    growth = _growth_after(launched_instance, kind, source, load=False)

    assert growth < ALLOWED_GROWTH, f"{growth / 2**20:.0f} MiB retained after {ROUNDS} rounds"


@pytest.mark.parametrize("kind, source", [("tools", TOOL), ("functions", FILTER)])
def test_deleting_loaded_plugins_releases_their_source(launched_instance, kind, source):
    growth = _growth_after(launched_instance, kind, source, load=True)

    assert growth < ALLOWED_GROWTH, f"{growth / 2**20:.0f} MiB retained after {ROUNDS} rounds"
