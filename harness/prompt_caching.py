"""The setup the Prompt Caching docs page prescribes, and the check that a request only appends.

A provider caches the start of a request: the tool definitions, then the messages from the
system prompt on. `cache_optimal_model(admin)` creates the model the page's checklist asks for
(a static system prompt with the citation rules, File Upload on, File Context and Citations off,
Builtin Tools on with every category, native function calling), readable by every account and
with a knowledge base every account may read attached. `turn_off_memory_system_context` is the
admin switch step 5 of the page offers.

`prefix_break(earlier, later)` serializes the cached part of two provider requests and returns
`None` when the earlier one is a byte-for-byte prefix of the later one. Otherwise it names the
first differing byte range, and which tool or message it falls in, so a failure shows exactly
what was rewritten. `first_break(requests)` is the first such break in a run of requests.

`cache_optimal_model(admin, with_knowledge=False)` leaves the knowledge base off the model, which
is when the knowledge discovery tools are offered; capabilities passed as keywords replace the
checklist's, for a control that turns one breaker back on.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from dataclasses import dataclass
from typing import Iterator

from harness.actors import Actor
from harness.knowledge_bases import add_text_file, knowledge_base
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID

SYSTEM_PROMPT = (
    "You are the Harbour Office assistant. Answer briefly.\n"
    "Cite passages from files and knowledge as the source filename the tool result carries, "
    "for example (handbook.txt).\n"
    "Cite web pages as markdown links built from the link field.\n"
    "Search the user's memories with the memory tools before answering personal questions."
)
CAPABILITIES = {
    "file_context": False,
    "vision": True,
    "file_upload": True,
    "web_search": True,
    "image_generation": True,
    "code_interpreter": True,
    "terminal": True,
    "citations": False,
    "status_updates": True,
    "memory": True,
    "builtin_tools": True,
}
HANDBOOK = ("handbook.txt", "Harbour Office handbook.\nThe ferry leaves pier 7 at 06:40 daily.\n")
ADMIN_CONFIG = "/api/v1/auths/admin/config"

# what a provider renders into the cached prompt ahead of the messages
CACHED_FIELDS = ("model", "tools", "tool_choice")
CONTEXT_BYTES = 60


def _serialize(value) -> bytes:
    # key order inside an object is not part of the rendered prompt; string bytes are
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def cached_parts(request: dict) -> list[tuple[str, bytes]]:
    """The cached part of a provider request, in the order the provider renders it."""
    parts = [
        (name, name.encode() + b"=" + _serialize(request[name]) + b"\n")
        for name in CACHED_FIELDS
        if name in request
    ]
    for index, message in enumerate(request.get("messages", [])):
        parts.append((f"messages[{index}] ({message.get('role')})", _serialize(message) + b"\n"))
    return parts


def prefix_break(earlier: dict, later: dict) -> str | None:
    """`None` when `earlier`'s cached part is a byte prefix of `later`'s, else where it breaks."""
    earlier_parts = cached_parts(earlier)
    earlier_bytes = b"".join(part for _, part in earlier_parts)
    later_bytes = b"".join(part for _, part in cached_parts(later))
    if later_bytes.startswith(earlier_bytes):
        return None
    offset = next(
        (index for index, (old, new) in enumerate(zip(earlier_bytes, later_bytes)) if old != new),
        min(len(earlier_bytes), len(later_bytes)),
    )
    part_name, part_start = _part_at(earlier_parts, offset)
    start = max(0, offset - CONTEXT_BYTES)
    end = offset + CONTEXT_BYTES
    return (
        f"the cached prefix changed at byte {offset} (in {part_name}, {offset - part_start} "
        f"bytes into it); bytes {start}..{end}:\n"
        f"  before: {earlier_bytes[start:end]!r}\n"
        f"  after:  {later_bytes[start:end]!r}"
    )


def _part_at(parts: list[tuple[str, bytes]], offset: int) -> tuple[str, int]:
    position = 0
    for name, part in parts:
        if offset < position + len(part):
            return name, position
        position += len(part)
    return "the end of the earlier request", position


def first_break(requests: list[dict]) -> str | None:
    for earlier, later in zip(requests, requests[1:]):
        broken = prefix_break(earlier, later)
        if broken:
            return broken
    return None


def assert_append_only(requests: list[dict]) -> None:
    """Every request keeps the cached part of the one before it, byte for byte."""
    assert len(requests) >= 2, f"expected at least two provider requests, got {len(requests)}"
    for number, (earlier, later) in enumerate(zip(requests, requests[1:]), start=1):
        broken = prefix_break(earlier, later)
        assert broken is None, f"request {number + 1} of {len(requests)}: {broken}"


@dataclass
class CachedModel:
    id: str
    knowledge_id: str
    handbook_file_id: str


@contextlib.contextmanager
def cache_optimal_model(
    admin: Actor, with_knowledge: bool = True, **capabilities: bool
) -> Iterator[CachedModel]:
    """The model the page's checklist describes, readable by everyone, with its knowledge base."""
    model_id = f"cached-{uuid.uuid4().hex[:8]}"
    with (
        admin.client() as client,
        knowledge_base(client, "Harbour handbook", [EVERYONE_READS]) as base_id,
    ):
        handbook_file_id = add_text_file(client, base_id, *HANDBOOK)
        form = {
            "id": model_id,
            "base_model_id": MOCK_MODEL_ID,
            "name": f"Cached Assistant {model_id[-8:]}",
            "meta": {
                "capabilities": {**CAPABILITIES, **capabilities},
                "knowledge": [{"type": "collection", "id": base_id, "name": "Harbour handbook"}]
                if with_knowledge
                else [],
            },
            "params": {"system": SYSTEM_PROMPT, "function_calling": "native"},
            "access_grants": [EVERYONE_READS],
        }
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, f"creating the cached model failed: {created.text}"
        try:
            client.get("/api/models", params={"refresh": "true"}).raise_for_status()
            yield CachedModel(model_id, base_id, handbook_file_id)
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def turn_off_memory_system_context(admin: Actor) -> None:
    """Settings > Admin > General > Memory System Context, switched off."""
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG).json()
        saved = client.post(ADMIN_CONFIG, json={**current, "ENABLE_MEMORY_SYSTEM_CONTEXT": False})
    assert saved.status_code == 200, saved.text
