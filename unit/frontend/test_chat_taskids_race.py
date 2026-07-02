"""Source-contract tests for the chat taskIds clobber race.

Regression for open-webui/open-webui#25217.

After a chat completed, the task-id reconciliation ran fire-and-forget
and, after `await getTaskIdsByChatId(...)`, wrote back to `taskIds`. A
concurrent `sendMessage` that resolved during that await would assign
fresh task IDs; the reconciliation then resumed and clobbered them, so
`stopResponse()` saw the wrong list and the stop-generation button went
inactive — in-flight title/follow-up/tag tasks could no longer be
cancelled.

The fix moved the reconciliation OUT of `chatCompletedHandler` (which
is now trivial — it just refreshes the sidebar) and INTO the `loadChat`
region, where it guards against the race with TWO interdependent parts:

  1. loadChat snapshots `const activeTaskIds = taskIds` before the
     await and bails on a reference-inequality check
     (`await getTaskIdsByChatId(...); if (taskIds !== activeTaskIds)
     { return; }`) — if anything was written while it was suspended,
     it leaves those fresh IDs alone.

  2. sendMessage writes a NEW array every time
     (`taskIds = [...(taskIds ?? []), ...newTaskIds]`) instead of
     mutating in place with `taskIds.push(...)`. The guard in (1) works
     by reference identity, so an in-place push would leave the
     reference unchanged and defeat it.

Either part regressing silently re-opens the race — which is exactly
why a source audit earns its keep here (a behavioral test of a Svelte
component would miss the (2) <-> (1) coupling). The external suite has
no JS toolchain; same approach as test_workspace_permissions.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# `const activeTaskIds = taskIds;` — the pre-await snapshot.
_SNAPSHOT = re.compile(r"(?:const|let)\s+(\w+)\s*=\s*taskIds\s*;")
# `await getTaskIdsByChatId(...)` — the suspension point being guarded.
_AWAIT_GET_TASK_IDS = re.compile(r"await\s+getTaskIdsByChatId\s*\(")


def _guard(snapshot: str) -> re.Pattern[str]:
    """`if (taskIds !== <snapshot>) { return; }` — the reference-
    inequality early return that discards a stale reconciliation."""
    return re.compile(
        r"if\s*\(\s*taskIds\s*!==\s*" + re.escape(snapshot) + r"\s*\)\s*\{\s*return\s*;?\s*\}"
    )


def _chat_svelte(open_webui_backend: Path) -> Path:
    return open_webui_backend.parent / "src" / "lib" / "components" / "chat" / "Chat.svelte"


def _read(path: Path) -> str:
    if not path.is_file():
        pytest.skip(f"source file not found: {path}")
    return path.read_text(encoding="utf-8")


def _extract_arrow_body(src: str, decl_pattern: str) -> str:
    """Return the brace-delimited body of an arrow function whose
    declaration matches decl_pattern (up to and including its `=> {`)."""
    m = re.search(decl_pattern, src)
    assert m, f"couldn't locate declaration: {decl_pattern!r}"
    open_brace = src.index("{", m.end() - 1)
    depth = 0
    for i in range(open_brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[open_brace : i + 1]
    raise AssertionError("unbalanced braces while extracting arrow body")


# =============================================================================
# Specific — loadChat must snapshot-and-guard the reconciliation (#25217)
# =============================================================================


@pytest.mark.regression
def test_loadchat_guards_taskids_reconciliation(open_webui_backend: Path) -> None:
    """Regression for open-webui/open-webui#25217.

    The fix moved the taskIds reconciliation into loadChat, which must
    snapshot taskIds into a local const BEFORE `await
    getTaskIdsByChatId(...)` and then bail on a reference-inequality
    check (`if (taskIds !== <snapshot>) { return; }`) before using the
    result — otherwise it clobbers IDs a concurrent sendMessage
    registered while it was suspended.
    """
    src = _read(_chat_svelte(open_webui_backend))
    body = _extract_arrow_body(src, r"const\s+loadChat\s*=\s*async[^=]*?=>\s*\{")

    # (a) taskIds is snapshotted into a local const, and (b) that
    #     snapshot precedes the `await getTaskIdsByChatId(...)` it guards.
    await_get = _AWAIT_GET_TASK_IDS.search(body)
    assert await_get, (
        "Regression of #25217: loadChat no longer awaits getTaskIdsByChatId "
        "— the reconciliation that must be guarded is gone."
    )
    snap = None
    for candidate in _SNAPSHOT.finditer(body):
        if candidate.start() < await_get.start():
            snap = candidate  # last snapshot before the await wins
    assert snap, (
        "Regression of #25217: loadChat doesn't snapshot taskIds before its "
        "`await getTaskIdsByChatId(...)`. Add `const activeTaskIds = taskIds;` "
        "before the await so the write-back can be reference-compared."
    )

    # (c) after the await, a reference-inequality guard on that exact
    #     snapshot short-circuits before taskIds is reassigned.
    guard = _guard(snap.group(1))
    guard_m = guard.search(body)
    assert guard_m and guard_m.start() > await_get.start(), (
        "Regression of #25217: loadChat doesn't bail when taskIds changed "
        f"during the await. Add `if (taskIds !== {snap.group(1)}) {{ return; }}` "
        "after `await getTaskIdsByChatId(...)` so a concurrent sendMessage's "
        "fresh IDs survive instead of being clobbered."
    )


# =============================================================================
# Broad — the reference-identity invariant the guard depends on
# =============================================================================


@pytest.mark.regression
def test_taskids_is_never_mutated_in_place(open_webui_backend: Path) -> None:
    """The snapshot guard compares taskIds by reference, so every write
    must REPLACE the array, never mutate it. A `taskIds.push(...)`
    anywhere leaves the reference intact and silently defeats the guard
    in loadChat — re-opening #25217 even with the guard in place."""
    src = _read(_chat_svelte(open_webui_backend))
    assert "taskIds.push(" not in src, (
        "Regression of #25217: taskIds is mutated in place via "
        "`taskIds.push(...)`. The loadChat snapshot guard detects concurrent "
        "writes by reference identity; an in-place push keeps the same "
        "reference and defeats it. Replace with a fresh array, e.g. "
        "`taskIds = [...(taskIds ?? []), ...newTaskIds]`."
    )


@pytest.mark.regression
def test_sendmessage_registers_task_ids_as_new_array(open_webui_backend: Path) -> None:
    """Corroborating: sendMessage must register the backend's task ids by
    assigning a new array (spread), so the reference changes and the
    loadChat guard can see the concurrent write."""
    src = _read(_chat_svelte(open_webui_backend))
    # A spread assignment to taskIds must exist (the fresh-array write).
    assert re.search(r"taskIds\s*=\s*\[\s*\.\.\.", src), (
        "Regression of #25217: no fresh-array assignment to taskIds "
        "(`taskIds = [...]`) found — sendMessage must replace the reference "
        "when registering new task ids so the snapshot guard works."
    )
