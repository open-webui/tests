"""Source-contract tests for the chat taskIds clobber race.

Regression for open-webui/open-webui#25217.

`chatCompletedHandler` (Chat.svelte) ran fire-and-forget and, after
`await getChatList(...)`, unconditionally did `taskIds = null`. A
concurrent `sendMessage` that resolved during that await would assign
fresh task IDs; the handler then resumed and wiped them, so
`stopResponse()` saw an empty list and the stop-generation button went
inactive — in-flight title/follow-up/tag tasks could no longer be
cancelled.

Fix (PR #25687) has TWO interdependent parts:

  1. chatCompletedHandler snapshots `const capturedTaskIds = taskIds`
     before the await and only clears when `taskIds === capturedTaskIds`
     (nothing was written while it was suspended).

  2. sendMessage writes a NEW array every time
     (`taskIds = [...(taskIds ?? []), ...newTaskIds]`) instead of
     mutating in place with `taskIds.push(...)`. The guard in (1) works
     by reference identity, so an in-place push would leave the
     reference unchanged and defeat it.

Either part regressing silently re-opens the race — which is exactly
why a source audit earns its keep here (a behavioral test of a Svelte
component would miss the (2) ↔ (1) coupling). The external suite has no
JS toolchain; same approach as test_workspace_permissions.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_AWAIT = re.compile(r"\bawait\b")
_SNAPSHOT = re.compile(r"(?:const|let)\s+(\w+)\s*=\s*taskIds\s*;")
# `if (taskIds === <snap>) { taskIds = null }`
_GUARDED_CLEAR = re.compile(
    r"if\s*\(\s*taskIds\s*===\s*\w+\s*\)\s*\{\s*taskIds\s*=\s*null\s*;?\s*\}"
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
# Specific — chatCompletedHandler must guard the clear (#25217)
# =============================================================================


@pytest.mark.regression
def test_chat_completed_handler_guards_taskids_clear(open_webui_backend: Path) -> None:
    """Regression for open-webui/open-webui#25217.

    chatCompletedHandler must not unconditionally null taskIds after its
    await. It has to snapshot taskIds before the await and only clear
    when the reference is unchanged — otherwise it wipes IDs a
    concurrent sendMessage registered while it was suspended.
    """
    src = _read(_chat_svelte(open_webui_backend))
    body = _extract_arrow_body(src, r"const\s+chatCompletedHandler\s*=\s*async[^=]*?=>\s*\{")

    # (a) a snapshot of taskIds is taken before the first await.
    snap = _SNAPSHOT.search(body)
    first_await = _AWAIT.search(body)
    assert snap and first_await and snap.start() < first_await.start(), (
        "Regression of #25217: chatCompletedHandler doesn't snapshot taskIds "
        "before its await. Add `const capturedTaskIds = taskIds;` before "
        "`await getChatList(...)`."
    )

    # (b) the only `taskIds = null` is the guarded one. Strip the guarded
    #     form; if a bare clear remains, it's the unconditional bug.
    stripped = _GUARDED_CLEAR.sub("", body)
    assert "taskIds = null" not in stripped, (
        "Regression of #25217: chatCompletedHandler clears taskIds "
        "unconditionally. Guard it with `if (taskIds === <snapshot>) { "
        "taskIds = null; }` so a concurrent sendMessage's IDs survive."
    )
    # (c) ...and that guarded clear actually exists (not merely removed).
    assert _GUARDED_CLEAR.search(body), (
        "Regression of #25217: no reference-compared `taskIds = null` guard "
        "found in chatCompletedHandler."
    )


# =============================================================================
# Broad — the reference-identity invariant the guard depends on
# =============================================================================


@pytest.mark.regression
def test_taskids_is_never_mutated_in_place(open_webui_backend: Path) -> None:
    """The snapshot guard compares taskIds by reference, so every write
    must REPLACE the array, never mutate it. A `taskIds.push(...)`
    anywhere leaves the reference intact and silently defeats the guard
    in chatCompletedHandler — re-opening #25217 even with the guard in
    place."""
    src = _read(_chat_svelte(open_webui_backend))
    assert "taskIds.push(" not in src, (
        "Regression of #25217: taskIds is mutated in place via "
        "`taskIds.push(...)`. The chatCompletedHandler snapshot guard "
        "detects concurrent writes by reference identity; an in-place push "
        "keeps the same reference and defeats it. Replace with a fresh "
        "array, e.g. `taskIds = [...(taskIds ?? []), ...newTaskIds]`."
    )


@pytest.mark.regression
def test_sendmessage_registers_task_ids_as_new_array(open_webui_backend: Path) -> None:
    """Corroborating: sendMessage must register the backend's task ids by
    assigning a new array (spread), so the reference changes and the
    handler guard can see the concurrent write."""
    src = _read(_chat_svelte(open_webui_backend))
    # A spread assignment to taskIds must exist (the fresh-array write).
    assert re.search(r"taskIds\s*=\s*\[\s*\.\.\.", src), (
        "Regression of #25217: no fresh-array assignment to taskIds "
        "(`taskIds = [...]`) found — sendMessage must replace the reference "
        "when registering new task ids so the snapshot guard works."
    )
