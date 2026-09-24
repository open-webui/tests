"""Regression: loading a chat overwrote the task ids of a message sent while it loaded.

open-webui issue #25217, fixed by 2856def6c. Loading a chat reconciles its running tasks: it
awaits `getTaskIdsByChatId` and then writes `taskIds`. A message sent during that await
registered fresh task ids, and the reconciliation, resuming with its stale answer, replaced them,
so Stop no longer covered the new generation. The fix snapshots `taskIds` before the await and
backs off when the reference changed. The check is by reference, so it also depends on every
write assigning a new array.

Stays a unit audit: in the current Chat.svelte the Stop button follows the streaming message's
own `done` flag and `stopResponse` cancels by chat id, so the clobber's only visible trace is the
message delete action being offered mid-stream, which a browser test could only catch after a
timed wait for the resumed lookup.

Discriminates: passes on bbfa876af; fails with the `if (taskIds !== activeTaskIds) return` guard
removed (narrow), or with the new task ids pushed onto the existing array (broad).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

CHAT_COMPONENT = "lib/components/chat/Chat.svelte"


@pytest.fixture(scope="module")
def chat_source(open_webui_backend: Path) -> str:
    path = open_webui_backend.parent / "src" / CHAT_COMPONENT
    assert path.is_file(), f"{CHAT_COMPONENT} is gone; retarget this audit at the chat component"
    return path.read_text(encoding="utf-8")


def enclosing_block(source: str, position: int) -> tuple[str, int]:
    """The `{...}` block around `position`, and `position` relative to it."""
    start, depth = position, 0
    while depth >= 0:
        start -= 1
        depth += {"}": 1, "{": -1}.get(source[start], 0)
    end, depth = position, 0
    while depth >= 0:
        depth += {"{": 1, "}": -1}.get(source[end], 0)
        end += 1
    return source[start:end], position - start


def test_the_task_reconciliation_backs_off_when_task_ids_changed_during_its_await(chat_source):
    awaits = [call.start() for call in re.finditer(r"await\s+getTaskIdsByChatId\b", chat_source)]
    blocks = [enclosing_block(chat_source, position) for position in awaits]
    # a reconciliation is an await whose block then writes the answer into taskIds
    reconciliations = [
        (block[:offset], block[offset:])
        for block, offset in blocks
        if re.search(r"\btaskIds\s*=[^=]", block[offset:])
    ]
    assert reconciliations, f"no task reconciliation left in {CHAT_COMPONENT}; retarget"

    for before, after in reconciliations:
        snapshots = re.findall(r"\bconst\s+(\w+)\s*=\s*taskIds\s*;", before)
        assert snapshots, "taskIds is not snapshotted before the reconciliation awaits (#25217)"
        guard = re.search(
            rf"if\s*\(\s*taskIds\s*!==\s*{snapshots[-1]}\s*\)\s*\{{[^{{}}]*\breturn\b", after
        )
        first_write = re.search(r"\btaskIds\s*=[^=]", after)
        assert guard and guard.start() < first_write.start(), (
            "the reconciliation writes taskIds without first checking that no message "
            "registered new task ids during its await, so it overwrites them (#25217)"
        )


def test_task_ids_are_replaced_never_mutated(chat_source):
    """The guard compares references, so an in-place change would slip past it."""
    in_place = re.findall(
        r"\btaskIds\s*(?:\?\.|\.)\s*(?:push|splice|unshift|pop|shift)\s*\(", chat_source
    )
    in_place += re.findall(r"\btaskIds\s*\[[^\]]*\]\s*=[^=]", chat_source)
    assert not in_place, f"taskIds is changed in place ({in_place}), defeating the guard (#25217)"
