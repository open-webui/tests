"""Regression: a tool result over the Tool Output Limit is saved in full to the chat's terminal.

open-webui/open-webui#34619. With the admin's Tool Output Limit set, a longer tool result reaches
the model cut off with a `[Truncated: N characters, limit M]` note. When the chat has an Open
Terminal connected, the full result of a tool that is not the terminal's own is now written to
`.tool_outputs/<id>.txt` on the terminal and the note names that path, so the model can read the
rest. Without a terminal the note stays plain and nothing is written.

An admin's Python tool, readable by everyone, returns a long report; the scripted model calls it
and the test reads the tool result Open WebUI sent back to the provider, the stored chat and the
terminal's home. The test that saves the report is red on dev until the fix merges.

Discriminates: passes on the fix 991bf3989 (open-webui dev 58fb2453b plus the change), fails on
dev 58fb2453b (the note names no path and the terminal holds no file). The test without a
terminal passes on both.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.python_tools import python_tool
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from harness.upstream import conversation

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

LIMIT = 1000
REPORT = "".join(
    f"row {index:04d} of the sample report, value {index * 7}\n" for index in range(120)
)
PLAIN_NOTE = f"[Truncated: {len(REPORT)} characters, limit {LIMIT}]"
SAVED_NOTE = re.compile(
    rf"\[Truncated: {len(REPORT)} characters, limit {LIMIT}, "
    r"full output saved to (?P<path>\S+) on the terminal\]$"
)
REPORT_TOOL = f'''
class Tools:
    def long_report(self) -> str:
        """Return the long sample report."""
        return {REPORT!r}
'''


@pytest.fixture
def member(make_user):
    return make_user()


@pytest.fixture
def report_tool(admin):
    with python_tool(admin, REPORT_TOOL, name="Report tools") as tool_id:
        yield tool_id


@pytest.fixture
def output_limit(admin, preserve):
    preserve("admin_config")
    with admin.client() as client:
        current = client.get("/api/v1/auths/admin/config").json()
        saved = client.post(
            "/api/v1/auths/admin/config", json={**current, "TOOL_OUTPUT_MAX_CHARS": LIMIT}
        )
    assert saved.status_code == 200, f"setting the Tool Output Limit failed: {saved.text}"


@pytest.fixture
def terminal_id(admin, member, open_terminal, preserve) -> str:
    """The Open Terminal, connected by the admin and shared with `member`; returns its id."""
    preserve(TERMINAL_SERVERS_CONFIG)
    connection = open_terminal.connection(config={"access_grants": [read_grant(member.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)
    return connection["id"]


def _run_report(actor, upstream, tool_id: str, **options) -> tuple[str, str]:
    """The model calls the report tool; returns what it was sent back and the stored chat."""
    prompt = f"summarise the report {uuid.uuid4().hex[:8]}"
    answering = reply.answering(prompt)
    upstream.queue(
        reply.tool_call("long_report", {}, match=answering), reply.text("done", match=answering)
    )
    with actor.client() as client:
        turn, message = ask(client, prompt, tool_ids=[tool_id], **options)
        stored = client.get(f"/api/v1/chats/{turn.chat_id}")
    assert message.get("content") == "done", f"the reply did not finish: {message}"
    assert stored.status_code == 200, stored.text
    requests = [request for request in upstream.chat_requests() if answering(request)]
    results = [entry["content"] for entry in conversation(requests[-1]) if entry["role"] == "tool"]
    assert results, f"long_report never ran; the model was sent {conversation(requests[-1])}"
    return results[-1], json.dumps(stored.json())


def _saved_outputs(home: Path) -> set[Path]:
    return set((home / ".tool_outputs").glob("*.txt"))


def test_a_cut_tool_result_is_saved_in_full_to_the_chats_terminal(
    member, report_tool, output_limit, terminal_id, open_terminal, upstream
):
    before = _saved_outputs(open_terminal.home)

    result, stored = _run_report(member, upstream, report_tool, terminal_id=terminal_id)

    kept, _, note = result.rpartition("\n\n")
    assert kept == REPORT[:LIMIT], f"the model did not get the report's first {LIMIT} characters"
    saved = SAVED_NOTE.fullmatch(note)
    assert saved, f"the note does not say where the full output went: {note!r}"
    new_files = _saved_outputs(open_terminal.home) - before
    assert new_files == {Path(saved["path"])}, (
        f"the terminal holds {sorted(new_files)}, the note names {saved['path']}"
    )
    assert Path(saved["path"]).read_bytes() == REPORT.encode(), (
        "the saved file differs from the tool's full output"
    )
    assert note in stored and REPORT[LIMIT:] not in stored.replace("\\n", "\n"), (
        "the stored chat does not hold the cut result"
    )


def test_without_a_terminal_a_cut_tool_result_gets_the_plain_note(
    member, report_tool, output_limit, open_terminal, upstream
):
    before = _saved_outputs(open_terminal.home)

    result, stored = _run_report(member, upstream, report_tool)

    assert result == f"{REPORT[:LIMIT]}\n\n{PLAIN_NOTE}", (
        f"a chat without a terminal got {result[LIMIT:]!r} after the cut"
    )
    assert _saved_outputs(open_terminal.home) == before, "a chat without a terminal saved a file"
    assert PLAIN_NOTE in stored, "the stored chat does not hold the cut result"
