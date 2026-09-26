"""Journey: the model reads knowledge files through `kb_exec` like a small read-only shell.

On an instance with `ENABLE_KB_EXEC`, a chat is offered `kb_exec`, which runs `cat`, `head`,
`tail`, `sed`, `wc`, `stat`, `ls`, `grep` and `find` over the knowledge bases the account can
read, joined with pipes. Files are named by id, file name or folder path; a name two files share
is refused as ambiguous, and a file in a knowledge base the account cannot read is not found by
any name. Unknown commands list the ones that exist, and output past the cap is cut with a note.

The grep, ls, tree and find orderings are pinned by test_knowledge_file_search and
test_knowledge_fs_sort_flags; this module covers the other commands and the pipes.

Discriminates: in a backend copy, `_resolve_file` skipping its access check for an id turned the
foreign file test red; `head` dropping its "showing" note turned the head test red; the pipe
executor handing piped text to no command turned the pipe tests red; `_kb_sed` accepting a
reversed range turned the range test red; and the ambiguity check returning the first match
turned the ambiguous name test red.
"""

from __future__ import annotations

import pytest

from harness.actors import admin_of, create_user
from harness.knowledge_bases import KB_EXEC, add_text_file, knowledge_base
from harness.tool_calls import run_tool

pytestmark = [
    pytest.mark.journey,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

README = ["Handbook", "line two", "line three", "line four", "line five"]
API = "GET /herons lists herons\nPOST /herons adds one"


@pytest.fixture(scope="module")
def handbook(instance_with):
    """A reader of the Handbook (readme.md, docs/api.md and two notes.md), and a hidden ledger."""
    launched = instance_with(KB_EXEC)
    reader = create_user(launched)
    grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
    with (
        admin_of(launched).client() as client,
        knowledge_base(client, "Handbook", [grant]) as handbook_id,
        knowledge_base(client, "Ledger") as ledger_id,
    ):
        add_text_file(client, handbook_id, "readme.md", "\n".join(README))
        docs = client.post(f"/api/v1/knowledge/{handbook_id}/dirs/create", json={"name": "docs"})
        assert docs.status_code == 200, docs.text
        for filename, text in (("api.md", API), ("notes.md", "notes in docs")):
            file_id = add_text_file(client, handbook_id, filename, text)
            moved = client.post(
                f"/api/v1/knowledge/{handbook_id}/file/move",
                json={"file_id": file_id, "directory_id": docs.json()["id"]},
            )
            assert moved.status_code == 200, moved.text
        add_text_file(client, handbook_id, "notes.md", "notes at the root")
        add_text_file(client, handbook_id, "big.txt", "x" * 31_000)
        ledger_file = add_text_file(client, ledger_id, "ledger.md", "secret balance 4000")
        yield {"instance": launched, "reader": reader, "ledger_file": ledger_file}


@pytest.fixture
def kb_exec(handbook):
    """`kb_exec(command)` runs the command for the reader and returns its output."""
    launched = handbook["instance"]
    client = handbook["reader"].client()
    yield lambda command: run_tool(client, launched.upstream, "kb_exec", {"command": command})
    client.close()


def test_cat_numbers_lines_on_request(kb_exec):
    assert kb_exec("cat readme.md") == "\n".join(README)
    assert kb_exec("cat -n readme.md").splitlines() == [
        f"{number}: {line}" for number, line in enumerate(README, 1)
    ]


def test_head_and_tail_say_how_much_they_show(kb_exec):
    assert kb_exec("head -2 readme.md") == "Handbook\nline two\n[showing 2 of 5 lines]"
    assert kb_exec("tail -1 readme.md") == "line five\n[showing last 1 of 5 lines]"
    assert kb_exec("head readme.md") == "\n".join(README)


def test_sed_shows_a_line_range_and_refuses_a_reversed_one(kb_exec):
    assert kb_exec("sed -n '2,3p' readme.md") == "line two\nline three\n[lines 2-3 of 5]"
    assert kb_exec("sed -n '3,2p' readme.md") == "Invalid range: start (3) > end (2)"
    assert kb_exec("sed -n readme.md") == "Usage: sed -n '40,60p' <file>"


def test_wc_and_stat_describe_a_file(kb_exec):
    words = len(" ".join(README).split())
    chars = len("\n".join(README))

    assert kb_exec("wc readme.md") == f"  5  {words}  {chars}  readme.md"
    assert kb_exec("wc -l readme.md") == "  5  readme.md"
    stat = kb_exec("stat readme.md").splitlines()
    assert stat[0] == "  File: readme.md"
    assert " Lines: 5" in stat and stat[-1].startswith("      KB: Handbook (")


def test_a_file_in_a_folder_is_named_by_its_path(kb_exec):
    assert kb_exec("cat docs/api.md") == API
    listing = kb_exec("ls docs/")
    assert "api.md" in listing and "readme.md" not in listing, listing
    [hit] = kb_exec('grep "GET" docs/').splitlines()
    assert hit.endswith("  api.md:1: GET /herons lists herons"), hit


def test_a_name_two_files_share_is_ambiguous(kb_exec):
    output = kb_exec("cat notes.md")

    assert output.startswith('Ambiguous filename "notes.md"'), output
    assert kb_exec("cat docs/notes.md") == "notes in docs"


@pytest.mark.parametrize(
    "command, expected",
    [
        ("cat readme.md | head -2", "Handbook\nline two"),
        ("cat readme.md | tail -2", "line four\nline five"),
        ("cat readme.md | sed -n '2,3p'", "line two\nline three"),
        ('grep "line" readme.md | wc -l', "4"),
        ("cat readme.md | wc", f"  5  9  {len(chr(10).join(README))}"),
    ],
)
def test_pipes_pass_the_text_along(kb_exec, command, expected):
    assert kb_exec(command) == expected


def test_a_file_in_an_unreadable_knowledge_base_is_not_found(kb_exec, handbook):
    ledger_file = handbook["ledger_file"]

    assert kb_exec("cat ledger.md") == "File not found: ledger.md"
    assert kb_exec(f"cat {ledger_file}") == f"File not found: {ledger_file}", (
        "a file id from a knowledge base the reader cannot open was read"
    )
    assert "ledger.md" not in kb_exec("ls -a")


def test_unknown_and_empty_commands_explain_themselves(kb_exec):
    assert kb_exec("rm readme.md").startswith("Unknown command: rm. Available: cat, find, grep")
    assert kb_exec("   ").startswith("Usage: kb_exec(")
    assert kb_exec("cat") == "Usage: cat [-n] <file_id or filename>"
    assert kb_exec("cat missing.md") == "File not found: missing.md"


def test_long_output_is_cut_with_a_note(kb_exec):
    output = kb_exec("cat big.txt")

    assert output.startswith("x" * 30_000)
    assert output[30_000:].startswith("\n[output truncated at 30,000 chars")
