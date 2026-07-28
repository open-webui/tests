"""Regression tests for knowledge-base search correctness in kb_exec grep.

Three fixes in backend/open_webui/tools/knowledge_fs.py, all shipped in 0.11.0:

* e18e249d5 (PR open-webui/open-webui#27249, issue #26744): the piped and
  single-file grep branches split their text on the two-character sequence
  backslash-n instead of a real newline, so a whole file collapsed into one
  "line": every hit was reported as line 1 and the returned text was the entire
  document.
* 504e724fd (PR open-webui/open-webui#26795, issue #26781): is_regex_pattern
  only recognised the BRE-escaped form of a pipe, so an alternation like
  "alpha|omega" was searched as one literal string and silently matched nothing.
* 8d2fee5d4 (PR open-webui/open-webui#26721, issue #26715): the piped branch
  ignored -c and -l and returned the matching lines regardless, while the
  file-backed branches honoured both flags.

The tests drive the real kb_exec entry point and stub only the model layer
(Knowledges, Files, Groups), so parsing, flag extraction, matcher construction
and output formatting all run as in production.

Discriminates: passes on v0.11.0, fails on v0.10.2 (line numbers collapse to 1,
alternation patterns find nothing, piped -c/-l return the matching lines).
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.asyncio]

KNOWLEDGE_ID = "kb-search"
ADMIN_USER = {"id": "user-1", "role": "admin"}
MODEL_KNOWLEDGE = [{"type": "collection", "id": KNOWLEDGE_ID}]

NOTES_LINES = [
    "alpha appears here",
    "filler line",
    "nothing of interest",
    "the needle is on line four",
    "omega closes the file",
]
NOTES = "\n".join(NOTES_LINES)

OTHER_LINES = [
    "first row of the other file",
    "a needle hides in the second row",
]
OTHER = "\n".join(OTHER_LINES)


def _stored_file(file_id: str, filename: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=file_id,
        filename=filename,
        data={"content": content},
        meta={"size": len(content), "content_type": "text/plain"},
        created_at=0,
        updated_at=0,
    )


@contextmanager
def knowledge_base(*stored_files: SimpleNamespace):
    """Serve the given files as the one knowledge base the user can read."""
    files_model = importlib.import_module("open_webui.models.files")
    groups_model = importlib.import_module("open_webui.models.groups")
    knowledge_model = importlib.import_module("open_webui.models.knowledge")

    by_id = {f.id: f for f in stored_files}
    knowledge = SimpleNamespace(
        id=KNOWLEDGE_ID, name="Search KB", description="", user_id=ADMIN_USER["id"]
    )

    with (
        patch.object(
            groups_model.Groups, "get_groups_by_member_id", AsyncMock(return_value=[])
        ),
        patch.object(
            knowledge_model.Knowledges,
            "get_knowledge_by_id",
            AsyncMock(return_value=knowledge),
        ),
        patch.object(
            knowledge_model.Knowledges,
            "get_files_with_directory_ids",
            AsyncMock(return_value=[(f, None) for f in stored_files]),
        ),
        patch.object(
            files_model.Files,
            "get_file_by_id",
            AsyncMock(side_effect=lambda file_id: by_id.get(file_id)),
        ),
    ):
        yield


@pytest.fixture()
def kb_exec(owui_module):
    module: ModuleType = owui_module("open_webui.tools.knowledge_fs")

    async def _run(command: str) -> str:
        return await module.kb_exec(
            command, __user__=ADMIN_USER, __model_knowledge__=MODEL_KNOWLEDGE
        )

    _run.module = module
    return _run


def _reported_line_numbers(output: str) -> list[int]:
    return [int(line.split(":", 1)[0]) for line in output.split("\n")]


# --- narrow ----------------------------------------------------------------


async def test_single_file_grep_reports_the_real_line_number(kb_exec) -> None:
    """Regression for open-webui/open-webui#26744."""
    with knowledge_base(_stored_file("f-notes", "notes.md", NOTES)):
        output = await kb_exec('grep "needle" notes.md')

    assert output.split("\n") == ["4: the needle is on line four"], (
        f"a hit on the 4th line must be reported as line 4, not {output!r} (#26744)"
    )


async def test_alternation_pattern_matches_every_alternative(kb_exec) -> None:
    """Regression for open-webui/open-webui#26781."""
    with knowledge_base(_stored_file("f-notes", "notes.md", NOTES)):
        output = await kb_exec('grep "alpha|omega" notes.md')

    assert _reported_line_numbers(output) == [1, 5], (
        f'"alpha|omega" must find both alternatives, got {output!r}: a pattern listing '
        "alternatives silently reported the terms as absent (#26781)"
    )


async def test_piped_grep_with_count_flag_returns_a_count(kb_exec) -> None:
    """Regression for open-webui/open-webui#26715."""
    with knowledge_base(_stored_file("f-notes", "notes.md", NOTES)):
        output = await kb_exec('cat notes.md | grep -c "line"')

    assert output.strip() == "2", (
        f"piped grep -c must return the number of matching lines, got {output!r} (#26715)"
    )


async def test_piped_grep_with_filenames_flag_returns_the_source_name(kb_exec) -> None:
    """Regression for open-webui/open-webui#26715."""
    with knowledge_base(_stored_file("f-notes", "notes.md", NOTES)):
        output = await kb_exec('cat notes.md | grep -l "needle"')

    assert output.strip() == "(standard input)", (
        f"piped grep -l must name the source instead of printing matching lines, "
        f"got {output!r} (#26715)"
    )


# --- broad -----------------------------------------------------------------


@pytest.mark.parametrize("flag", ["-c", "-l"])
async def test_line_suppressing_flags_apply_to_files_and_pipes_alike(
    kb_exec, flag: str
) -> None:
    """A flag that suppresses matching lines must suppress them in both modes."""
    with knowledge_base(_stored_file("f-notes", "notes.md", NOTES)):
        from_file = await kb_exec(f'grep {flag} "needle" notes.md')
        from_pipe = await kb_exec(f'cat notes.md | grep {flag} "needle"')

    assert "the needle is on line four" not in from_file, (
        f"grep {flag} on a file leaked the matching line: {from_file!r}"
    )
    assert "the needle is on line four" not in from_pipe, (
        f"grep {flag} on piped input leaked the matching line: {from_pipe!r}: flags "
        "must not be silently dropped when the text arrives through a pipe (#26715)"
    )


@pytest.mark.parametrize("flag,pattern", [("", "needle"), ("-i", "NEEDLE")])
async def test_matching_lines_are_returned_for_files_and_pipes_alike(
    kb_exec, flag: str, pattern: str
) -> None:
    """Without a suppressing flag both modes return the same numbered line."""
    with knowledge_base(_stored_file("f-notes", "notes.md", NOTES)):
        from_file = await kb_exec(f'grep {flag} "{pattern}" notes.md')
        from_pipe = await kb_exec(f'cat notes.md | grep {flag} "{pattern}"')

    expected = "4: the needle is on line four"
    assert from_file == expected, f"file-backed grep {flag!r} returned {from_file!r}"
    assert from_pipe == expected, (
        f"piped grep {flag!r} returned {from_pipe!r}: piped and file-backed search must "
        "agree on every flag they both support"
    )


async def test_count_flag_reports_the_same_total_for_files_and_pipes(kb_exec) -> None:
    """Both modes count the same lines, so -c must agree on the number."""
    with knowledge_base(_stored_file("f-notes", "notes.md", NOTES)):
        from_file = await kb_exec('grep -c "line" notes.md')
        from_pipe = await kb_exec('cat notes.md | grep -c "line"')

    file_count = int(from_file.rsplit(":", 1)[1])
    assert file_count == int(from_pipe.strip()), (
        f"grep -c reported {file_count} for the file but {from_pipe.strip()} for the "
        "same text piped in; the two paths must count identically"
    )


# --- nearby ----------------------------------------------------------------


async def test_absent_pattern_reports_no_matches(kb_exec) -> None:
    with knowledge_base(_stored_file("f-notes", "notes.md", NOTES)):
        from_file = await kb_exec('grep "zebra" notes.md')
        from_pipe = await kb_exec('cat notes.md | grep "zebra"')

    assert from_file == 'No matches for "zebra" in notes.md', from_file
    assert from_pipe == 'No matches for "zebra"', from_pipe


async def test_case_insensitive_search_finds_the_lowercase_line(kb_exec) -> None:
    with knowledge_base(_stored_file("f-notes", "notes.md", NOTES)):
        output = await kb_exec('grep -i "NEEDLE" notes.md')

    assert "the needle is on line four" in output, output


async def test_cross_file_search_attributes_each_hit_to_its_own_file(kb_exec) -> None:
    with knowledge_base(
        _stored_file("f-notes", "notes.md", NOTES),
        _stored_file("f-other", "other.md", OTHER),
    ):
        output = await kb_exec('grep "needle"')

    assert output.split("\n") == [
        "f-notes  notes.md:4: the needle is on line four",
        "f-other  other.md:2: a needle hides in the second row",
    ], f"each hit must carry its own file and line number, got {output!r}"


async def test_match_cap_truncates_and_reports_the_true_total(kb_exec) -> None:
    module = kb_exec.module
    cap = getattr(module, "KNOWLEDGE_GREP_MAX_MATCHES", None) or module.MAX_GREP_MATCHES
    total = cap + 10
    content = "\n".join(f"hit {i}" for i in range(total))

    with knowledge_base(_stored_file("f-many", "many.md", content)):
        output = await kb_exec('grep "hit"')

    lines = output.split("\n")
    assert len(lines) == cap + 1, f"expected {cap} results plus a footer, got {len(lines)}"
    assert lines[-1] == f"[showing {cap} of {total} matches]", (
        f"the cap must be reported honestly so the model knows results were cut: {lines[-1]!r}"
    )


async def test_pattern_matching_every_line_returns_every_line(kb_exec) -> None:
    with knowledge_base(_stored_file("f-notes", "notes.md", NOTES)):
        output = await kb_exec('grep ".*"')

    assert output.split("\n") == [
        f"f-notes  notes.md:{i}: {text}" for i, text in enumerate(NOTES_LINES, 1)
    ], output
