"""Regression: kb_exec grep reported every hit as line 1, missed alternations and ignored -c/-l.

Three fixes in `tools/knowledge_fs.py`, all shipped in 0.11.0:

* `e18e249d5` (PR #27249, issue #26744): the piped and single-file grep branches split their
  text on the two characters backslash-n instead of a newline, so a whole file was one "line":
  every hit came back as line 1 carrying the entire document.
* `504e724fd` (PR #26795, issue #26781): `is_regex_pattern` only recognised the escaped pipe, so
  "alpha|omega" was searched as one literal string and silently matched nothing.
* `8d2fee5d4` (PR #26721, issue #26715): the piped branch ignored -c and -l and returned the
  matching lines, while the file branches honoured both flags.

The model calls kb_exec on a preset with the knowledge base attached, and the test reads the
tool result the provider is sent back.

Twin of unit/tools/test_knowledge_file_search.py.

Discriminates: passes on dev bbfa876af; fails with the literal backslash-n split restored in
both grep branches (the hit is reported as line 1 of one long line), with `|` dropped from the
regex detection (the alternation finds nothing) and with the piped branch's -c/-l handling
removed (the pipe returns the matching line); the nearby tests pass on all three.
"""

from __future__ import annotations

import pytest

from harness.actors import admin_of
from harness.knowledge_bases import KB_EXEC, add_text_file, knowledge_base, model_with_knowledge
from harness.tool_calls import run_tool

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

NOTES_LINES = [
    "alpha appears here",
    "filler line",
    "nothing of interest",
    "the needle is on line four",
    "omega closes the file",
]
OTHER_LINES = ["first row of the other file", "a needle hides in the second row"]
MANY_HITS = 80
NEEDLE_HIT = "4: the needle is on line four"


@pytest.fixture(scope="module")
def notes_model(instance_with):
    """A preset whose knowledge base holds notes.md, other.md and many.md."""
    launched = instance_with(KB_EXEC)
    with admin_of(launched).client() as client, knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "notes.md", "\n".join(NOTES_LINES))
        add_text_file(client, knowledge_id, "other.md", "\n".join(OTHER_LINES))
        add_text_file(client, knowledge_id, "many.md", "\n".join(["hit"] * MANY_HITS))
        with model_with_knowledge(client, knowledge_id) as model_id:
            yield model_id


@pytest.fixture
def kb_exec(instance_with, notes_model):
    """`kb_exec(command)` has the model run the command and returns the tool's output."""
    launched = instance_with(KB_EXEC)
    client = admin_of(launched).client()
    yield lambda command: run_tool(
        client, launched.upstream, "kb_exec", {"command": command}, model=notes_model
    )
    client.close()


# narrow


def test_single_file_grep_reports_the_real_line_number(kb_exec):
    output = kb_exec('grep "needle" notes.md')
    assert output == NEEDLE_HIT, (
        f"a hit on the fourth line must be reported as line 4, got {output!r} (#26744)"
    )


def test_an_alternation_matches_every_alternative(kb_exec):
    output = kb_exec('grep "alpha|omega" notes.md')
    assert output.splitlines() == ["1: alpha appears here", "5: omega closes the file"], (
        f'"alpha|omega" must find both alternatives, got {output!r} (#26781)'
    )


def test_piped_grep_with_the_count_flag_returns_a_count(kb_exec):
    output = kb_exec('cat notes.md | grep -c "line"')
    assert output.strip() == "2", f"piped grep -c must return a count, got {output!r} (#26715)"


def test_piped_grep_with_the_filenames_flag_names_the_source(kb_exec):
    output = kb_exec('cat notes.md | grep -l "needle"')
    assert output.strip() == "(standard input)", (
        f"piped grep -l must name the source, got {output!r} (#26715)"
    )


# broad: a pipe and a file agree on every flag


@pytest.mark.parametrize("flag", ["-c", "-l"])
def test_line_suppressing_flags_apply_to_files_and_pipes_alike(kb_exec, flag):
    from_file = kb_exec(f'grep {flag} "needle" notes.md')
    from_pipe = kb_exec(f'cat notes.md | grep {flag} "needle"')

    assert "the needle is on line four" not in from_file, from_file
    assert "the needle is on line four" not in from_pipe, (
        f"grep {flag} on piped text leaked the matching line: {from_pipe!r} (#26715)"
    )


@pytest.mark.parametrize("flag, pattern", [("", "needle"), ("-i", "NEEDLE")])
def test_matching_lines_are_numbered_alike_for_files_and_pipes(kb_exec, flag, pattern):
    assert kb_exec(f'grep {flag} "{pattern}" notes.md') == NEEDLE_HIT
    assert kb_exec(f'cat notes.md | grep {flag} "{pattern}"') == NEEDLE_HIT


def test_the_count_flag_agrees_for_files_and_pipes(kb_exec):
    from_file = kb_exec('grep -c "line" notes.md')
    from_pipe = kb_exec('cat notes.md | grep -c "line"')

    assert int(from_file.rsplit(":", 1)[1]) == int(from_pipe.strip()) == 2


# nearby


def test_an_absent_pattern_reports_no_matches(kb_exec):
    assert kb_exec('grep "zebra" notes.md') == 'No matches for "zebra" in notes.md'
    assert kb_exec('cat notes.md | grep "zebra"') == 'No matches for "zebra"'


def test_a_pattern_matching_every_line_numbers_every_line(kb_exec):
    output = kb_exec('grep ".*" notes.md')
    assert output.splitlines() == [
        f"{number}: {line}" for number, line in enumerate(NOTES_LINES, 1)
    ]


def test_a_search_across_files_attributes_each_hit_to_its_file(kb_exec):
    hits = kb_exec('grep "needle"').splitlines()

    assert [hit.split("  ", 1)[1] for hit in hits] == [
        "notes.md:4: the needle is on line four",
        "other.md:2: a needle hides in the second row",
    ]


def test_the_match_cap_reports_the_true_total(kb_exec):
    lines = kb_exec('grep "hit"').splitlines()

    shown = len(lines) - 1
    assert 0 < shown < MANY_HITS
    assert lines[-1] == f"[showing {shown} of {MANY_HITS} matches]"
