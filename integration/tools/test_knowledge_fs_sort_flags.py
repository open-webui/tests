"""Regression: kb_exec ls, tree and find listed files in database order and dropped -t.

open-webui 0.11.4, fix `1b67da700` (#29840): the model-facing knowledge base filesystem listed
files in whatever order the database returned and silently accepted `ls -t`, so a model asking
for the newest files held an unordered list. Every listing now goes through `_sort_files`: by
name by default, -t newest first, -S largest first, -r reversed, flags combinable.

The files are uploaded in an order that is neither by name, age nor size, so the database order
differs from every order a flag asks for.

Twin of unit/tools/test_knowledge_fs_sort_flags.py.

Discriminates: passes on dev bbfa876af, fails with `_sort_files` returning its input unchanged
(every listing comes back in upload order); the nearby tests pass on both.
"""

from __future__ import annotations

import time

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

# upload order, oldest first, with sizes crossed against both names and ages
UPLOADS = [("b-second.md", 100), ("c-third.md", 300), ("a-first.md", 200)]
BY_NAME = ["a-first.md", "b-second.md", "c-third.md"]
BY_NAME_REVERSED = ["c-third.md", "b-second.md", "a-first.md"]
BY_NEWEST = ["a-first.md", "c-third.md", "b-second.md"]
BY_LARGEST = ["c-third.md", "a-first.md", "b-second.md"]
BY_SMALLEST = ["b-second.md", "a-first.md", "c-third.md"]


@pytest.fixture(scope="module")
def sorted_model(instance_with):
    """A preset whose knowledge base holds the three files, each modified a second apart."""
    launched = instance_with(KB_EXEC)
    with admin_of(launched).client() as client, knowledge_base(client, "Sorting") as knowledge_id:
        for index, (filename, size) in enumerate(UPLOADS):
            if index:
                time.sleep(1.1)  # modification times have one-second resolution
            add_text_file(client, knowledge_id, filename, "x" * size)
        with model_with_knowledge(client, knowledge_id) as model_id:
            yield model_id


@pytest.fixture
def kb_exec(instance_with, sorted_model):
    """`kb_exec(command)` has the model run the command and returns the tool's output."""
    launched = instance_with(KB_EXEC)
    client = admin_of(launched).client()
    yield lambda command: run_tool(
        client, launched.upstream, "kb_exec", {"command": command}, model=sorted_model
    )
    client.close()


def ls_names(output: str) -> list[str]:
    """Filenames from ls lines like `  <id>  <name>  100 bytes  2026-09-24`."""
    return [line.strip().split("  ")[1] for line in output.splitlines() if "bytes" in line]


def find_names(output: str) -> list[str]:
    """Filenames from find lines like `<id>  <name> (<knowledge base>)`."""
    return [line.split("  ")[1].rsplit(" (", 1)[0] for line in output.strip().splitlines()]


def tree_names(output: str) -> list[str]:
    branches = [line.strip() for line in output.splitlines()]
    return [line[4:] for line in branches if line.startswith(("├── ", "└── "))]


# narrow


@pytest.mark.parametrize(
    "command, expected",
    [
        ("ls", BY_NAME),
        ("ls -t", BY_NEWEST),
        ("ls -S", BY_LARGEST),
        ("ls -r", BY_NAME_REVERSED),
        ("ls -a", BY_NAME),
    ],
)
def test_ls_orders_files_by_its_flags(kb_exec, command, expected):
    output = kb_exec(command)
    assert ls_names(output) == expected, (
        f"{command!r} listed the files out of order, the model could not rely on it (#29840): "
        f"{output!r}"
    )


@pytest.mark.parametrize(
    "command, expected",
    [('find "*.md"', BY_NAME), ('find -t "*.md"', BY_NEWEST), ('find -r "*.md"', BY_NAME_REVERSED)],
)
def test_find_orders_its_matches_by_its_flags(kb_exec, command, expected):
    output = kb_exec(command)
    assert find_names(output) == expected, f"{command!r} returned {output!r} (#29840)"


@pytest.mark.parametrize("command, expected", [("tree", BY_NAME), ("tree -t", BY_NEWEST)])
def test_tree_orders_its_files_by_its_flags(kb_exec, command, expected):
    output = kb_exec(command)
    assert tree_names(output) == expected, f"{command!r} returned {output!r} (#29840)"


# broad: flags combine


def test_combined_flags_apply_together(kb_exec):
    assert ls_names(kb_exec("ls -Sr")) == BY_SMALLEST


# nearby


def test_an_empty_knowledge_base_says_so(instance_with):
    launched = instance_with(KB_EXEC)
    with admin_of(launched).client() as client, knowledge_base(client, "Empty") as knowledge_id:
        with model_with_knowledge(client, knowledge_id) as model_id:
            listing = run_tool(
                client, launched.upstream, "kb_exec", {"command": "ls"}, model=model_id
            )
    assert "(empty)" in listing


def test_grep_is_unaffected_by_the_sort_flags(kb_exec):
    assert kb_exec('grep "nothing-matches-this"').startswith("No matches")
