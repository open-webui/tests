"""Regression: kb_exec ls, tree and find sort their file listings.

open-webui 0.11.4 fix `1b67da700` (#29840): the "ls", "tree" and "find"
commands of the model-facing knowledge-base filesystem tool returned files in
whatever order the database happened to produce, and silently dropped the
"-t" flag the model already knows from a shell. A model asking for "ls -t"
believed it had a newest-first list while actually holding an unordered one.
The fix routes every file listing through `_sort_files`: by name (path)
by default, "-t" newest first, "-S" largest first, "-r" reversed, flags
combinable. Directory lines keep their name order and stay grouped first.

The tests drive the real `kb_exec` entry point and stub only the model
layer, following unit/security/test_knowledge_search_match_budget.py, so
parsing, flag extraction and output formatting all run as in production.
The file store is deliberately handed to the code in a non-sorted order.

Discriminates: passes on v0.11.4, fails on v0.11.3 (listings come back in
the raw store order, so the name-sorted and newest-first assertions fail).
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.asyncio]

KNOWLEDGE_ID = "kb-sort"
ADMIN_USER = {"id": "user-1", "role": "admin"}
MODEL_KNOWLEDGE = [{"type": "collection", "id": KNOWLEDGE_ID}]

# Names deliberately not in sorted order, sizes and dates crossed so a sort
# by the wrong key gives a different sequence than the right one.
FILES = [
    {"id": "f-mid", "name": "b-second.md", "size": 300, "updated_at": 200},
    {"id": "f-new", "name": "c-third.md", "size": 100, "updated_at": 300},
    {"id": "f-old", "name": "a-first.md", "size": 200, "updated_at": 100},
]
BY_NAME = ["a-first.md", "b-second.md", "c-third.md"]
BY_NAME_REVERSED = ["c-third.md", "b-second.md", "a-first.md"]
BY_NEWEST = ["c-third.md", "b-second.md", "a-first.md"]
BY_LARGEST = ["b-second.md", "a-first.md", "c-third.md"]


def _stored_file(file_id: str, name: str, size: int, updated_at: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=file_id,
        filename=name,
        data={"content": ""},
        meta={"size": size, "content_type": "text/plain"},
        created_at=0,
        updated_at=updated_at,
    )


def _files_in_store():
    return [
        _stored_file(entry["id"], entry["name"], entry["size"], entry["updated_at"])
        for entry in FILES
    ]


async def _run(kb_exec: callable, command: str) -> str:
    return await kb_exec(command, __user__=ADMIN_USER, __model_knowledge__=MODEL_KNOWLEDGE)


@pytest.fixture()
def kb_exec(owui_module):
    module: ModuleType = owui_module("open_webui.tools.knowledge_fs")

    async def _run(command: str) -> str:
        return await module.kb_exec(
            command, __user__=ADMIN_USER, __model_knowledge__=MODEL_KNOWLEDGE
        )

    return _run


@pytest.fixture()
def knowledge_base():
    """Serve the files in the deliberately unsorted store order."""
    import importlib

    files_model = importlib.import_module("open_webui.models.files")
    groups_model = importlib.import_module("open_webui.models.groups")
    knowledge_model = importlib.import_module("open_webui.models.knowledge")

    knowledge = SimpleNamespace(
        id=KNOWLEDGE_ID, name="Sort KB", description="", user_id=ADMIN_USER["id"]
    )
    stored = _files_in_store()
    by_id = {f.id: f for f in stored}

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
            AsyncMock(return_value=[(f, None) for f in stored]),
        ),
        patch.object(
            knowledge_model.Knowledges,
            "get_all_directories",
            AsyncMock(return_value=[]),
        ),
        patch.object(
            files_model.Files,
            "get_file_by_id",
            AsyncMock(side_effect=lambda file_id: by_id.get(file_id)),
        ),
    ):
        yield


def _ls_names(output: str) -> list[str]:
    """Extract the filenames from an ls listing, skipping headers."""
    names = []
    for line in output.split("\n"):
        stripped = line.strip()
        if "bytes" not in stripped:
            continue
        # "f-id  name.md  100 bytes  1970-01-01"
        names.append(stripped.split("  ")[1])
    return names


def _find_names(output: str) -> list[str]:
    """Extract the filenames from a find listing: "f-id  name.md (KB)"."""
    names = []
    for line in output.strip().split("\n"):
        name = line.split("  ")[1]
        names.append(name.rsplit(" (", 1)[0])
    return names


def _tree_names(output: str) -> list[str]:
    """Extract file names from a tree listing, preserving order."""
    names = []
    for line in output.split("\n"):
        stripped = line.strip()
        for marker in ("├── ", "└── "):
            if stripped.startswith(marker) and not stripped[4:].startswith("📁"):
                names.append(stripped[4:])
    return names


# ── Narrow: the bug itself ──────────────────────────────────────────────────


async def test_ls_sorts_by_name_by_default(kb_exec, knowledge_base):
    output = await kb_exec("ls")
    assert _ls_names(output) == BY_NAME, (
        f"a plain ls must list files sorted by name, got {output!r}: the model "
        "could not rely on the listing order (#29840)"
    )


async def test_ls_dash_t_lists_newest_first(kb_exec, knowledge_base):
    output = await kb_exec("ls -t")
    assert _ls_names(output) == BY_NEWEST, (
        f'ls -t must sort newest first, got {output!r}: the flag was accepted and '
        "dropped, so the model believed it had a newest-first list (#29840)"
    )


async def test_ls_dash_S_lists_largest_first(kb_exec, knowledge_base):
    output = await kb_exec("ls -S")
    assert _ls_names(output) == BY_LARGEST, (
        f"ls -S must sort largest first, got {output!r} (#29840)"
    )


async def test_ls_dash_r_reverses_the_order(kb_exec, knowledge_base):
    output = await kb_exec("ls -r")
    assert _ls_names(output) == BY_NAME_REVERSED, (
        f"ls -r must reverse the listing, got {output!r} (#29840)"
    )


async def test_find_sorts_matches_by_name(kb_exec, knowledge_base):
    output = await kb_exec('find "*.md"')
    assert _find_names(output) == BY_NAME, (
        f"find must return its matches sorted by name, got {output!r} (#29840)"
    )


async def test_tree_sorts_files_by_name(kb_exec, knowledge_base):
    output = await kb_exec("tree")
    assert _tree_names(output) == BY_NAME, (
        f"tree must list files sorted by name, got {output!r} (#29840)"
    )


# ── Broad: the same flags work everywhere the fix applied them ───────────────


async def test_find_supports_the_sort_flags(kb_exec, knowledge_base):
    newest = await kb_exec('find -t "*.md"')
    reversed_ = await kb_exec('find -r "*.md"')
    assert _find_names(newest) == BY_NEWEST, (
        f"find -t must sort newest first, got {newest!r} (#29840)"
    )
    assert _find_names(reversed_) == BY_NAME_REVERSED, (
        f"find -r must reverse, got {reversed_!r} (#29840)"
    )


async def test_tree_supports_the_sort_flags(kb_exec, knowledge_base):
    newest = await kb_exec("tree -t")
    assert _tree_names(newest) == BY_NEWEST, (
        f"tree -t must sort newest first, got {newest!r} (#29840)"
    )


async def test_ls_flat_mode_is_sorted_too(kb_exec, knowledge_base):
    output = await kb_exec("ls -a")
    # flat lines are "  id  path  size  date", path is the filename at root
    paths = [
        line.strip().split("  ")[1]
        for line in output.split("\n")
        if line.startswith("  f-")
    ]
    assert paths == BY_NAME, (
        f"ls -a must sort its full-path listing by path, got {output!r} (#29840)"
    )


async def test_combined_flags_apply_together(kb_exec, knowledge_base):
    output = await kb_exec("ls -tr")
    assert _ls_names(output) == ["a-first.md", "b-second.md", "c-third.md"], (
        f"ls -tr must reverse the newest-first order into oldest first, got "
        f"{output!r} (#29840)"
    )


# ── Nearby: behaviour the fix must not disturb ──────────────────────────────


async def test_ls_without_files_reports_empty(kb_exec, owui_module):
    import importlib

    knowledge_model = importlib.import_module("open_webui.models.knowledge")
    groups_model = importlib.import_module("open_webui.models.groups")
    knowledge = SimpleNamespace(
        id=KNOWLEDGE_ID, name="Empty KB", description="", user_id=ADMIN_USER["id"]
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
            AsyncMock(return_value=[]),
        ),
        patch.object(
            knowledge_model.Knowledges, "get_all_directories", AsyncMock(return_value=[])
        ),
    ):
        output = await kb_exec("ls")

    assert "(empty)" in output, f"an empty knowledge base must say so, got {output!r}"


async def test_grep_is_unaffected_by_sort_flags(kb_exec, knowledge_base):
    output = await kb_exec('grep "nothing-matches-this"')
    assert "No matches" in output, f"a grep with no hits must still say so, got {output!r}"
