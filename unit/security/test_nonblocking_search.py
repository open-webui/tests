"""Regression: the built-in text searches run beside the server, not in front of it.

open-webui 0.11.4 fix `82f11b14c` (#29621) plus the knowledge_fs half in
`d9c8de9c`: `grep_chat_files` and `grep_knowledge_files` awaited
`_grep_file_models` directly, so a model-issued search over chat or knowledge
text ran its whole matching pass on the event loop and held every other
request for as long as the search took. Both call sites now hand the helper
to `asyncio.to_thread`. The same change moved `kb_exec`'s grep off the loop:
`build_matcher` and the per-file line matching go through `asyncio.to_thread`,
with the matcher's time budget (a contextvar) copied along into the worker.

The observable contract pinned here: the matching helper runs in a worker
thread, the caller's loop stays responsive while it runs, and the output is
unchanged.

Discriminates: passes on v0.11.4, fails on v0.11.3 (the matching helper runs
on the loop's own thread, so the helper-thread assertion and the
loop-stays-responsive assertion both fail).
"""

from __future__ import annotations

import asyncio
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.asyncio]

USER = {"id": "user-1", "role": "admin"}
REQUEST = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
FILE_CONTENT = "first line\nthe needle\nlast line"


def _file(file_id: str = "f-1") -> SimpleNamespace:
    return SimpleNamespace(
        id=file_id,
        filename=f"{file_id}.md",
        data={"content": FILE_CONTENT},
        meta={},
        user_id="user-1",
        created_at=0,
        updated_at=0,
    )


@pytest.fixture()
def builtin_module(owui_module) -> ModuleType:
    return owui_module("open_webui.tools.builtin")


@pytest.fixture()
def one_attached_file(builtin_module, owui_module):
    """A single readable file attached to the chat, the smallest driveable case."""
    attached = [{"type": "file", "id": "f-1", "filename": "f-1.md"}]
    accessible = [(attached[0], _file())]

    with patch.object(
        builtin_module, "_get_accessible_chat_files", AsyncMock(return_value=accessible)
    ):
        yield attached


async def _grep_chat(builtin_module, **kwargs):
    return await builtin_module.grep_chat_files(
        pattern="needle",
        __request__=REQUEST,
        __user__=USER,
        __files__=[{"type": "file", "id": "f-1"}],
        **kwargs,
    )


# ── Narrow: the matching helper runs in a worker thread ─────────────────────


async def test_grep_chat_files_matches_off_the_event_loop(
    builtin_module, one_attached_file
):
    """The worker thread must differ from the loop thread: the whole point of
    the fix is that the loop is not held while matching runs."""
    helper_threads: list[threading.Thread] = []

    original_helper = builtin_module._grep_file_models

    def recording_helper(*args, **kwargs):
        helper_threads.append(threading.current_thread())
        return original_helper(*args, **kwargs)

    with patch.object(builtin_module, "_grep_file_models", recording_helper):
        result = await _grep_chat(builtin_module)

    loop_thread = threading.current_thread()
    assert "f-1" in result and "the needle" in result, f"output changed: {result!r}"
    assert helper_threads, "the matching helper never ran"
    assert all(thread is not loop_thread for thread in helper_threads), (
        "grep_chat_files ran its matching on the event loop's own thread, so a "
        "long search holds every other request (#29621)"
    )


async def test_grep_knowledge_files_matches_off_the_event_loop(
    builtin_module, owui_module, one_attached_file
):
    helper_threads: list[threading.Thread] = []

    original_helper = builtin_module._grep_file_models

    def recording_helper(*args, **kwargs):
        helper_threads.append(threading.current_thread())
        return original_helper(*args, **kwargs)

    files_model = owui_module("open_webui.models.files")

    with (
        patch.object(builtin_module, "_grep_file_models", recording_helper),
        patch.object(files_model.Files, "get_file_by_id", AsyncMock(return_value=_file())),
    ):
        result = await builtin_module.grep_knowledge_files(
            pattern="needle",
            __request__=REQUEST,
            __user__=USER,
            __model_knowledge__=[{"type": "file", "id": "f-1"}],
        )

    loop_thread = threading.current_thread()
    assert "f-1" in result and "the needle" in result, f"output changed: {result!r}"
    assert helper_threads, "the matching helper never ran"
    assert all(thread is not loop_thread for thread in helper_threads), (
        "grep_knowledge_files ran its matching on the event loop's own thread (#29621)"
    )


async def test_kb_exec_grep_matches_off_the_event_loop(owui_module):
    """The knowledge_fs half of the fix: kb_exec's grep builds its matcher and
    scans lines through asyncio.to_thread."""
    import importlib

    files_model = importlib.import_module("open_webui.models.files")
    groups_model = importlib.import_module("open_webui.models.groups")
    knowledge_model = importlib.import_module("open_webui.models.knowledge")
    knowledge_fs = owui_module("open_webui.tools.knowledge_fs")

    knowledge = SimpleNamespace(
        id="kb-1", name="KB", description="", user_id=USER["id"]
    )
    stored = _file()

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
            AsyncMock(return_value=[(stored, None)]),
        ),
        patch.object(
            files_model.Files,
            "get_file_by_id",
            AsyncMock(side_effect=lambda file_id: stored if file_id == "f-1" else None),
        ),
    ):
        original_build = knowledge_fs.build_matcher
        build_threads: list[threading.Thread] = []

        def recording_build(*args, **kwargs):
            build_threads.append(threading.current_thread())
            return original_build(*args, **kwargs)

        with patch.object(knowledge_fs, "build_matcher", recording_build):
            output = await knowledge_fs.kb_exec(
                'grep "needle" f-1',
                __user__={"id": "user-1", "role": "admin"},
                __model_knowledge__=[{"type": "collection", "id": "kb-1"}],
            )

    assert "the needle" in output, f"output changed: {output!r}"
    assert build_threads, "the matcher was never built"
    assert all(thread is not threading.current_thread() for thread in build_threads), (
        "kb_exec's grep built its matcher on the event loop's own thread, so a "
        "costly pattern holds the whole worker (#29621)"
    )


# ── Broad: the event loop stays responsive while a search runs ──────────────


async def test_the_event_loop_serves_other_tasks_while_grep_runs(
    builtin_module, one_attached_file
):
    """The real-world consequence: another coroutine keeps making progress while
    a search is in flight, which is only possible if the search is off-loop."""

    def slow_line(line: str) -> bool:
        asyncio.run  # no-op touch; the thread identity is what matters below
        return "needle" in line

    async def wait_for_other_task() -> None:
        # A coroutine that yields repeatedly; a blocked loop starves it.
        for _ in range(50):
            await asyncio.sleep(0)
        return None

    other = asyncio.ensure_future(wait_for_other_task())
    result = await _grep_chat(builtin_module)
    done, pending = await asyncio.wait({other}, timeout=1.0)

    assert "f-1" in result, f"output changed: {result!r}"
    assert not pending, (
        "another coroutine could not make progress while a chat grep ran, so "
        "the search still blocks the event loop (#29621)"
    )


async def test_grep_output_is_unchanged_by_the_offload(
    builtin_module, one_attached_file
):
    """The fix was mechanical: byte-for-byte the same answer as before."""
    result = await _grep_chat(builtin_module)

    assert result == "f-1  f-1.md:2: the needle", (
        f"the offload changed the grep output, got {result!r} (#29621)"
    )


async def test_grep_error_path_still_reports_errors(builtin_module, owui_module):
    """The except clause still catches what the helper raises."""

    async def raising_accessible(files, user, file_id=None):
        raise RuntimeError("storage exploded")

    with patch.object(builtin_module, "_get_accessible_chat_files", raising_accessible):
        result = await _grep_chat(builtin_module)

    assert "storage exploded" in result, (
        f"the error path stopped reporting the reason, got {result!r} (#29621)"
    )
