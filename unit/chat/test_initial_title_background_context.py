"""Regression for open-webui/open-webui#30339: the initial title task crashed every new chat.

Two same-day refactors on 2026-06-29 (both in v0.10.0) drifted apart: `2560533c1`
made `background_tasks_handler` read `ctx['model']` unconditionally for the memory
review, and `754787f43` added the early title task in `main.py`, which calls the
same handler with a slimmed context that has no `model` key. Every new saved chat
therefore ended the title task with `KeyError: 'model'`, logged as "Error generating
initial chat title". The title itself had already been written by then, and the
memory settings played no part (the read happens before any of them are consulted).
`log.debug` hid the traceback until PR #30106 (v0.11.4) raised it to
`log.exception`, which is how it got reported.

The fix reads `ctx.get('model')`; the review already accepts `None` and stops on the
title path because that context carries no assistant message.

Discriminates: the narrow tests and the context audit fail on 344ea5306 (v0.11.4) and
pass on the fix. The nearby tests pass on both and pin the behaviour the fix relies on.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

TITLE_RESPONSE = {"choices": [{"message": {"content": '{"title":"Generated title"}'}}]}


@pytest.fixture(scope="module")
def middleware_module(owui_module):
    return owui_module("open_webui.utils.middleware")


@pytest.fixture(scope="module")
def memory_module(owui_module):
    return owui_module("open_webui.utils.memory")


@pytest.fixture(scope="module")
def tasks_enum(owui_module):
    return owui_module("open_webui.constants").TASKS


def _messages_map():
    """A fresh chat as the title task sees it: the user turn and an empty assistant stub."""
    return {
        "u1": {
            "id": "u1",
            "role": "user",
            "content": "hello there",
            "parentId": None,
            "childrenIds": ["a1"],
        },
        "a1": {
            "id": "a1",
            "role": "assistant",
            "content": "",
            "model": "gpt-4o-mini",
            "parentId": "u1",
            "childrenIds": [],
        },
    }


def _title_ctx(tasks_enum, events, *, memory_feature=True, title_task=True, **extra):
    """Mirror of the `title_ctx` dict `main.py` hands to `background_tasks_handler`."""

    async def event_emitter(event):
        events.append(event)

    return {
        "request": SimpleNamespace(),
        "form_data": {"model": "gpt-4o-mini", "messages": []},
        "user": SimpleNamespace(id="alice", role="admin"),
        "metadata": {
            "chat_id": "chat-1",
            "message_id": "a1",
            "features": {"memory": memory_feature},
        },
        "tasks": {tasks_enum.TITLE_GENERATION: title_task},
        "event_emitter": event_emitter,
        **extra,
    }


@contextmanager
def _saved_chat(middleware, *, messages_map=None, title_response=TITLE_RESPONSE):
    """Serve the chat store and task-model calls the handler makes for a saved chat."""
    chats = middleware.Chats
    mocks = {
        "get_messages_map": AsyncMock(return_value=messages_map or _messages_map()),
        "update_title": AsyncMock(),
        "update_tags": AsyncMock(),
        "upsert_message": AsyncMock(),
        "generate_title": AsyncMock(return_value=title_response),
        "generate_tags": AsyncMock(),
        "generate_follow_ups": AsyncMock(),
    }
    with (
        patch.object(middleware, "is_saved_chat_id", lambda chat_id: True),
        patch.object(chats, "get_messages_map_by_chat_id", mocks["get_messages_map"]),
        patch.object(chats, "update_chat_title_by_id", mocks["update_title"]),
        patch.object(chats, "update_chat_tags_by_id", mocks["update_tags"]),
        patch.object(chats, "upsert_message_to_chat_by_id_and_message_id", mocks["upsert_message"]),
        patch.object(middleware, "generate_title", mocks["generate_title"]),
        patch.object(middleware, "generate_chat_tags", mocks["generate_tags"]),
        patch.object(middleware, "generate_follow_ups", mocks["generate_follow_ups"]),
    ):
        yield mocks


@contextmanager
def _memory_review_probe(memory_module, *, enabled=True):
    """Leave `review_memory_after_turn` real, record whether it got past its cheap gates."""
    config = AsyncMock(
        return_value={
            "memories.enable": enabled,
            "memories.background_review.enable": enabled,
            "memories.review_interval_turns": 1,
            "user.permissions": {"features": {"memories": True}},
        }
    )
    review = AsyncMock()
    with (
        patch.object(memory_module.Config, "get_many", config),
        patch.object(memory_module, "_review_memory", review),
    ):
        yield SimpleNamespace(config=config, review=review)


# --- narrow: the title-only context ------------------------------------------------


@pytest.mark.asyncio
async def test_title_only_context_completes_without_a_key_error(
    middleware_module, memory_module, tasks_enum
):
    events = []
    with _saved_chat(middleware_module), _memory_review_probe(memory_module):
        await middleware_module.background_tasks_handler(_title_ctx(tasks_enum, events))


@pytest.mark.asyncio
async def test_title_only_context_writes_the_title_and_emits_it(
    middleware_module, memory_module, tasks_enum
):
    events = []
    with (
        _saved_chat(middleware_module) as store,
        _memory_review_probe(memory_module),
    ):
        await middleware_module.background_tasks_handler(_title_ctx(tasks_enum, events))

    store["update_title"].assert_awaited_once_with("chat-1", "Generated title")
    assert [e for e in events if e["type"] == "chat:title"] == [
        {"type": "chat:title", "data": "Generated title"}
    ]


@pytest.mark.asyncio
async def test_title_only_context_does_not_review_memory(
    middleware_module, memory_module, tasks_enum
):
    """No model, no assistant text: the review must not even read the memory config."""
    events = []
    with (
        _saved_chat(middleware_module),
        _memory_review_probe(memory_module) as probe,
    ):
        await middleware_module.background_tasks_handler(_title_ctx(tasks_enum, events))

    assert probe.config.call_count == 0
    assert probe.review.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("memory_feature", [True, False])
@pytest.mark.parametrize("memories_enabled", [True, False])
async def test_title_only_context_is_independent_of_every_memory_setting(
    middleware_module, memory_module, tasks_enum, memory_feature, memories_enabled
):
    """The reporter blamed `memories.enable`; the crash never looked at it."""
    events = []
    with (
        _saved_chat(middleware_module) as store,
        _memory_review_probe(memory_module, enabled=memories_enabled) as probe,
    ):
        await middleware_module.background_tasks_handler(
            _title_ctx(tasks_enum, events, memory_feature=memory_feature)
        )

    store["update_title"].assert_awaited_once()
    assert probe.review.call_count == 0


@pytest.mark.asyncio
async def test_title_only_context_falls_back_to_the_user_message_as_title(
    middleware_module, memory_module, tasks_enum
):
    """The fallback title sits on the same path and crashed the same way."""
    events = []
    with (
        _saved_chat(middleware_module, title_response=None) as store,
        _memory_review_probe(memory_module),
    ):
        await middleware_module.background_tasks_handler(_title_ctx(tasks_enum, events))

    store["update_title"].assert_awaited_once_with("chat-1", "hello there")


@pytest.mark.asyncio
async def test_title_only_context_with_generation_switched_off_still_completes(
    middleware_module, memory_module, tasks_enum
):
    """`initial_title_generation=False` is passed through as a task value, not dropped."""
    events = []
    with (
        _saved_chat(middleware_module) as store,
        _memory_review_probe(memory_module),
    ):
        await middleware_module.background_tasks_handler(
            _title_ctx(tasks_enum, events, title_task=False)
        )

    store["generate_title"].assert_not_awaited()
    store["update_title"].assert_awaited_once_with("chat-1", "hello there")


@pytest.mark.asyncio
async def test_title_only_context_runs_no_other_task(middleware_module, memory_module, tasks_enum):
    """Tags and follow-ups belong to the main path after the reply is complete."""
    events = []
    with (
        _saved_chat(middleware_module) as store,
        _memory_review_probe(memory_module),
    ):
        await middleware_module.background_tasks_handler(_title_ctx(tasks_enum, events))

    store["generate_tags"].assert_not_awaited()
    store["generate_follow_ups"].assert_not_awaited()
    store["update_tags"].assert_not_awaited()
    assert {e["type"] for e in events} == {"chat:title"}


@pytest.mark.asyncio
async def test_title_only_context_survives_a_longer_thread(
    middleware_module, memory_module, tasks_enum
):
    """A regenerated first reply already has three messages in the map."""
    messages_map = _messages_map()
    messages_map["a0"] = {
        "id": "a0",
        "role": "assistant",
        "content": "first attempt",
        "model": "gpt-4o-mini",
        "parentId": "u1",
        "childrenIds": [],
    }
    messages_map["u1"]["childrenIds"] = ["a0", "a1"]
    events = []
    with (
        _saved_chat(middleware_module, messages_map=messages_map) as store,
        _memory_review_probe(memory_module) as probe,
    ):
        await middleware_module.background_tasks_handler(_title_ctx(tasks_enum, events))

    store["update_title"].assert_awaited_once_with("chat-1", "Generated title")
    assert probe.review.call_count == 0


# --- nearby: the main path keeps its review ---------------------------------------


@pytest.mark.asyncio
async def test_full_context_still_reviews_memory_once_with_the_model(
    middleware_module, memory_module, tasks_enum
):
    events = []
    ctx = _title_ctx(
        tasks_enum,
        events,
        model={"id": "gpt-4o-mini"},
        assistant_message={"role": "assistant", "content": "a finished answer"},
    )
    with (
        _saved_chat(middleware_module),
        _memory_review_probe(memory_module) as probe,
    ):
        await middleware_module.background_tasks_handler(ctx)

    assert probe.review.call_count == 1
    kwargs = probe.review.call_args.kwargs
    assert kwargs["model"] == {"id": "gpt-4o-mini"}
    assert kwargs["assistant_message"] == {"role": "assistant", "content": "a finished answer"}
    assert [m["role"] for m in kwargs["messages"]] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_full_context_review_is_gated_by_the_memory_feature(
    middleware_module, memory_module, tasks_enum
):
    events = []
    ctx = _title_ctx(
        tasks_enum,
        events,
        memory_feature=False,
        model={"id": "gpt-4o-mini"},
        assistant_message={"role": "assistant", "content": "a finished answer"},
    )
    with (
        _saved_chat(middleware_module),
        _memory_review_probe(memory_module) as probe,
    ):
        await middleware_module.background_tasks_handler(ctx)

    assert probe.review.call_count == 0


# --- nearby: the review accepts a missing model -----------------------------------


def _review_inputs(**overrides):
    return {
        "request": SimpleNamespace(),
        "user": SimpleNamespace(id="alice", role="admin"),
        "model": None,
        "metadata": {"features": {"memory": True}},
        "form_data": {"model": "from-form"},
        "assistant_message": {"role": "assistant", "content": "some answer"},
        "messages": [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "some answer"},
        ],
        **overrides,
    }


def test_model_allows_memory_treats_a_missing_model_as_allowed(memory_module):
    assert memory_module.model_allows_memory(None) is True
    assert memory_module.model_allows_memory({}) is True
    assert memory_module.model_allows_memory({"info": {"meta": {"capabilities": {}}}}) is True
    assert (
        memory_module.model_allows_memory({"info": {"meta": {"capabilities": {"memory": False}}}})
        is False
    )


@pytest.mark.asyncio
async def test_review_accepts_a_missing_model(memory_module):
    with _memory_review_probe(memory_module) as probe:
        await memory_module.review_memory_after_turn(**_review_inputs())

    assert probe.review.call_count == 1
    assert probe.review.call_args.kwargs["model"] is None


@pytest.mark.asyncio
async def test_review_stops_before_any_config_read_without_assistant_text(memory_module):
    """The gate the title path relies on: an empty assistant message ends the review."""
    with _memory_review_probe(memory_module) as probe:
        await memory_module.review_memory_after_turn(**_review_inputs(assistant_message={}))
        await memory_module.review_memory_after_turn(
            **_review_inputs(assistant_message={"role": "assistant", "content": "   "})
        )

    assert probe.config.call_count == 0
    assert probe.review.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,expected_model_id",
    [
        (None, "from-form"),
        ({"id": "from-dict"}, "from-dict"),
    ],
)
async def test_review_resolves_the_model_id_without_a_model_dict(
    memory_module, model, expected_model_id
):
    generate = AsyncMock(return_value=[])
    with (
        patch.object(memory_module.Memories, "get_memories_by_user_id", AsyncMock(return_value=[])),
        patch.object(memory_module, "_generate_memory_operations", generate),
    ):
        await memory_module._review_memory(**_review_inputs(model=model))

    assert generate.await_args.kwargs["model_id"] == expected_model_id


# --- broad: the title context must satisfy every hard read in the handler ---------


def _title_ctx_keys(main_source: str) -> set[str]:
    tree = ast.parse(main_source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if any(isinstance(t, ast.Name) and t.id == "title_ctx" for t in node.targets):
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    raise AssertionError("main.py no longer builds `title_ctx` as a dict literal")


def _hard_ctx_reads(middleware_source: str, function_name: str) -> set[str]:
    tree = ast.parse(middleware_source)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == function_name:
            return {
                sub.slice.value
                for sub in ast.walk(node)
                if isinstance(sub, ast.Subscript)
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "ctx"
                and isinstance(sub.ctx, ast.Load)
                and isinstance(sub.slice, ast.Constant)
            }
    raise AssertionError(f"{function_name} not found in middleware.py")


def test_title_context_supplies_every_key_the_handler_reads_unconditionally(open_webui_backend):
    """Any `ctx[...]` the handler reads without a default must be in the title context."""
    main_src = (open_webui_backend / "open_webui" / "main.py").read_text(encoding="utf-8")
    mw_src = (open_webui_backend / "open_webui" / "utils" / "middleware.py").read_text(
        encoding="utf-8"
    )
    supplied = _title_ctx_keys(main_src)
    required = _hard_ctx_reads(mw_src, "background_tasks_handler")

    missing = required - supplied
    assert not missing, (
        f"background_tasks_handler reads {sorted(missing)} from ctx without a default, but "
        "the initial title task in main.py never supplies them, so every new chat ends the "
        "title task with a KeyError (#30339)"
    )


def test_title_context_carries_the_keys_the_handler_unpacks(open_webui_backend):
    """The six keys the handler binds up front are the contract the title caller was written to."""
    main_src = (open_webui_backend / "open_webui" / "main.py").read_text(encoding="utf-8")
    assert {"request", "form_data", "user", "metadata", "tasks", "event_emitter"} <= (
        _title_ctx_keys(main_src)
    )


def test_initial_title_failures_are_logged_with_their_traceback(open_webui_backend):
    """PR #30106: the title task logs `log.exception`, which is what exposed #30339."""
    main_src = (open_webui_backend / "open_webui" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(main_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_initial_title_generation":
            calls = {
                c.func.attr
                for c in ast.walk(node)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            }
            assert "exception" in calls, (
                "run_initial_title_generation swallows failures below error level again, so a "
                "broken title path goes unnoticed (#29533, #30106)"
            )
            return
    raise AssertionError("run_initial_title_generation not found in main.py")
