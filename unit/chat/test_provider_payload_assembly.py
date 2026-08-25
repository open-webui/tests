"""Regression tests for the payload Open WebUI hands the model provider (0.11.1).

Ten separate 0.11.1 fixes all land on the same object: the request body that
finally reaches the provider.

- Memory context was assembled in whatever order the rows arrived in, so the
  system prompt changed byte-for-byte between turns and prefix caching missed
  every time (ff74bfa6a1 / d22bb6703f, #28292). Each rendered section is now
  sorted, and every memory sort gains `memory.id` as a final tiebreak.
- `merge_model_params` deep-merges `custom_params` (11739a2de8, #28241). The
  old dict splat let one custom param on a model discard every custom param in
  the global defaults, and mutated the shared config object besides.
- `include_usage` moved out of the Anthropic passthrough into
  `chat_completion`, so automations, timers, sub-agents and channel chats get
  token counts too (PR27661, fcc130c9b).
- `process_messages_with_output` now strips the same bookkeeping keys the
  removed `strip_compaction_fields` did, so `id`/`files`/`usage`/`model` stop
  leaking into the provider payload (a32a17965c).
- `get_reasoning_format` keys on `owned_by == 'ollama'` and returns the native
  `'thinking'` format instead of pasting `<think>` tags into content
  (3258330729).
- `get_reasoning_details` reads provider-nested reasoning, and reasoning that
  cannot be replayed without a signature is dropped rather than sent and
  rejected (b6dc70c93b).
- `_usage_token_count` stops adding llama.cpp's `cache_n` on top of a
  `prompt_tokens` that already includes it, which had a 39k chat reading as 77k
  and compacting at half the configured limit (0b27fa5e87, #28590).
- `add_file_context` keeps chat attachments, which carry an id and no url and
  were being filtered out entirely (PR28788, 2e7df5467).
- `_generate_summary` compacts with the chat's own model instead of falling
  through to the configured task model (5093a99389, #27603).
- `MESSAGE_REPLAY_KEYS` carries `model`, so a message's own model survives the
  reload and the previous model's reasoning is not replayed to a provider that
  rejects it (c4b3e6840f, #28240).

Discriminates: passes on v0.11.1, fails on v0.11.0 (unsorted memory sections,
clobbered custom_params, only `output` stripped from replayed messages, no
native Ollama thinking format, unsignable reasoning replayed, cache_n
double-counted, chat attachments dropped, task model used for compaction, and
the per-message model lost on reload).
"""

from __future__ import annotations

import ast
import itertools
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.regression


# -----------------------------------------------------------------------------
# Module fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def memory_module(owui_module) -> ModuleType:
    return owui_module("open_webui.utils.memory")


@pytest.fixture(scope="session")
def memories_router_module(owui_module) -> ModuleType:
    return owui_module("open_webui.routers.memories")


@pytest.fixture(scope="session")
def middleware_module(owui_module) -> ModuleType:
    return owui_module("open_webui.utils.middleware")


@pytest.fixture(scope="session")
def compaction_module(owui_module) -> ModuleType:
    return owui_module("open_webui.utils.context_compaction")


@pytest.fixture(scope="session")
def chat_module(owui_module) -> ModuleType:
    return owui_module("open_webui.utils.chat")


@pytest.fixture(scope="session")
def chat_completion_source(misc_module: ModuleType) -> str:
    """Source text of `main.chat_completion`.

    main.py builds the app at import time, so it is read rather than imported.
    """
    main_py = Path(misc_module.__file__).resolve().parent.parent / "main.py"
    tree = ast.parse(main_py.read_text(encoding="utf-8"))
    handler = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "chat_completion"
    )
    return ast.get_source_segment(main_py.read_text(encoding="utf-8"), handler) or ""


# -----------------------------------------------------------------------------
# 41. Deterministic memory context (#28292)
# -----------------------------------------------------------------------------


@dataclass
class _Memory:
    id: str
    content: str
    path: str | None = None
    type: str = "user"
    updated_at: int = 100


class _NoConfig:
    @staticmethod
    async def get_many(*keys: str) -> dict:
        return {}


def _memories_stub(rows: list[_Memory], real_memories):
    class _Memories:
        normalize_memory_type = real_memories.normalize_memory_type

        @staticmethod
        async def get_memories_by_user_id(user_id: str) -> list[_Memory]:
            return list(rows)

    return _Memories


async def _render_memory_context(
    memory_module: ModuleType,
    memories_router_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[_Memory],
) -> str:
    monkeypatch.setattr(memory_module, "Config", _NoConfig)
    monkeypatch.setattr(memory_module, "Memories", _memories_stub(rows, memory_module.Memories))
    # The only I/O here: the vector-store query. Nothing else in this path talks out.
    monkeypatch.setattr(
        memories_router_module,
        "query_memory",
        lambda *a, **kw: _async_value(SimpleNamespace(documents=None, metadatas=None, ids=None)),
    )

    form_data = {"messages": [{"role": "user", "content": "what did I say about the project?"}]}
    result = await memory_module.add_memory_context(
        SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())),
        form_data,
        SimpleNamespace(id="u1", name="Tester", email="t@example.com", role="user"),
        {},
    )
    system = result["messages"][0]
    assert system["role"] == "system", result["messages"]
    return system["content"]


async def _async_value(value):
    return value


@pytest.mark.asyncio
async def test_memory_context_is_byte_identical_for_any_row_order(
    memory_module: ModuleType,
    memories_router_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same memories, different arrival order, same bytes. Anything else
    invalidates the provider's prefix cache on every turn (#28292)."""
    rows = [
        _Memory(id="m1", content="Zeta note", path="work/project"),
        _Memory(id="m2", content="alpha note", path="work/project"),
        _Memory(id="m3", content="Beta note", path="work/project"),
    ]

    rendered = set()
    for permutation in itertools.permutations(rows):
        rendered.add(
            await _render_memory_context(
                memory_module, memories_router_module, monkeypatch, list(permutation)
            )
        )

    assert len(rendered) == 1, (
        f"memory context differs by row order, so the cached prefix is lost every "
        f"turn (#28292); got {len(rendered)} distinct renderings: {sorted(rendered)}"
    )


@pytest.mark.asyncio
async def test_memory_context_section_order_is_case_insensitive_alphabetical(
    memory_module: ModuleType,
    memories_router_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _Memory(id="m1", content="Zeta note", path="work/project"),
        _Memory(id="m2", content="alpha note", path="work/project"),
        _Memory(id="m3", content="Beta note", path="work/project"),
    ]

    content = await _render_memory_context(
        memory_module, memories_router_module, monkeypatch, rows
    )

    entries = [line[2:] for line in content.splitlines() if line.startswith("- ")]
    assert entries == [
        "work/project: alpha note",
        "work/project: Beta note",
        "work/project: Zeta note",
    ], content


@pytest.mark.asyncio
async def test_memory_context_still_carries_every_memory(
    memory_module: ModuleType,
    memories_router_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sorting must not drop anything: all three still reach the model."""
    rows = [
        _Memory(id="m1", content="Zeta note", path="work/project"),
        _Memory(id="m2", content="alpha note", path="work/project"),
        _Memory(id="m3", content="Beta note", path="other"),
    ]

    content = await _render_memory_context(
        memory_module, memories_router_module, monkeypatch, rows
    )

    for row in rows:
        assert row.content in content, content
    assert memory_module.MEMORY_CONTEXT_OPEN in content


@pytest.mark.asyncio
async def test_no_memories_leaves_the_payload_untouched(
    memory_module: ModuleType,
    memories_router_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory_module, "Config", _NoConfig)
    monkeypatch.setattr(memory_module, "Memories", _memories_stub([], memory_module.Memories))
    monkeypatch.setattr(
        memories_router_module,
        "query_memory",
        lambda *a, **kw: _async_value(SimpleNamespace(documents=None, metadatas=None, ids=None)),
    )

    form_data = {"messages": [{"role": "user", "content": "hello"}]}
    result = await memory_module.add_memory_context(
        SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())),
        form_data,
        SimpleNamespace(id="u1", name="Tester", email="t@example.com", role="user"),
        {},
    )

    assert result["messages"] == [{"role": "user", "content": "hello"}]


def test_every_memory_sort_key_breaks_ties_on_id(memory_module: ModuleType) -> None:
    """Broad: ordering that stops at `updated_at` is order-dependent again the
    moment two rows share a timestamp."""
    source = Path(memory_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    sort_keys = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "sort_key"
    ]
    assert sort_keys, "no sort_key helpers found in utils/memory.py"

    for node in sort_keys:
        segment = ast.get_source_segment(source, node) or ""
        assert "memory.id" in segment, f"sort_key without an id tiebreak:\n{segment}"


# -----------------------------------------------------------------------------
# 42. Custom model params deep-merge (#28241)
# -----------------------------------------------------------------------------


def test_custom_params_from_defaults_survive_a_model_override(misc_module: ModuleType) -> None:
    """The exact #28241 scenario: one custom param saved on a model wiped every
    custom param configured globally."""
    defaults = {"temperature": 0.2, "custom_params": {"top_k": 40, "repeat_penalty": 1.1}}
    model_params = {"custom_params": {"top_k": 10}}

    merged = misc_module.merge_model_params(defaults, model_params)

    assert merged["custom_params"] == {"top_k": 10, "repeat_penalty": 1.1}


def test_request_body_custom_param_wins_over_the_saved_model_setting(
    misc_module: ModuleType,
) -> None:
    model_params = {"custom_params": {"top_k": 10, "mirostat": 2}}
    request_params = {"custom_params": {"top_k": 99}}

    merged = misc_module.merge_model_params(model_params, request_params)

    assert merged["custom_params"] == {"top_k": 99, "mirostat": 2}


def test_merge_does_not_mutate_either_side(misc_module: ModuleType) -> None:
    defaults = {"custom_params": {"top_k": 40}}
    override = {"custom_params": {"top_k": 10}}

    misc_module.merge_model_params(defaults, override)

    assert defaults == {"custom_params": {"top_k": 40}}
    assert override == {"custom_params": {"top_k": 10}}


@pytest.mark.parametrize(
    ("base", "override", "expected"),
    [
        ({}, {}, {}),
        ({"custom_params": {"a": 1}}, {}, {"custom_params": {"a": 1}}),
        ({}, {"custom_params": {"a": 1}}, {"custom_params": {"a": 1}}),
        ({"custom_params": {"a": 1}}, {"custom_params": None}, {"custom_params": {"a": 1}}),
        # a non-dict on either side is passed through by the plain splat, not merged
        ({"custom_params": "junk"}, {"custom_params": {"a": 1}}, {"custom_params": {"a": 1}}),
        ({"custom_params": {"a": 1}}, {"custom_params": "junk"}, {"custom_params": "junk"}),
    ],
)
def test_merge_edge_shapes(misc_module: ModuleType, base, override, expected) -> None:
    assert misc_module.merge_model_params(base, override) == expected


def test_chat_completion_merges_params_and_copies_the_global_defaults(
    chat_completion_source: str,
) -> None:
    """The call site: a plain splat here reopens #28241, and mutating the shared
    config object poisons every later request."""
    assert "merge_model_params(" in chat_completion_source, (
        "chat_completion no longer routes model params through merge_model_params, "
        "so one custom param on a model discards the global custom params (#28241)"
    )

    defaults_line = next(
        line
        for line in chat_completion_source.splitlines()
        if line.strip().startswith("default_model_params =")
    )
    assert "deepcopy" in defaults_line, (
        f"global default params are used without a copy, so the config object is "
        f"mutated in place: {defaults_line.strip()}"
    )


# -----------------------------------------------------------------------------
# 63. include_usage for every caller (PR27661)
# -----------------------------------------------------------------------------


def test_chat_completion_requests_usage_for_every_caller(chat_completion_source: str) -> None:
    """Asking for usage only inside the Anthropic passthrough left automations,
    timers, sub-agents and channel chats with no token counts at all."""
    assert "'include_usage': True" in chat_completion_source, (
        "chat_completion does not set stream_options.include_usage, so every "
        "non-passthrough caller loses its token counts (PR27661)"
    )
    assert "stream_options" in chat_completion_source


def test_usage_capability_is_read_before_the_fallback_rebinds_model(
    chat_completion_source: str,
) -> None:
    """A custom model falling back to the default must not be asked about the
    fallback's usage capability."""
    capabilities_at = chat_completion_source.find("model_capabilities =")
    rebind_at = chat_completion_source.find("model = fallback_model")

    assert capabilities_at != -1, "model_capabilities is no longer read in chat_completion"
    assert rebind_at != -1, "the custom-model fallback rebind is gone; retarget this test"
    assert capabilities_at < rebind_at, (
        "model_capabilities is read after the fallback rebinds model, so the "
        "custom model's usage capability is read off the fallback instead"
    )


# -----------------------------------------------------------------------------
# 86. Bookkeeping keys stripped from the replayed history (a32a17965c)
# -----------------------------------------------------------------------------

_BOOKKEEPING_KEYS = ("id", "files", "output", "contextSummary", "context_summary", "usage")


def test_replayed_message_carries_no_bookkeeping_keys(middleware_module: ModuleType) -> None:
    """Two half-strippers used to run over the same history: one dropped
    `output`, the other dropped the compaction fields, and whichever ran alone
    left the rest in the provider payload."""
    message = {
        "role": "assistant",
        "content": "hi",
        "id": "msg-1",
        "files": [{"id": "f1"}],
        "output": None,
        "contextSummary": "summary so far",
        "context_summary": "summary so far",
        "usage": {"prompt_tokens": 10},
        "model": "gpt-4o",
    }

    processed = middleware_module.process_messages_with_output([message])

    assert processed == [{"role": "assistant", "content": "hi"}], processed


@pytest.mark.parametrize("key", [*_BOOKKEEPING_KEYS, "model"])
def test_each_bookkeeping_key_is_stripped_individually(
    middleware_module: ModuleType, key: str
) -> None:
    """Broad: every key in the set, one at a time, so a partial re-introduction
    is caught."""
    message = {"role": "user", "content": "hello", key: "value"}

    processed = middleware_module.process_messages_with_output([message])

    assert key not in processed[0], processed


def test_real_payload_fields_are_preserved(middleware_module: ModuleType) -> None:
    """Nearby: stripping must not over-correct onto fields the provider needs."""
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        ],
        "name": "helper",
    }

    processed = middleware_module.process_messages_with_output([message])

    assert processed == [message]


def test_processing_does_not_mutate_the_caller_list(middleware_module: ModuleType) -> None:
    message = {"role": "user", "content": "hello", "id": "m1", "usage": {"prompt_tokens": 1}}

    middleware_module.process_messages_with_output([message])

    assert message["id"] == "m1"
    assert message["usage"] == {"prompt_tokens": 1}


# -----------------------------------------------------------------------------
# 90. Native Ollama reasoning (3258330729)
# -----------------------------------------------------------------------------


def test_ollama_model_gets_the_native_thinking_format(middleware_module: ModuleType) -> None:
    """Ollama models are tagged by `owned_by`, never `provider`, so the old key
    matched nothing and reasoning was pasted into content as <think> tags."""
    assert middleware_module.get_reasoning_format({"owned_by": "ollama"}) == "thinking"


def test_ollama_reasoning_becomes_a_thinking_field(misc_module: ModuleType) -> None:
    output = [
        {"type": "reasoning", "summary": [{"type": "output_text", "text": "step one"}]},
        {"type": "message", "content": [{"type": "output_text", "text": "answer"}]},
    ]

    messages = misc_module.convert_output_to_messages(output, reasoning_format="thinking")

    assistant = next(m for m in messages if m["role"] == "assistant")
    assert assistant.get("thinking") == "step one", assistant
    assert "<think>" not in assistant.get("content", "")
    assert "reasoning_content" not in assistant


def test_llama_cpp_still_uses_reasoning_content(middleware_module: ModuleType) -> None:
    assert middleware_module.get_reasoning_format({"provider": "llama.cpp"}) == "reasoning_content"


def test_strict_providers_still_get_no_reasoning(middleware_module: ModuleType) -> None:
    assert middleware_module.get_reasoning_format({}) is None
    assert middleware_module.get_reasoning_format({"owned_by": "openai"}) is None


def test_think_tags_format_still_wraps_content(misc_module: ModuleType) -> None:
    """Nearby: the legacy tag replay path is untouched."""
    output = [{"type": "reasoning", "summary": [{"type": "output_text", "text": "step one"}]}]

    messages = misc_module.convert_output_to_messages(output, reasoning_format="think_tags")

    assert "<think>step one</think>" in messages[0]["content"]


# -----------------------------------------------------------------------------
# 110. Reasoning carried between turns (b6dc70c93b)
# -----------------------------------------------------------------------------


def test_unsignable_anthropic_reasoning_is_not_replayed(misc_module: ModuleType) -> None:
    """Anthropic rejects a thinking block replayed without its signature, so the
    whole request fails. Drop the block instead of sending it."""
    output = [
        {
            "type": "reasoning",
            "summary": [{"type": "output_text", "text": "step one"}],
            "reasoning_details": [{"format": "anthropic-claude-v1", "text": "step one"}],
        }
    ]

    messages = misc_module.convert_output_to_messages(output, raw=True)

    assert all("reasoning_details" not in message for message in messages), messages


def test_signed_anthropic_reasoning_is_still_replayed(misc_module: ModuleType) -> None:
    detail = {"format": "anthropic-claude-v1", "text": "step one", "signature": "sig"}
    output = [
        {
            "type": "reasoning",
            "summary": [{"type": "output_text", "text": "step one"}],
            "reasoning_details": [detail],
        }
    ]

    messages = misc_module.convert_output_to_messages(output, raw=True)

    assert messages[0]["reasoning_details"] == [detail], messages


def test_other_providers_reasoning_details_are_untouched(misc_module: ModuleType) -> None:
    detail = {"format": "openai-responses-v1", "text": "step one"}
    output = [
        {
            "type": "reasoning",
            "summary": [{"type": "output_text", "text": "step one"}],
            "reasoning_details": [detail],
        }
    ]

    messages = misc_module.convert_output_to_messages(output, raw=True)

    assert messages[0]["reasoning_details"] == [detail], messages


def test_provider_nested_reasoning_details_are_found(misc_module: ModuleType) -> None:
    """LiteLLM-style providers hide the details a level down; unrecognised there,
    the reasoning was silently lost between turns."""
    nested = [{"format": "anthropic-claude-v1", "signature": "sig"}]

    payload = {"provider_specific_fields": {"reasoning_details": nested}}

    assert misc_module.get_reasoning_details(payload) == nested


def test_top_level_reasoning_details_win(misc_module: ModuleType) -> None:
    top = [{"format": "openai-responses-v1"}]
    nested = [{"format": "anthropic-claude-v1", "signature": "sig"}]

    payload = {"reasoning_details": top, "provider_specific_fields": {"reasoning_details": nested}}

    assert misc_module.get_reasoning_details(payload) == top


@pytest.mark.parametrize(
    "payload",
    [{}, None, "not a dict", {"provider_specific_fields": None}, {"provider_specific_fields": "x"}],
)
def test_missing_reasoning_details_is_none(misc_module: ModuleType, payload) -> None:
    assert misc_module.get_reasoning_details(payload) is None


# -----------------------------------------------------------------------------
# 129. llama.cpp cached input counted once (#28590)
# -----------------------------------------------------------------------------


def test_llama_cpp_cached_prompt_is_not_counted_twice(compaction_module: ModuleType) -> None:
    """llama.cpp reports `cache_n` as the already-included cached slice of
    `prompt_tokens`. Adding it on top turned a 39k chat into 77k and compacted
    at half the configured limit (#28590)."""
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "hello",
            "usage": {"prompt_tokens": 39000, "completion_tokens": 500, "cache_n": 38000},
        },
    ]

    assert (
        compaction_module._exceeds_token_threshold(messages, "", None, 70000) is False
    ), "39.5k of real context read as over a 70k threshold (#28590)"


def test_llama_cpp_prompt_n_plus_cache_n_is_the_prompt_total(
    compaction_module: ModuleType,
) -> None:
    """Native llama.cpp reports the uncached and cached halves separately; those
    two do add up."""
    messages = [
        {"role": "assistant", "content": "x", "usage": {"prompt_n": 40000, "cache_n": 38000}},
    ]

    assert compaction_module._exceeds_token_threshold(messages, "", None, 70000) is True


def test_a_genuinely_oversized_chat_still_compacts(compaction_module: ModuleType) -> None:
    messages = [
        {
            "role": "assistant",
            "content": "x",
            "usage": {"prompt_tokens": 90000, "completion_tokens": 500},
        },
    ]

    assert compaction_module._exceeds_token_threshold(messages, "", None, 70000) is True


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"prompt_tokens": 100, "completion_tokens": 20}, 120),
        ({"prompt_eval_count": 100, "eval_count": 20}, 120),
        ({"input_tokens": 100, "output_tokens": 20}, 120),
        ({"prompt_n": 60, "cache_n": 40, "predicted_n": 20}, 120),
        ({"prompt_tokens": 100, "cache_n": 40, "completion_tokens": 20}, 120),
    ],
)
def test_usage_token_count_per_provider_dialect(
    compaction_module: ModuleType, usage: dict, expected: int
) -> None:
    """Broad: every dialect the counter claims to read, counted once each.

    Pinned through the threshold comparison rather than the private counter, so
    the count is measured the way the compaction decision measures it.
    """
    messages = [{"role": "assistant", "content": "x", "usage": usage}]

    assert compaction_module._exceeds_token_threshold(messages, "", None, expected - 1) is True
    assert compaction_module._exceeds_token_threshold(messages, "", None, expected) is False


# -----------------------------------------------------------------------------
# 135. Chat attachments reach the model (PR28788)
# -----------------------------------------------------------------------------


def _chat_with_files(files: list[dict]):
    return SimpleNamespace(
        chat={
            "history": {
                "currentId": "u1",
                "messages": {"u1": {"id": "u1", "role": "user", "content": "look", "files": files}},
            }
        }
    )


def _patch_chat(monkeypatch: pytest.MonkeyPatch, middleware_module: ModuleType, chat) -> None:
    class _Chats:
        @staticmethod
        async def get_chat_by_id_and_user_id(chat_id: str, user_id: str):
            return chat

    monkeypatch.setattr(middleware_module, "Chats", _Chats)


@pytest.mark.asyncio
async def test_attached_chat_reaches_the_model(
    middleware_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chat attached to a message is addressed by id and carries no url, so
    the url filter dropped it and the model never learned it was there."""
    _patch_chat(
        monkeypatch,
        middleware_module,
        _chat_with_files([{"type": "chat", "id": "chat-abc", "name": "Design notes"}]),
    )

    messages = [{"role": "user", "content": "look"}]
    result = await middleware_module.add_file_context(
        messages, "c1f2e3d4-0000-0000-0000-000000000000", SimpleNamespace(id="u1")
    )

    assert 'id="chat-abc"' in result[0]["content"], result
    assert 'type="chat"' in result[0]["content"]


@pytest.mark.asyncio
async def test_file_tag_always_carries_an_id(
    middleware_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_chat(
        monkeypatch,
        middleware_module,
        _chat_with_files(
            [{"type": "file", "id": "f1", "url": "/api/v1/files/f1", "name": "spec.pdf"}]
        ),
    )

    messages = [{"role": "user", "content": "look"}]
    result = await middleware_module.add_file_context(
        messages, "c1f2e3d4-0000-0000-0000-000000000000", SimpleNamespace(id="u1")
    )

    assert 'id="f1"' in result[0]["content"], result
    assert 'url="/api/v1/files/f1"' in result[0]["content"]


@pytest.mark.asyncio
async def test_inline_data_url_attachments_stay_out(
    middleware_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nearby: a base64 data: url is already in the payload as an image part and
    must not be repeated as a file tag."""
    _patch_chat(
        monkeypatch,
        middleware_module,
        _chat_with_files([{"type": "image", "id": "i1", "url": "data:image/png;base64,AAA"}]),
    )

    messages = [{"role": "user", "content": "look"}]
    result = await middleware_module.add_file_context(
        messages, "c1f2e3d4-0000-0000-0000-000000000000", SimpleNamespace(id="u1")
    )

    assert result[0]["content"] == "look"


@pytest.mark.asyncio
async def test_message_without_attachments_is_untouched(
    middleware_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_chat(monkeypatch, middleware_module, _chat_with_files([]))

    messages = [{"role": "user", "content": "look"}]
    result = await middleware_module.add_file_context(
        messages, "c1f2e3d4-0000-0000-0000-000000000000", SimpleNamespace(id="u1")
    )

    assert result[0]["content"] == "look"


# -----------------------------------------------------------------------------
# 151. Compaction uses the chat's own model (#27603)
# -----------------------------------------------------------------------------


class _CompactionConfig:
    values: dict = {}

    @classmethod
    async def get_many(cls, *keys: str) -> dict:
        return {key: cls.values.get(key) for key in keys}


@pytest.mark.asyncio
async def test_compaction_summarises_with_the_chats_own_model(
    compaction_module: ModuleType, chat_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured task model used to hijack compaction, so a long chat was
    shortened by an unrelated (often much weaker) model (#27603)."""
    _CompactionConfig.values = {
        "task.model.default": "task-model",
        "task.model.external": "task-model",
        "chat.context_compaction.model": None,
        "task.model.params": None,
    }
    monkeypatch.setattr(compaction_module, "Config", _CompactionConfig)

    models = {
        "chat-model": {"id": "chat-model", "connection_type": "local", "owned_by": "ollama"},
        "task-model": {"id": "task-model", "connection_type": "local", "owned_by": "ollama"},
    }
    sent: dict = {}

    async def fake_generate_chat_completion(request, form_data, user, **kwargs):
        sent.update(form_data)
        return {"choices": [{"message": {"content": "summary"}}]}

    monkeypatch.setattr(chat_module, "generate_chat_completion", fake_generate_chat_completion)

    summary = await compaction_module._generate_summary(
        SimpleNamespace(state=SimpleNamespace(metadata={})),
        {"id": "u1", "name": "Tester", "email": "t@example.com"},
        "chat-model",
        models,
        [{"role": "user", "content": "old"}],
        [{"role": "user", "content": "new"}],
        None,
        "",
    )

    assert summary == "summary"
    assert sent["model"] == "chat-model", (
        f"context compaction ran on {sent['model']!r} instead of the chat's own "
        f"model; the task model must not hijack it (#27603)"
    )


@pytest.mark.asyncio
async def test_explicit_compaction_model_is_still_honoured(
    compaction_module: ModuleType, chat_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _CompactionConfig.values = {
        "task.model.default": "task-model",
        "task.model.external": "task-model",
        "chat.context_compaction.model": "compaction-model",
        "task.model.params": None,
    }
    monkeypatch.setattr(compaction_module, "Config", _CompactionConfig)

    models = {
        "chat-model": {"id": "chat-model", "connection_type": "local"},
        "task-model": {"id": "task-model", "connection_type": "local"},
        "compaction-model": {"id": "compaction-model", "connection_type": "local"},
    }
    sent: dict = {}

    async def fake_generate_chat_completion(request, form_data, user, **kwargs):
        sent.update(form_data)
        return {"choices": [{"message": {"content": "summary"}}]}

    monkeypatch.setattr(chat_module, "generate_chat_completion", fake_generate_chat_completion)

    await compaction_module._generate_summary(
        SimpleNamespace(state=SimpleNamespace(metadata={})),
        {"id": "u1", "name": "Tester", "email": "t@example.com"},
        "chat-model",
        models,
        [{"role": "user", "content": "old"}],
        [{"role": "user", "content": "new"}],
        None,
        "",
    )

    assert sent["model"] == "compaction-model"


@pytest.mark.asyncio
async def test_unknown_compaction_model_falls_back_to_the_chat_model(
    compaction_module: ModuleType, chat_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _CompactionConfig.values = {
        "task.model.default": None,
        "task.model.external": None,
        "chat.context_compaction.model": "deleted-model",
        "task.model.params": None,
    }
    monkeypatch.setattr(compaction_module, "Config", _CompactionConfig)

    models = {"chat-model": {"id": "chat-model", "connection_type": "local"}}
    sent: dict = {}

    async def fake_generate_chat_completion(request, form_data, user, **kwargs):
        sent.update(form_data)
        return {"choices": [{"message": {"content": "summary"}}]}

    monkeypatch.setattr(chat_module, "generate_chat_completion", fake_generate_chat_completion)

    await compaction_module._generate_summary(
        SimpleNamespace(state=SimpleNamespace(metadata={})),
        {"id": "u1", "name": "Tester", "email": "t@example.com"},
        "chat-model",
        models,
        [{"role": "user", "content": "old"}],
        [{"role": "user", "content": "new"}],
        None,
        "",
    )

    assert sent["model"] == "chat-model"


@pytest.mark.asyncio
async def test_compaction_without_any_usable_model_raises(
    compaction_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _CompactionConfig.values = {"chat.context_compaction.model": None, "task.model.params": None}
    monkeypatch.setattr(compaction_module, "Config", _CompactionConfig)

    with pytest.raises(ValueError):
        await compaction_module._generate_summary(
            SimpleNamespace(state=SimpleNamespace(metadata={})),
            {"id": "u1", "name": "Tester", "email": "t@example.com"},
            "gone-model",
            {},
            [{"role": "user", "content": "old"}],
            [{"role": "user", "content": "new"}],
            None,
            "",
        )


# -----------------------------------------------------------------------------
# 199. Per-message model survives the reload (#28240)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reloaded_message_keeps_its_own_model(
    middleware_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the per-message model, the previous reasoning model's stored
    reasoning is replayed to whatever provider comes next, which rejects every
    later request (#28240)."""

    class _Chats:
        @staticmethod
        async def get_messages_map_by_chat_id(chat_id: str):
            return {
                "a1": {
                    "id": "a1",
                    "role": "assistant",
                    "content": "hello",
                    "model": "o3-mini",
                    "output": [],
                    "usage": {"prompt_tokens": 5},
                }
            }

    monkeypatch.setattr(middleware_module, "Chats", _Chats)

    messages = await middleware_module.load_messages_from_db("chat-1", "a1")

    assert messages[0].get("model") == "o3-mini", messages


@pytest.mark.asyncio
async def test_reload_still_drops_non_replay_fields(
    middleware_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nearby: widening the key set must not start replaying bookkeeping."""

    class _Chats:
        @staticmethod
        async def get_messages_map_by_chat_id(chat_id: str):
            return {
                "a1": {
                    "id": "a1",
                    "role": "assistant",
                    "content": "hello",
                    "model": "o3-mini",
                    "timestamp": 1,
                    "parentId": None,
                    "childrenIds": [],
                    "statusHistory": [{"description": "thinking"}],
                }
            }

    monkeypatch.setattr(middleware_module, "Chats", _Chats)

    messages = await middleware_module.load_messages_from_db("chat-1", "a1")

    for key in ("timestamp", "childrenIds", "statusHistory"):
        assert key not in messages[0], messages


@pytest.mark.asyncio
async def test_reload_of_an_unknown_chat_is_none(
    middleware_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Chats:
        @staticmethod
        async def get_messages_map_by_chat_id(chat_id: str):
            return None

    monkeypatch.setattr(middleware_module, "Chats", _Chats)

    assert await middleware_module.load_messages_from_db("chat-1", "a1") is None


def test_replayed_reasoning_is_stripped_when_the_model_changed(
    middleware_module: ModuleType,
) -> None:
    """Broad: knowing the per-message model is only useful if the payload path
    actually uses it to drop provider-bound reasoning."""
    output = [
        {"type": "reasoning", "reasoning_details": [{"format": "anthropic-claude-v1"}], "id": "r1"},
        {"type": "message", "content": [{"type": "output_text", "text": "hi"}]},
    ]

    stripped = middleware_module.strip_reasoning_details(output)

    assert all("reasoning_details" not in item for item in stripped), stripped
    assert stripped[0]["id"] == "r1"
    assert stripped[1] == output[1]
