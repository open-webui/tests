"""The parts of the 0.11.1 provider-payload fixes that no HTTP request can reach.

The request bodies themselves are pinned by the integration twin,
integration/chat/test_provider_payload_assembly.py. Three pieces stay here:

- `3258330729`: `get_reasoning_format` keyed Ollama on `provider`, a key Ollama models never
  carry (they are tagged by `owned_by`), and pasted reasoning into content as `<think>` tags.
  Ollama now gets its native `thinking` field. The scripted provider is OpenAI-shaped, so no
  Ollama model exists on the test instance.
- `ff74bfa6a1` (#28292): every memory `sort_key` gained `memory.id` as the final tiebreak, so
  rows sharing a timestamp come out in one order. Ties within a second cannot be staged over
  HTTP without racing the clock, so the tiebreak is audited in the source.
- `fcc130c9b` (PR #27661): `chat_completion` reads the model's `usage` capability before the
  custom-model fallback rebinds `model`, so the fallback's capability is not used instead. The
  fallback needs `ENABLE_CUSTOM_MODEL_FALLBACK`, an environment-only switch, so the order is
  audited.

Discriminates: passes on bbfa876af; fails with `get_reasoning_format` keyed on `provider` again
(Ollama reasoning becomes `<think>` content), with `memory.id` dropped from a `sort_key`, and
with the capability read moved below the fallback rebind.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

REASONING_REPLY = {
    "role": "assistant",
    "output": [
        {"type": "reasoning", "summary": [{"type": "output_text", "text": "step one"}]},
        {"type": "message", "content": [{"type": "output_text", "text": "answer"}]},
    ],
}


@pytest.fixture(scope="session")
def middleware_module(owui_module):
    return owui_module("open_webui.utils.middleware")


def _replayed(middleware_module, model: dict) -> dict:
    """The stored reply as the payload replays it to `model`."""
    [message] = middleware_module.process_messages_with_output(
        messages=[REASONING_REPLY],
        reasoning_format=middleware_module.get_reasoning_format(model),
    )
    return message


def test_ollama_reasoning_is_replayed_in_the_native_thinking_field(middleware_module):
    message = _replayed(middleware_module, {"id": "llama3", "owned_by": "ollama"})

    assert message.get("thinking") == "step one", message
    assert message["content"] == "answer", "reasoning was pasted into the content as <think> tags"


@pytest.mark.parametrize(
    ("model", "field"),
    [({"provider": "llama.cpp"}, "reasoning_content"), ({"owned_by": "openai"}, None)],
    ids=["llama-cpp", "strict-provider"],
)
def test_other_providers_keep_their_reasoning_format(middleware_module, model, field):
    message = _replayed(middleware_module, model)

    reasoning_fields = {"thinking", "reasoning_content"} & set(message)
    assert reasoning_fields == ({field} if field else set()), message
    assert message["content"] == "answer"


def _functions(path: Path, name: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert found, f"no `{name}` in {path.name}; retarget this audit"
    return found


def _reads_id(node: ast.AST) -> bool:
    return any(isinstance(inner, ast.Attribute) and inner.attr == "id" for inner in ast.walk(node))


def test_every_memory_sort_key_breaks_ties_on_the_memory_id(open_webui_backend):
    memory_py = open_webui_backend / "open_webui" / "utils" / "memory.py"

    for sort_key in _functions(memory_py, "sort_key"):
        for returned in [node for node in ast.walk(sort_key) if isinstance(node, ast.Return)]:
            assert isinstance(returned.value, ast.Tuple), ast.unparse(returned)
            assert _reads_id(returned.value.elts[-1]), (
                f"line {returned.lineno}: `{ast.unparse(returned)}` has no id tiebreak, so rows "
                "sharing a timestamp come out in arrival order (#28292)"
            )


def _first_line_assigning(function: ast.AST, target: str, value: str | None = None) -> int:
    lines = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(name, ast.Name) and name.id == target for name in node.targets)
        and (value is None or ast.unparse(node.value) == value)
    ]
    assert lines, f"chat_completion no longer assigns `{target}`; retarget this audit"
    return min(lines)


def test_the_usage_capability_is_read_before_the_fallback_rebinds_the_model(
    open_webui_backend,
):
    [chat_completion] = _functions(open_webui_backend / "open_webui" / "main.py", "chat_completion")

    read_at = _first_line_assigning(chat_completion, "model_capabilities")
    rebound_at = _first_line_assigning(chat_completion, "model", "fallback_model")
    assert read_at < rebound_at, (
        "the usage capability is read after the custom-model fallback rebinds `model`, so the "
        "fallback's capability decides whether token counts are requested (PR #27661)"
    )
