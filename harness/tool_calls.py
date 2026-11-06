"""Have the scripted model call one tool and read what the tool returned.

`run_tool(client, upstream, name, arguments)` scripts the call and a closing reply, sends a
message the way the web client does (so the builtin tools are offered) and returns the tool
result Open WebUI sent back to the provider on the follow-up request. `offered_tools` sends one
plain message and returns the names of the tools the provider was offered with it. Each prompt
carries a fresh tag that its replies and the requests read back are tied to.

Open WebUI hands the model the attached knowledge, the skill list and recalled memories as calls
of its own, an assistant tool call and its tool result ahead of the model's first reply (since dev
003c9e085). `without_context_calls(messages)` drops those pairs from a provider payload, so a test
reads the conversation and the tool results the model's own calls got back;
`without_hidden_calls(output)` drops them, and preloaded skills, from a reply's stored output.
"""

from __future__ import annotations

import uuid

import httpx

from harness import upstream as reply
from harness.chat import ask
from harness.upstream import MockUpstream

CONTEXT_TOOLS = frozenset({"list_attached_knowledge", "list_available_skills", "recall_memories"})


def run_tool(
    client: httpx.Client, upstream: MockUpstream, name: str, arguments: dict, **options
) -> str:
    prompt = _tagged(f"use {name}")
    answering = reply.answering(prompt)
    upstream.queue(
        reply.tool_call(name, arguments, match=answering), reply.text("done", match=answering)
    )
    ask(client, prompt, **options)
    requests = _requests_answering(upstream, prompt)
    sent_back = requests[-1]["messages"] if requests else []
    results = [entry["content"] for entry in sent_back if entry["role"] == "tool"]
    assert results, f"{name} never ran; the provider was sent {sent_back}"
    return results[-1]


def offered_tools(client: httpx.Client, upstream: MockUpstream, **options) -> set[str]:
    prompt = _tagged("which tools are there?")
    upstream.queue(reply.text("done", match=reply.answering(prompt)))
    ask(client, prompt, **options)
    requests = _requests_answering(upstream, prompt)
    assert requests, f"the provider never received {prompt!r}"
    offered = requests[0].get("tools") or []
    return {tool["function"]["name"] for tool in offered}


def without_context_calls(messages: list[dict]) -> list[dict]:
    """`messages` without the tool calls Open WebUI made itself to give the model its context."""
    context_ids = {
        call["id"]
        for entry in messages
        for call in entry.get("tool_calls") or []
        if call["function"]["name"] in CONTEXT_TOOLS
    }
    return [entry for entry in messages if not _is_context_entry(entry, context_ids)]


def without_hidden_calls(output: list[dict]) -> list[dict]:
    """A reply's stored `output` without the calls Open WebUI made itself and keeps out of view."""
    hidden_ids = {item["call_id"] for item in output if item.get("hidden")}
    return [item for item in output if item.get("call_id") not in hidden_ids]


def _is_context_entry(entry: dict, context_ids: set[str]) -> bool:
    if entry["role"] == "tool":
        return entry.get("tool_call_id") in context_ids
    calls = entry.get("tool_calls") or []
    return bool(calls) and all(call["id"] in context_ids for call in calls)


def _tagged(prompt: str) -> str:
    return f"{prompt} ({uuid.uuid4().hex[:8]})"


def _requests_answering(upstream: MockUpstream, prompt: str) -> list[dict]:
    answering = reply.answering(prompt)
    return [request for request in upstream.chat_requests() if answering(request)]
