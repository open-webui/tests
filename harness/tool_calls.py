"""Have the scripted model call one tool and read what the tool returned.

`run_tool(client, upstream, name, arguments)` scripts the call and a closing reply, sends a
message the way the web client does (so the builtin tools are offered) and returns the tool
result Open WebUI sent back to the provider on the follow-up request. `offered_tools` sends one
plain message and returns the names of the tools the provider was offered with it. Each prompt
carries a fresh tag that its replies and the requests read back are tied to.

Open WebUI stores the context it hands the model (attached knowledge, the skill list, recalled
memories, preloaded skills) as hidden tool calls of its own at the start of a reply (since dev
003c9e085). `without_hidden_calls(output)` drops them from a reply's stored output, the way
`upstream.conversation` does for a provider payload. A citing tool's result (knowledge, file and
web tools) carries its source list in the citations template after it (since dev 8e2ba2937);
`without_citation_context(result)` gives the tool's own result. With the Skills tool on, the
skill list comes as a `list_available_skills` result and a selected or mentioned skill as a
preloaded `view_skill` result (since dev 6e8a6c446); `skill_list` and `viewed_skills` read them.
"""

from __future__ import annotations

import json
import uuid

import httpx

from harness import upstream as reply
from harness.chat import ask
from harness.upstream import MockUpstream

# Where the default RAG template starts in the source list after a citing tool's result
CITATION_CONTEXT = "\n\n### Task:\n"


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


def skill_list(body: dict) -> str:
    """The `<available_skills>` list handed to the model in a provider payload, or ''."""
    return "\n".join(reply.context_results(body, "list_available_skills"))


def viewed_skills(body: dict) -> dict[str, dict]:
    """The skills `view_skill` returned in a provider payload, preloaded or called, by id."""
    results = [json.loads(result) for result in reply.context_results(body, "view_skill")]
    return {skill["id"]: skill for skill in results if "id" in skill}


def without_citation_context(result: str) -> str:
    """A tool's own result, without the source list a citing tool's result carries after it."""
    own, marker, _ = result.rpartition(CITATION_CONTEXT)
    return own if marker else result


def without_hidden_calls(output: list[dict]) -> list[dict]:
    """A reply's stored `output` without the calls Open WebUI made itself and keeps out of view."""
    hidden_ids = {item["call_id"] for item in output if item.get("hidden")}
    return [item for item in output if item.get("call_id") not in hidden_ids]


def _tagged(prompt: str) -> str:
    return f"{prompt} ({uuid.uuid4().hex[:8]})"


def _requests_answering(upstream: MockUpstream, prompt: str) -> list[dict]:
    answering = reply.answering(prompt)
    return [request for request in upstream.chat_requests() if answering(request)]
