"""Have the scripted model call one tool and read what the tool returned.

`run_tool(client, upstream, name, arguments)` scripts the call and a closing reply, sends a
message the way the web client does (so the builtin tools are offered) and returns the tool
result Open WebUI sent back to the provider on the follow-up request.
"""

from __future__ import annotations

import httpx

from harness import upstream as reply
from harness.chat import ask
from harness.upstream import MockUpstream


def run_tool(
    client: httpx.Client, upstream: MockUpstream, name: str, arguments: dict, **options
) -> str:
    upstream.queue(reply.tool_call(name, arguments), reply.text("done"))
    ask(client, f"use {name}", **options)
    sent_back = upstream.chat_requests()[-1]["messages"]
    results = [entry["content"] for entry in sent_back if entry["role"] == "tool"]
    assert results, f"{name} never ran; the provider was sent {sent_back}"
    return results[-1]
