"""A `listener` added as one more OpenAI connection, for replies the scripted provider cannot shape.

`attach(client, listener, model_id, **config)` appends the listener to the admin's OpenAI
connection list with that per-connection config (a `prefix_id`, a `provider`) and registers the
model it serves. Snapshot `OPENAI_CONFIG` with `preserve` first so the list is restored.
`sse(...)` is a whole streamed chat reply in the provider's own wire format: reasoning after the
answer, `reasoning_details`, or a plain JSON error line without the `data:` prefix.
"""

from __future__ import annotations

import json

import httpx

from harness.listener import Answer, Listener, json_answer, text_answer

OPENAI_CONFIG = ("/openai/config", "/openai/config/update")


def attach(client: httpx.Client, listener: Listener, model_id: str, **config) -> None:
    listener.route("GET", "/v1/models", json_answer({"object": "list", "data": [{"id": model_id}]}))
    current = client.get(OPENAI_CONFIG[0])
    current.raise_for_status()
    connections = current.json()
    index = str(len(connections["OPENAI_API_BASE_URLS"]))
    updated = {
        **connections,
        "OPENAI_API_BASE_URLS": [*connections["OPENAI_API_BASE_URLS"], f"{listener.base_url}/v1"],
        "OPENAI_API_KEYS": [*connections["OPENAI_API_KEYS"], "sk-second"],
        "OPENAI_API_CONFIGS": {
            **connections["OPENAI_API_CONFIGS"],
            index: {"enable": True, **config},
        },
    }
    client.post(OPENAI_CONFIG[1], json=updated).raise_for_status()
    listed = client.get("/api/models")
    listed.raise_for_status()
    served = [model["id"] for model in listed.json()["data"] if model["id"].endswith(model_id)]
    assert served, f"the second connection's {model_id} was never registered"


def tool_call_delta(name: str, arguments: dict) -> dict:
    """The delta of a streamed reply that calls one tool."""
    function = {"name": name, "arguments": json.dumps(arguments)}
    return {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": function}]}


def sse(*events: dict | str, finish_reason: str = "stop") -> Answer:
    """A streamed chat reply: a dict is one chunk's delta, a str is written as a raw line."""
    lines = []
    for event in (*events, None):
        if isinstance(event, str):
            lines.append(f"{event}\n\n")
            continue
        choice = {
            "index": 0,
            "delta": event or {},
            "finish_reason": None if event else finish_reason,
        }
        chunk = {"id": "second", "object": "chat.completion.chunk", "choices": [choice]}
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    lines.append("data: [DONE]\n\n")
    return text_answer("".join(lines), content_type="text/event-stream")
