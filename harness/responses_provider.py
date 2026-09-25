"""A `listener` added as an OpenAI connection that speaks the Responses API (`api_type: responses`).

`connect_responses(client, listener)` attaches it through `harness.second_provider.attach` and
serves `RESPONSES_MODEL`; snapshot `OPENAI_CONFIG` with `preserve` first. `provider.answer(...)`
lines up one answer per `/v1/responses` call, each an event stream built from the helpers here
(`message`, `reasoning`, `function_call`, `completed`, `failed`) or a whole JSON result for a
non-streamed call; `provider.sent()` is what each call carried.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx

from harness.listener import Answer, Listener, ReceivedRequest, text_answer
from harness.second_provider import attach

RESPONSES_MODEL = "responses-model"


def events_stream(*events: dict) -> Answer:
    """An SSE body with one `data:` frame per event, as the Responses API streams them."""
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return text_answer(body, "text/event-stream")


def message(*deltas: str, index: int = 0, item_id: str = "msg_1") -> list[dict]:
    """A message item streamed as `output_text` deltas and closed."""
    item = {"type": "message", "id": item_id, "role": "assistant", "content": []}
    part = {"type": "output_text", "text": ""}
    text = "".join(deltas)
    return [
        {"type": "response.output_item.added", "output_index": index, "item": item},
        {
            "type": "response.content_part.added",
            "output_index": index,
            "content_index": 0,
            "part": part,
        },
        *(
            {
                "type": "response.output_text.delta",
                "output_index": index,
                "content_index": 0,
                "delta": delta,
            }
            for delta in deltas
        ),
        {
            "type": "response.output_text.done",
            "output_index": index,
            "content_index": 0,
            "text": text,
        },
        {
            "type": "response.content_part.done",
            "output_index": index,
            "content_index": 0,
            "part": {**part, "text": text},
        },
    ]


def reasoning(summary: str, text: str = "", index: int = 0) -> list[dict]:
    """A reasoning item with a streamed summary and, when given, a streamed reasoning body."""
    item = {"type": "reasoning", "id": f"rs_{index}", "status": "in_progress", "summary": []}
    summary_part = {"type": "summary_text", "text": ""}
    events = [
        {"type": "response.output_item.added", "output_index": index, "item": item},
        {
            "type": "response.reasoning_summary_part.added",
            "output_index": index,
            "summary_index": 0,
            "part": summary_part,
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "output_index": index,
            "summary_index": 0,
            "delta": summary,
        },
        {
            "type": "response.reasoning_summary_part.done",
            "output_index": index,
            "summary_index": 0,
            "part": {**summary_part, "text": summary},
        },
    ]
    if text:
        events.append(
            {"type": "response.reasoning_text.delta", "output_index": index, "delta": text}
        )
    events.append(
        {"type": "response.reasoning_summary_text.done", "output_index": index, "text": summary}
    )
    return events


def function_call(
    name: str, arguments: dict, call_id: str = "call_1", index: int = 0
) -> list[dict]:
    """A function call item whose arguments stream in two halves."""
    item = {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": "",
        "status": "in_progress",
    }
    encoded = json.dumps(arguments)
    half = len(encoded) // 2
    return [
        {"type": "response.output_item.added", "output_index": index, "item": item},
        *(
            {
                "type": "response.function_call_arguments.delta",
                "output_index": index,
                "delta": piece,
            }
            for piece in (encoded[:half], encoded[half:])
        ),
        {
            "type": "response.function_call_arguments.done",
            "output_index": index,
            "arguments": encoded,
        },
        {
            "type": "response.output_item.done",
            "output_index": index,
            "item": {**item, "arguments": encoded, "status": "completed"},
        },
    ]


def completed(usage: dict | None = None, response_id: str = "resp_1") -> dict:
    """`response.completed` without a final `output`, so the streamed items stand."""
    response = {"id": response_id, "status": "completed"}
    if usage:
        response["usage"] = usage
    return {"type": "response.completed", "response": response}


def failed(message_text: str) -> dict:
    error = {"code": "server_error", "message": message_text}
    return {"type": "response.failed", "response": {"id": "resp_1", "error": error}}


@dataclass
class ResponsesProvider:
    listener: Listener
    answers: list[Answer] = field(default_factory=list)

    def answer(self, *answers: Answer) -> None:
        self.answers.extend(answers)

    def respond(self, _request: ReceivedRequest) -> Answer:
        if self.answers:
            return self.answers.pop(0)
        return events_stream(*message("ok"), completed())

    def sent(self) -> list[dict]:
        return [request.json() for request in self.listener.requests_to("/v1/responses")]


def connect_responses(client: httpx.Client, listener: Listener, **config) -> ResponsesProvider:
    provider = ResponsesProvider(listener)
    listener.route("POST", "/v1/responses", provider.respond)
    attach(client, listener, RESPONSES_MODEL, api_type="responses", **config)
    return provider
