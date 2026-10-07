"""A local OpenAI-shaped provider a browser can call directly, for a user's own connection.

A direct connection is fetched by the page itself, so the provider must answer the browser's
CORS preflight as well as the calls. `serve(listener, model_id)` routes `/v1/models` and
`/v1/chat/completions` on a `listener` with the headers a cross-origin page needs;
`provider.reply_with(text)` sets what the next chats answer. `provider.models_requests()` and
`provider.chat_requests()` are what the browser sent, headers included. Further model ids after
the first are listed too, and `provider.refuse(status, message)` makes the model list answer that
status with an OpenAI-shaped error body, the way a provider refuses a wrong key. A chat answer
is streamed one event at a time with short gaps.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Iterator

from harness.listener import Answer, Listener, ReceivedRequest, json_answer

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "authorization, content-type",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
}


@dataclass
class BrowserProvider:
    listener: Listener
    model_id: str
    other_model_ids: tuple[str, ...] = ()
    text: str = "hello from the browser side"
    refusal: tuple[int, str] | None = None

    @property
    def base_url(self) -> str:
        return f"{self.listener.base_url}/v1"

    def reply_with(self, text: str) -> None:
        self.text = text

    def refuse(self, status: int, message: str) -> None:
        self.refusal = (status, message)

    def models_requests(self) -> list[ReceivedRequest]:
        return [r for r in self.listener.requests_to("/v1/models") if r.method == "GET"]

    def chat_requests(self) -> list[ReceivedRequest]:
        return [r for r in self.listener.requests_to("/v1/chat/completions") if r.method == "POST"]

    def _models(self, _request: ReceivedRequest) -> Answer:
        if self.refusal:
            status, message = self.refusal
            status, headers, body = json_answer({"error": {"message": message}}, status)
            return status, {**headers, **CORS}, body
        ids = (self.model_id, *self.other_model_ids)
        status, headers, body = json_answer(
            {"object": "list", "data": [{"id": model_id} for model_id in ids]}
        )
        return status, {**headers, **CORS}, body

    def _chat(self, _request: ReceivedRequest) -> Answer:
        delta = {"index": 0, "delta": {"role": "assistant", "content": self.text}}
        stop = {"index": 0, "delta": {}, "finish_reason": "stop"}
        events = [
            f"data: {json.dumps({'object': 'chat.completion.chunk', 'choices': [choice]})}\n\n"
            for choice in (delta, stop)
        ] + ["data: [DONE]\n\n"]
        headers = {"Content-Type": "text/event-stream", **CORS}
        return 200, headers, _paced(events)


def _paced(events: list[str]) -> Iterator[bytes]:
    # the page relays each line over a socket and the server may reorder a burst, so space them out
    for event in events:
        yield event.encode()
        time.sleep(0.15)


def _preflight(_request: ReceivedRequest) -> Answer:
    return 204, CORS, b""


def serve(listener: Listener, model_id: str, *other_model_ids: str) -> BrowserProvider:
    provider = BrowserProvider(listener, model_id, other_model_ids)
    listener.route("GET", "/v1/models", provider._models)
    listener.route("POST", "/v1/chat/completions", provider._chat)
    for path in ("/v1/models", "/v1/chat/completions"):
        listener.route("OPTIONS", path, _preflight)
    return provider
