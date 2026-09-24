"""An OpenAI-shaped model provider that records every request and answers from a script.

`reset(mode)` sets the fallback every reply uses (`ok`, `stream` or `error`). `queue(...)` lines
up scripted replies (text, reasoning, tool calls, usage, an HTTP error, a slow stream) that the
next chat completions consume in order; a reply with `match` only answers a request it accepts,
so a title or follow-up task cannot eat the reply meant for the chat. `requests` holds what
Open WebUI actually sent, which is how a test sees the payload it built.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterator

MOCK_MODEL_ID = "mock-model"


@dataclass
class Reply:
    content: str = ""
    reasoning: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict | None = None
    status: int = 200
    error_message: str = "upstream failed"
    chunk_delay: float = 0.0
    match: Callable[[dict], bool] | None = None


def text(content: str, **options) -> Reply:
    return Reply(content=content, **options)


def tool_call(name: str, arguments: dict | str, call_id: str = "call_1", **options) -> Reply:
    encoded = arguments if isinstance(arguments, str) else json.dumps(arguments)
    call = {"id": call_id, "type": "function", "function": {"name": name, "arguments": encoded}}
    return Reply(tool_calls=[call], **options)


def error(status: int = 500, message: str = "upstream failed", **options) -> Reply:
    return Reply(status=status, error_message=message, **options)


@dataclass
class UpstreamRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: dict | None


@dataclass
class MockUpstream:
    base_url: str
    behaviour: dict
    requests: list[UpstreamRequest] = field(default_factory=list)
    replies: deque[Reply] = field(default_factory=deque)
    models: list[str] = field(default_factory=lambda: [MOCK_MODEL_ID])
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(self, mode: str = "ok", **options) -> None:
        with self.lock:
            self.behaviour = {"mode": mode, **options}
            self.replies.clear()
            self.requests.clear()

    def queue(self, *replies: Reply) -> None:
        with self.lock:
            self.replies.extend(replies)

    def chat_requests(self) -> list[dict]:
        with self.lock:
            return [
                entry.body
                for entry in self.requests
                if entry.path.endswith("/chat/completions") and entry.body is not None
            ]

    def requests_to(self, suffix: str) -> list[UpstreamRequest]:
        with self.lock:
            return [entry for entry in self.requests if entry.path.endswith(suffix)]

    def next_reply(self, body: dict) -> Reply | None:
        with self.lock:
            for reply in self.replies:
                if reply.match is None or reply.match(body):
                    self.replies.remove(reply)
                    return reply
        return None


def serve(upstream_port: int) -> tuple[MockUpstream, Callable[[], None]]:
    """Start the provider on a daemon thread; returns it with its shutdown function."""
    upstream = MockUpstream(
        base_url=f"http://127.0.0.1:{upstream_port}/v1", behaviour={"mode": "ok"}
    )
    server = ThreadingHTTPServer(("127.0.0.1", upstream_port), _handler(upstream))
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def shutdown() -> None:
        server.shutdown()
        server.server_close()

    return upstream, shutdown


def _handler(upstream: MockUpstream):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # chunked framing needs it

        def log_message(self, *args) -> None:
            pass

        def _record(self, body: dict | None) -> None:
            entry = UpstreamRequest(self.command, self.path, dict(self.headers.items()), body)
            with upstream.lock:
                upstream.requests.append(entry)

        def _send_json(self, status: int, payload: dict) -> None:
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            self._record(None)
            if self.path.endswith("/models"):
                data = [{"id": model_id, "object": "model"} for model_id in upstream.models]
                self._send_json(200, {"object": "list", "data": data})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                body = json.loads(raw) if raw else {}
            except ValueError:
                body = None
            self._record(body)
            if self.path.endswith("/embeddings"):
                inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
                vectors = [{"embedding": [0.1, 0.2, 0.3]} for _ in inputs]
                self._send_json(200, {"object": "list", "data": vectors})
                return
            if not self.path.endswith("/chat/completions"):
                self._send_json(404, {"error": "not found"})
                return
            reply = upstream.next_reply(body or {})
            if reply is not None:
                self._answer(reply, bool((body or {}).get("stream")))
                return
            mode = upstream.behaviour["mode"]
            if mode == "error":
                self._send_json(500, {"error": {"message": "upstream failed"}})
            elif mode == "stream":
                self._stream_repeated(
                    upstream.behaviour["chunks"], upstream.behaviour["chunk_text"]
                )
            else:
                reply = text(upstream.behaviour.get("text", "ok"))
                self._answer(reply, bool((body or {}).get("stream")))

        def _answer(self, reply: Reply, stream: bool) -> None:
            if reply.status != 200:
                self._send_json(reply.status, {"error": {"message": reply.error_message}})
            elif stream:
                self._start_stream()
                for delta, finish_reason in _deltas(reply):
                    self._event(
                        _chunk(delta, finish_reason, reply.usage if finish_reason else None)
                    )
                    time.sleep(reply.chunk_delay)
                self._end_stream()
            else:
                self._send_json(200, _completion(reply))

        def _stream_repeated(self, chunks: int, chunk_text: str) -> None:
            self._start_stream()
            for i in range(chunks):
                delta = {"content": chunk_text} if i else {"role": "assistant", "content": ""}
                self._event(_chunk(delta, None, None))
            self._end_stream()

        def _start_stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

        def _event(self, payload: dict) -> None:
            self._write_chunk(f"data: {json.dumps(payload)}\n\n")

        def _end_stream(self) -> None:
            self._write_chunk("data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def _write_chunk(self, text: str) -> None:
            data = text.encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

    return Handler


def _deltas(reply: Reply) -> Iterator[tuple[dict, str | None]]:
    yield {"role": "assistant", "content": ""}, None
    if reply.reasoning:
        yield {"reasoning_content": reply.reasoning}, None
    if reply.content:
        yield {"content": reply.content}, None
    for index, call in enumerate(reply.tool_calls):
        header = {**call, "function": {"name": call["function"]["name"], "arguments": ""}}
        yield {"tool_calls": [{"index": index, **header}]}, None
        arguments = {"arguments": call["function"]["arguments"]}
        yield {"tool_calls": [{"index": index, "function": arguments}]}, None
    yield {}, "tool_calls" if reply.tool_calls else "stop"


def _chunk(delta: dict, finish_reason: str | None, usage: dict | None) -> dict:
    chunk = {
        "id": "mock",
        "object": "chat.completion.chunk",
        "model": MOCK_MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage:
        chunk["usage"] = usage
    return chunk


def _completion(reply: Reply) -> dict:
    message: dict = {"role": "assistant", "content": reply.content}
    if reply.reasoning:
        message["reasoning_content"] = reply.reasoning
    if reply.tool_calls:
        message["tool_calls"] = reply.tool_calls
    return {
        "id": "mock",
        "object": "chat.completion",
        "model": MOCK_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if reply.tool_calls else "stop",
            }
        ],
        "usage": reply.usage or {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
