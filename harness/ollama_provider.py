"""A `listener` played as an Ollama server, and the admin's Ollama connection pointed at it.

`serve_ollama(listener, *models)` keeps the server's models in `server.models` (append one later
and the server reports it) and answers what Open WebUI asks an Ollama server: `/api/tags`,
`/api/ps` (the names in `server.loaded`), `/api/version`, `/api/show`, `/api/chat` with one
finished reply (or the next answer queued with `server.queue_chat(...)`, which `chat_stream` and
`chat_line` shape), `/api/generate` (an empty prompt with `keep_alive: 0` unloads, as Ollama does),
`/api/embed`, and the model management calls. `/api/pull` and `/api/create` stream NDJSON
progress and add the model, `/api/copy` adds the copy and `/api/delete` removes the model; an
unknown model is a 404 with Ollama's error body. `server.sent(path)` is what each call carried.
`connect_ollama(client, listener, **config)` switches the Ollama API on with the listener as its
only connection and that per-connection config (a `prefix_id`). Snapshot `OLLAMA_CONFIG` with
`preserve` first so the setting is restored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx

from harness.listener import Answer, Listener, ReceivedRequest, json_answer

OLLAMA_CONFIG = ("/ollama/config", "/ollama/config/update")
OLLAMA_VERSION = "0.12.3"
EMBEDDING = [0.1, 0.2, 0.3]
GENERATED = "generated text"


def ndjson(*lines: dict) -> Answer:
    body = "".join(json.dumps(line) + "\n" for line in lines)
    return 200, {"Content-Type": "application/x-ndjson"}, body.encode()


def chat_line(model: str, message: dict, **final) -> dict:
    """One `/api/chat` line carrying `message`; `final` (a `done_reason`, the counters) ends it."""
    line = {"model": model, "message": {"role": "assistant", **message}, "done": bool(final)}
    return {**line, **final}


def chat_stream(model: str, *messages: dict, **counters) -> Answer:
    """A streamed `/api/chat` reply: a line per message, then the finishing line with `counters`."""
    finished = chat_line(model, {"content": ""}, done_reason="stop", **counters)
    return ndjson(*(chat_line(model, message) for message in messages), finished)


def model_not_found(name: str) -> Answer:
    return json_answer({"error": f"model '{name}' not found"}, status=404)


def _requested_model(request: ReceivedRequest) -> str:
    body = request.json()
    return body.get("model") or body.get("name") or ""


@dataclass
class OllamaServer:
    listener: Listener
    models: list[str] = field(default_factory=list)
    loaded: list[str] = field(default_factory=list)
    version: str = OLLAMA_VERSION
    chat_answers: list[Answer] = field(default_factory=list)

    def sent(self, path: str) -> list[dict]:
        return [request.json() for request in self.listener.requests_to(path) if request.body]

    def chat_requests(self) -> list[dict]:
        return self.sent("/api/chat")

    def queue_chat(self, *answers: Answer) -> None:
        self.chat_answers.extend(answers)

    def chat(self, _request: ReceivedRequest) -> Answer:
        if self.chat_answers:
            return self.chat_answers.pop(0)
        return json_answer({"message": {"role": "assistant", "content": "pong"}, "done": True})

    def tags(self, _request: ReceivedRequest) -> Answer:
        return json_answer({"models": [_tag(name) for name in self.models]})

    def running(self, _request: ReceivedRequest) -> Answer:
        return json_answer({"models": [_tag(name) for name in self.loaded]})

    def show(self, request: ReceivedRequest) -> Answer:
        name = _requested_model(request)
        if name not in self.models:
            return model_not_found(name)
        return json_answer(
            {
                "modelfile": f"FROM {name}\n",
                "parameters": "temperature 0.7",
                "template": "{{ .Prompt }}",
                "details": _details(),
                "model_info": {"general.architecture": "llama"},
                "capabilities": ["completion"],
            }
        )

    def pull(self, request: ReceivedRequest) -> Answer:
        name = _requested_model(request)
        self._add(name)
        return ndjson(
            {"status": "pulling manifest"},
            {"status": f"pulling {name}", "digest": "sha256:0f", "total": 100, "completed": 50},
            {"status": f"pulling {name}", "digest": "sha256:0f", "total": 100, "completed": 100},
            {"status": "verifying sha256 digest"},
            {"status": "writing manifest"},
            {"status": "success"},
        )

    def create(self, request: ReceivedRequest) -> Answer:
        base = request.json().get("from")
        if base and base not in self.models:
            return model_not_found(base)
        self._add(_requested_model(request))
        return ndjson(
            {"status": "using existing layer sha256:0f"},
            {"status": "writing manifest"},
            {"status": "success"},
        )

    def copy(self, request: ReceivedRequest) -> Answer:
        body = request.json()
        if body["source"] not in self.models:
            return model_not_found(body["source"])
        self._add(body["destination"])
        return 200, {}, b""

    def delete(self, request: ReceivedRequest) -> Answer:
        name = _requested_model(request)
        if name not in self.models:
            return model_not_found(name)
        self.models.remove(name)
        return 200, {}, b""

    def generate(self, request: ReceivedRequest) -> Answer:
        body = request.json()
        name = body["model"]
        if name not in self.models:
            return model_not_found(name)
        if body.get("keep_alive") == 0 and not body.get("prompt"):
            if name in self.loaded:
                self.loaded.remove(name)
            return json_answer(
                {"model": name, "response": "", "done": True, "done_reason": "unload"}
            )
        finished = {"model": name, "response": "", "done": True, "done_reason": "stop"}
        if body.get("stream", True):
            return ndjson({"model": name, "response": GENERATED, "done": False}, finished)
        return json_answer({**finished, "response": GENERATED})

    def embed(self, request: ReceivedRequest) -> Answer:
        body = request.json()
        if body["model"] not in self.models:
            return model_not_found(body["model"])
        inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
        return json_answer({"model": body["model"], "embeddings": [EMBEDDING for _ in inputs]})

    def _add(self, name: str) -> None:
        if name not in self.models:
            self.models.append(name)


def _details() -> dict:
    return {"format": "gguf", "family": "llama", "parameter_size": "1B", "quantization_level": "Q4"}


def _tag(name: str) -> dict:
    return {"name": name, "model": name, "size": 1024, "digest": "0f", "details": _details()}


def serve_ollama(listener: Listener, *models: str) -> OllamaServer:
    server = OllamaServer(listener, list(models))
    listener.route("GET", "/api/tags", server.tags)
    listener.route("GET", "/api/ps", server.running)
    listener.route("GET", "/api/version", lambda _request: json_answer({"version": server.version}))
    listener.route("POST", "/api/show", server.show)
    listener.route("POST", "/api/chat", server.chat)
    listener.route("POST", "/api/generate", server.generate)
    listener.route("POST", "/api/embed", server.embed)
    listener.route("POST", "/api/pull", server.pull)
    listener.route("POST", "/api/create", server.create)
    listener.route("POST", "/api/copy", server.copy)
    listener.route("DELETE", "/api/delete", server.delete)
    return server


def connect_ollama(client: httpx.Client, listener: Listener, **config) -> None:
    current = client.get(OLLAMA_CONFIG[0])
    current.raise_for_status()
    connected = {
        **current.json(),
        "ENABLE_OLLAMA_API": True,
        "OLLAMA_BASE_URLS": [listener.base_url],
        "OLLAMA_API_CONFIGS": {"0": config} if config else {},
    }
    saved = client.post(OLLAMA_CONFIG[1], json=connected)
    assert saved.status_code == 200, f"connecting the Ollama stand-in failed: {saved.text}"
