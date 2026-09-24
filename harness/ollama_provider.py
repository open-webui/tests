"""A `listener` played as an Ollama server, and the admin's Ollama connection pointed at it.

`serve_ollama(listener, *models)` answers `/api/tags` with the names in `server.models` (append
one later and the server reports it, the way `ollama pull` adds a model), `/api/ps` with no
loaded models and `/api/chat` with one finished reply. `connect_ollama(client, listener, **config)`
switches the Ollama API on with the listener as its only connection and that per-connection
config (a `prefix_id`). Snapshot `OLLAMA_CONFIG` with `preserve` first so the setting is restored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from harness.listener import Answer, Listener, ReceivedRequest, json_answer

OLLAMA_CONFIG = ("/ollama/config", "/ollama/config/update")


@dataclass
class OllamaServer:
    listener: Listener
    models: list[str] = field(default_factory=list)

    def tags(self, _request: ReceivedRequest) -> Answer:
        return json_answer({"models": [{"name": name, "model": name} for name in self.models]})

    def chat_requests(self) -> list[dict]:
        return [request.json() for request in self.listener.requests_to("/api/chat")]


def serve_ollama(listener: Listener, *models: str) -> OllamaServer:
    server = OllamaServer(listener, list(models))
    listener.route("GET", "/api/tags", server.tags)
    listener.route("GET", "/api/ps", json_answer({"models": []}))
    listener.route(
        "POST",
        "/api/chat",
        json_answer({"message": {"role": "assistant", "content": "pong"}, "done": True}),
    )
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
