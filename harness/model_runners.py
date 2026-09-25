"""llama.cpp and LM Studio played by a `listener`, added as an OpenAI connection of that provider.

`serve_llama_cpp(listener, *models)` answers as llama-server in router mode: `/v1/models` and the
catalog at `/models` give each model a `status` from `runner.loaded`, `/models/load` and
`/models/unload` change it, `POST /models` downloads a model, `DELETE /models?model=` removes one
and `/models/sse` streams one status event. `serve_lm_studio(listener, *models)` answers LM
Studio's REST API: `/api/v1/models` with `loaded_instances`, load and unload by instance id, and
a download job that completes (adding the model) when its status is read; `/v1/models` is its
OpenAI-compatible list. `runner.sent(path)` is what each call carried.
`connect_runner(client, runner)` appends the listener at `/v1` as a connection of the runner's
provider type and returns its index. Snapshot `OPENAI_CONFIG` with `preserve` first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlsplit

import httpx

from harness.listener import Answer, Listener, ReceivedRequest, json_answer, text_answer
from harness.second_provider import OPENAI_CONFIG

DOWNLOAD_JOB = "job-1"


@dataclass
class ModelRunner:
    listener: Listener
    provider: str
    models: list[str] = field(default_factory=list)
    loaded: list[str] = field(default_factory=list)
    downloading: list[str] = field(default_factory=list)

    def sent(self, path: str) -> list[dict]:
        return [request.json() for request in self.listener.requests_to(path) if request.body]

    def openai_models(self, _request: ReceivedRequest) -> Answer:
        return json_answer({"object": "list", "data": [self._entry(name) for name in self.models]})

    def _entry(self, name: str) -> dict:
        entry = {"id": name, "object": "model", "owned_by": self.provider}
        if self.provider == "llama.cpp":
            entry["status"] = {"value": "loaded" if name in self.loaded else "unloaded"}
        return entry

    def _unknown(self, name: str) -> Answer | None:
        if name in self.models:
            return None
        return json_answer({"error": {"message": f"model {name} not found"}}, status=404)

    def llama_load(self, request: ReceivedRequest) -> Answer:
        name = request.json()["model"]
        if refused := self._unknown(name):
            return refused
        self.loaded.append(name)
        return json_answer({"success": True})

    def llama_unload(self, request: ReceivedRequest) -> Answer:
        name = request.json()["model"]
        if name not in self.loaded:
            return json_answer({"error": {"message": f"model {name} is not loaded"}}, status=400)
        self.loaded.remove(name)
        return json_answer({"success": True})

    def llama_download(self, request: ReceivedRequest) -> Answer:
        self.models.append(request.json()["model"])
        return json_answer({"success": True})

    def llama_delete(self, request: ReceivedRequest) -> Answer:
        name = parse_qs(urlsplit(request.path).query).get("model", [""])[0]
        if refused := self._unknown(name):
            return refused
        self.models.remove(name)
        return json_answer({"success": True})

    def llama_events(self, _request: ReceivedRequest) -> Answer:
        event = {"models": [self._entry(name) for name in self.models]}
        return text_answer(f"data: {json.dumps(event)}\n\n", content_type="text/event-stream")

    def studio_models(self, _request: ReceivedRequest) -> Answer:
        listed = [
            {
                "type": "llm",
                "key": name,
                "display_name": name,
                "loaded_instances": [{"id": name}] if name in self.loaded else [],
            }
            for name in self.models
        ]
        return json_answer({"models": listed})

    def studio_download(self, request: ReceivedRequest) -> Answer:
        self.downloading.append(request.json()["model"])
        return json_answer({"job_id": DOWNLOAD_JOB, "status": "downloading"})

    def studio_download_status(self, _request: ReceivedRequest) -> Answer:
        self.models.extend(self.downloading)
        self.downloading.clear()
        return json_answer({"job_id": DOWNLOAD_JOB, "status": "completed"})

    def studio_load(self, request: ReceivedRequest) -> Answer:
        name = request.json()["model"]
        if refused := self._unknown(name):
            return refused
        self.loaded.append(name)
        return json_answer({"type": "llm", "instance_id": name, "status": "loaded"})

    def studio_unload(self, request: ReceivedRequest) -> Answer:
        instance_id = request.json().get("instance_id")
        if instance_id not in self.loaded:
            return json_answer({"error": {"message": f"no instance {instance_id}"}}, status=404)
        self.loaded.remove(instance_id)
        return json_answer({"instance_id": instance_id})


def serve_llama_cpp(listener: Listener, *models: str) -> ModelRunner:
    runner = ModelRunner(listener, "llama.cpp", list(models))
    listener.route("GET", "/v1/models", runner.openai_models)
    listener.route("GET", "/models", runner.openai_models)
    listener.route("POST", "/models", runner.llama_download)
    listener.route("DELETE", "/models", runner.llama_delete)
    listener.route("POST", "/models/load", runner.llama_load)
    listener.route("POST", "/models/unload", runner.llama_unload)
    listener.route("GET", "/models/sse", runner.llama_events)
    return runner


def serve_lm_studio(listener: Listener, *models: str) -> ModelRunner:
    runner = ModelRunner(listener, "lmstudio", list(models))
    listener.route("GET", "/v1/models", runner.openai_models)
    listener.route("GET", "/api/v1/models", runner.studio_models)
    listener.route("POST", "/api/v1/models/download", runner.studio_download)
    listener.route(
        "GET", f"/api/v1/models/download/status/{DOWNLOAD_JOB}", runner.studio_download_status
    )
    listener.route("POST", "/api/v1/models/load", runner.studio_load)
    listener.route("POST", "/api/v1/models/unload", runner.studio_unload)
    return runner


def connect_runner(client: httpx.Client, runner: ModelRunner, **config) -> int:
    """Append the runner as a connection with its provider type; returns the connection index."""
    current = client.get(OPENAI_CONFIG[0])
    current.raise_for_status()
    connections = current.json()
    index = len(connections["OPENAI_API_BASE_URLS"])
    updated = {
        **connections,
        "OPENAI_API_BASE_URLS": [
            *connections["OPENAI_API_BASE_URLS"],
            f"{runner.listener.base_url}/v1",
        ],
        "OPENAI_API_KEYS": [*connections["OPENAI_API_KEYS"], "sk-runner"],
        "OPENAI_API_CONFIGS": {
            **connections["OPENAI_API_CONFIGS"],
            str(index): {"enable": True, "provider": runner.provider, **config},
        },
    }
    saved = client.post(OPENAI_CONFIG[1], json=updated)
    assert saved.status_code == 200, (
        f"connecting the {runner.provider} stand-in failed: {saved.text}"
    )
    return index
