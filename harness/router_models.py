"""Router models, saved by the admin the way Admin Settings > Models > Router Models saves one.

The panel stores the router model in the router settings, then its access on a model row of
the same id, which makes the saving admin the row's owner.

`create_target` makes a preset of the scripted model for a router to route to: its system prompt
marks the provider requests routed to it, and `queue_target_answers` lets each target answer a
prompt as itself, so the reply names the model that answered.

`Decider` stands in for the Decisions API that scores a router model's conditions: a `listener`
route at `<API Base URL>/decisions` that answers each condition with the probability the test set
for it, or with whatever broken answer the test hands it.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable

import httpx

from harness import upstream as reply
from harness.listener import Answer, Listener, ReceivedRequest, json_answer
from harness.upstream import MOCK_MODEL_ID, MockUpstream

ROUTER_MODELS_CONFIG = ("/api/v1/configs/router_models", "/api/v1/configs/router_models")
EVERYONE_READS = {"principal_type": "user", "principal_id": "*", "permission": "read"}
DECIDER_PATH = "/v1/decisions"
DECIDER_KEY = "sk-decider-key"
DECIDER_MODEL = "router-decider-1"


def default_routes(*model_ids: str) -> list[dict]:
    """Routes without a pattern or a condition: the models the router picks from by strategy."""
    return [{"model_id": model_id} for model_id in model_ids]


def save_router_model(
    client: httpx.Client,
    model_id: str,
    name: str,
    routes: list[dict],
    access_grants: list[dict],
    strategy: str = "random",
    meta: dict | None = None,
    config: dict | None = None,
) -> None:
    """Add a router model next to the ones already saved and set its access, as the panel does."""
    current = client.get(ROUTER_MODELS_CONFIG[0])
    assert current.status_code == 200, current.text
    router_model = {
        "id": model_id,
        "name": name,
        "meta": {**(meta or {}), "routes": routes, "strategy": strategy},
        "config": config or {},
    }
    saved = client.post(
        ROUTER_MODELS_CONFIG[1],
        json={"ROUTER_MODELS": [*current.json()["ROUTER_MODELS"], router_model]},
    )
    assert saved.status_code == 200, f"saving router model {name} failed: {saved.text}"
    access = client.post(
        "/api/v1/models/model/access/update",
        json={"id": model_id, "name": name, "access_grants": access_grants},
    )
    assert access.status_code == 200, f"setting the access of {name} failed: {access.text}"


@dataclass
class Target:
    """A model a router model routes to."""

    id: str
    name: str
    marker: str  # its system prompt, which marks the provider requests routed to it
    answer: str  # what it answers, which leaves out its name so only a header can show that

    def was_sent(self, body: dict) -> bool:
        return any(
            message.get("role") == "system" and self.marker in str(message.get("content"))
            for message in body.get("messages", [])
        )


def create_target(
    client: httpx.Client, label: str, model_id: str | None = None, private: bool = False
) -> Target:
    """A preset of the scripted model, readable by everyone unless `private`."""
    tag = uuid.uuid4().hex[:8]
    model_id = model_id or f"{label}-{tag}"
    target = Target(
        model_id,
        f"{label.title()} {tag}",
        f"You are the {label} model {tag}.",
        f"Routed answer {uuid.uuid4().hex[:10]}.",
    )
    created = client.post(
        "/api/v1/models/create",
        json={
            "id": target.id,
            "base_model_id": MOCK_MODEL_ID,
            "name": target.name,
            "meta": {},
            "params": {"system": target.marker},
            "access_grants": [] if private else [EVERYONE_READS],
        },
    )
    assert created.status_code == 200, f"creating the target {label} failed: {created.text}"
    return target


def queue_target_answers(upstream: MockUpstream, prompt: str, targets: list[Target]) -> None:
    """Each of `targets` answers `prompt` with its own `answer`, if the router sends it there."""

    def routed_to(target: Target) -> Callable[[dict], bool]:
        asked = reply.answering(prompt)
        return lambda body: target.was_sent(body) and asked(body)

    upstream.queue(*(reply.text(target.answer, match=routed_to(target)) for target in targets))


@dataclass
class Decider:
    """The Decisions API of a router model, served by `listener` and scripted by the test."""

    listener: Listener
    scores: dict[str, float] = field(default_factory=dict)  # condition text -> probability
    answer: Callable[[ReceivedRequest], Answer] | None = None  # replaces the scored answer

    def __post_init__(self) -> None:
        self.listener.route("POST", DECIDER_PATH, self._answer)

    @property
    def base_url(self) -> str:
        return f"{self.listener.base_url}/v1"

    def config(self, threshold: float = 0.5, prompt: str = "", base_url: str | None = None) -> dict:
        """The router model's `config` with this decider as its Decisions API."""
        return {
            "decisions": {
                "api_base_url": base_url or self.base_url,
                "api_key": DECIDER_KEY,
                "model": DECIDER_MODEL,
                "threshold": threshold,
                "prompt": prompt,
            }
        }

    def requests(self) -> list[ReceivedRequest]:
        return self.listener.requests_to(DECIDER_PATH)

    def inputs(self) -> list[str]:
        return [str(request.json().get("input")) for request in self.requests()]

    def _answer(self, request: ReceivedRequest) -> Answer:
        if self.answer:
            return self.answer(request)
        answers = [
            {
                "type": "predicate",
                "name": question["name"],
                "probability": self._score(question),
            }
            for question in request.json()["questions"]
        ]
        return json_answer({"answers": answers})

    def _score(self, question: dict) -> float:
        asked = json.dumps(question)
        return next((score for text, score in self.scores.items() if text in asked), 0.0)


def conditions_asked(request: ReceivedRequest, conditions: list[str]) -> list[str]:
    """Which of `conditions` each question of a Decisions request carries, in question order."""
    asked = []
    for question in request.json().get("questions", []):
        text = json.dumps(question)
        asked.extend(condition for condition in conditions if condition in text)
    return asked


def held_answer(release: threading.Event, answer: Answer) -> Callable[[ReceivedRequest], Answer]:
    """A decider answer that waits for `release` (or 60 seconds) first, like a stalled service."""

    def respond(_request: ReceivedRequest) -> Answer:
        release.wait(60)
        return answer

    return respond
