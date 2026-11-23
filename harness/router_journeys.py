"""Router models with patterns, conditions and default models, on provider models of their own.

`make_targets(admin, upstream, "code", "general")` adds a provider model per label that every
account may use, named "<Label> <tag>" so a reply header names it uniquely. `Decider` serves the
Decisions API on a `listener`: it scores each condition it is asked about by the test's
`scores` (keyed by a piece of the condition text, 0 for the rest) and counts its calls.
`save_router(admin, ...)` saves a router model the way the Router Models dialog does, with its
Decisions API settings, and sets who may pick it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import httpx

from harness.actors import Actor
from harness.listener import Answer, Listener, ReceivedRequest, json_answer
from harness.router_models import ROUTER_MODELS_CONFIG
from harness.upstream import MockUpstream

EVERYONE_READS = {"principal_type": "user", "principal_id": "*", "permission": "read"}
DECIDER_KEY = "sk-decider-secret-4b1d"
DECIDER_MODEL = "mock-decider"


@dataclass
class Target:
    id: str
    name: str


@dataclass
class Decider:
    listener: Listener
    scores: dict[str, float] = field(default_factory=dict)
    # a field a real service may add to its answer, which the chat must never show
    note: str = "decider-note-" + uuid.uuid4().hex[:6]

    def __post_init__(self) -> None:
        self.listener.route("POST", "/v1/decisions", self._answer)

    @property
    def base_url(self) -> str:
        return f"{self.listener.base_url}/v1"

    def calls(self) -> list[dict]:
        return [request.json() for request in self.listener.requests_to("/v1/decisions")]

    def _answer(self, request: ReceivedRequest) -> Answer:
        answers = [
            {
                "type": "predicate",
                "name": question["name"],
                "probability": self._score(question.get("instructions", "")),
                "note": self.note,
            }
            for question in request.json().get("questions", [])
        ]
        usage = {"prompt_tokens": 7000, "completion_tokens": 900, "total_tokens": 7900}
        return json_answer({"answers": answers, "usage": usage})

    def _score(self, instructions: str) -> float:
        return next((score for key, score in self.scores.items() if key in instructions), 0.0)


def make_targets(admin: Actor, upstream: MockUpstream, *labels: str) -> dict[str, Target]:
    """A provider model per label, readable by every account, by label."""
    tag = uuid.uuid4().hex[:6]
    targets = {label: Target(f"{label}-{tag}", f"{label.title()} {tag}") for label in labels}
    upstream.models.extend(target.id for target in targets.values())
    with admin.client() as client:
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        for target in targets.values():
            created = client.post(
                "/api/v1/models/create",
                json={
                    "id": target.id,
                    "base_model_id": None,
                    "name": target.name,
                    "meta": {},
                    "params": {},
                    "access_grants": [EVERYONE_READS],
                },
            )
            assert created.status_code == 200, f"adding {target.name} failed: {created.text}"
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
    return targets


def remove_targets(admin: Actor, upstream: MockUpstream, targets: dict[str, Target]) -> None:
    with admin.client() as client:
        for target in targets.values():
            client.post("/api/v1/models/model/delete", json={"id": target.id})
            if target.id in upstream.models:
                upstream.models.remove(target.id)


def pattern(target: Target, regex: str) -> dict:
    return {"model_id": target.id, "pattern": regex}


def condition(target: Target, text: str) -> dict:
    return {"model_id": target.id, "condition": text}


def default(target: Target) -> dict:
    return {"model_id": target.id}


def save_router(
    admin: Actor,
    routes: list[dict],
    *,
    decider: Decider | None = None,
    strategy: str = "first",
    access_grants: list[dict] | None = None,
    threshold: float = 0.5,
) -> Target:
    """Save a router model next to the saved ones, then set who may pick it, as the dialog does."""
    tag = uuid.uuid4().hex[:6]
    router = Target(f"router-{tag}", f"Router {tag}")
    decisions = (
        {
            "api_base_url": decider.base_url,
            "api_key": DECIDER_KEY,
            "model": DECIDER_MODEL,
            "threshold": threshold,
            "prompt": "",
        }
        if decider
        else {}
    )
    entry = {
        "id": router.id,
        "name": router.name,
        "meta": {"routes": routes, "strategy": strategy},
        "config": {"decisions": decisions} if decisions else {},
    }
    with admin.client() as client:
        current = client.get(ROUTER_MODELS_CONFIG[0])
        assert current.status_code == 200, current.text
        saved = client.post(
            ROUTER_MODELS_CONFIG[1],
            json={"ROUTER_MODELS": [*current.json()["ROUTER_MODELS"], entry]},
        )
        assert saved.status_code == 200, f"saving {router.name} failed: {saved.text}"
        access = client.post(
            "/api/v1/models/model/access/update",
            json={
                "id": router.id,
                "name": router.name,
                "access_grants": [EVERYONE_READS] if access_grants is None else access_grants,
            },
        )
        assert access.status_code == 200, (
            f"setting the access of {router.name} failed: {access.text}"
        )
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
    return router


def models_listed(client: httpx.Client) -> dict[str, dict]:
    listed = client.get("/api/models", params={"refresh": "true"})
    assert listed.status_code == 200, listed.text
    return {model["id"]: model for model in listed.json()["data"]}
