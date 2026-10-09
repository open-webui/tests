"""Pins the per-model Skills builtin tool toggle (upstream 55e1c44c9).

The model editor stores the Skills switch as `meta.builtinTools.skills`. A web chat on a native
function calling model gets the `<available_skills>` manifest and the `view_skill` tool unless
that key is false; with it false the skills attached to the model reach the system prompt whole,
as in non-native mode, and the other builtin tools are still offered.

Discriminates: in a backend copy whose chat middleware and `get_builtin_tools` ignore
`builtinTools.skills`, the skills-off tests fail (manifest and `view_skill` still present, the
attached skill not injected whole).
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Iterator

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


@contextlib.contextmanager
def skilled_model(admin, builtin_tools: dict | None, attach: bool = False) -> Iterator[dict]:
    """A native function calling model and a skill the admin owns; yields their details."""
    tag = uuid.uuid4().hex[:8]
    text = f"Tie the bow line first {tag}."
    skill = {"id": f"dock-{tag}", "name": f"Dock manners {tag}"}
    model_id = f"skilled-{tag}"
    meta: dict = {"capabilities": {}}
    if builtin_tools is not None:
        meta["builtinTools"] = builtin_tools
    if attach:
        meta["skillIds"] = [skill["id"]]
    with admin.client() as client:
        created = client.post(
            "/api/v1/skills/create",
            json={
                **skill,
                "description": f"How to behave at the dock {tag}",
                "content": text,
                "meta": {},
                "access_grants": [],
            },
        )
        assert created.status_code == 200, created.text
        try:
            made = client.post(
                "/api/v1/models/create",
                json={
                    "id": model_id,
                    "base_model_id": MOCK_MODEL_ID,
                    "name": model_id,
                    "meta": meta,
                    "params": {"function_calling": "native"},
                    "access_grants": [EVERYONE_READS],
                },
            )
            assert made.status_code == 200, made.text
            client.get("/api/models", params={"refresh": "true"}).raise_for_status()
            yield {"model": model_id, "skill": skill, "text": text, "tag": tag}
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model_id})
            client.delete(f"/api/v1/skills/id/{skill['id']}/delete")


def provider_request(admin, upstream, model: str) -> tuple[str, set[str]]:
    """The system prompt and tool names the provider got for one web chat on `model`."""
    upstream.queue(reply.text("done"))
    with admin.client() as client:
        ask(client, "what is at the dock?", model=model)
    request = upstream.chat_requests()[-1]
    system = "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )
    return system, {tool["function"]["name"] for tool in request.get("tools") or []}


def test_skills_are_listed_and_viewable_by_default(admin, upstream):
    with skilled_model(admin, None) as setup:
        system, offered = provider_request(admin, upstream, setup["model"])

    assert "<available_skills>" in system and setup["skill"]["name"] in system, system
    assert setup["text"] not in system, system
    assert "view_skill" in offered, sorted(offered)


def test_the_skills_toggle_off_removes_the_manifest_and_the_viewer(admin, upstream):
    with skilled_model(admin, {"skills": False}) as setup:
        system, offered = provider_request(admin, upstream, setup["model"])

    assert "<available_skills>" not in system and setup["skill"]["name"] not in system, system
    assert "view_skill" not in offered, sorted(offered)
    assert "get_current_timestamp" in offered, sorted(offered)


def test_with_skills_off_an_attached_skill_arrives_whole(admin, upstream):
    with skilled_model(admin, {"skills": False}, attach=True) as setup:
        system, offered = provider_request(admin, upstream, setup["model"])

    assert f'<skill id="{setup["skill"]["id"]}"' in system, system
    assert f'name="{setup["skill"]["name"]}">' in system, system
    assert setup["text"] in system, system
    assert "<available_skills>" not in system, system
    assert "view_skill" not in offered, sorted(offered)
