"""Pins the per-model Skills builtin tool toggle (upstream 55e1c44c9) and lazy skill loading.

The model editor stores the Skills switch as `meta.builtinTools.skills`. A web chat on a native
function calling model gets the `<available_skills>` manifest and the `view_skill` tool unless
that key is false; with it false the skills attached to the model reach the system prompt whole,
as in non-native mode, and the other builtin tools are still offered. Since dev 6e8a6c446 and
003c9e085 the manifest comes as a `list_available_skills` result and a skill that arrives whole
as a preloaded `view_skill` result, both ahead of the model's reply.

Since 079fb19e7 (0.12.4) the admin's "Load selected skills on demand" (`ENABLE_SKILL_LAZY_LOADING`,
off by default) keeps the skills selected in a chat from being loaded: they are only listed for
`view_skill`, while a skill mentioned in the message still arrives whole. The setting is saved
through the admin config and kept by a save that leaves it out.

Discriminates: in a backend copy whose chat middleware and `get_builtin_tools` ignore
`builtinTools.skills`, the skills-off tests fail (manifest and `view_skill` still present, the
attached skill not injected whole). On dev f36490bf2, a backend copy whose chat middleware
ignores the lazy loading setting fails the lazy selected-skill test (the instructions arrive
whole), and one whose admin config resets the setting on a save that leaves it out fails the
round-trip test.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Iterator, NamedTuple

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.python_tools import EVERYONE_READS
from harness.tool_calls import skill_list, viewed_skills
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
LAZY_LOADING = "ENABLE_SKILL_LAZY_LOADING"


@pytest.fixture
def lazy_loading(admin, preserve):
    """`lazy_loading(on)` saves "Load selected skills on demand" for the rest of the test."""
    preserve("admin_config")

    def save(on: bool) -> dict:
        with admin.client() as client:
            current = client.get(ADMIN_CONFIG)
            assert current.status_code == 200, current.text
            saved = client.post(ADMIN_CONFIG, json={**current.json(), LAZY_LOADING: on})
        assert saved.status_code == 200, saved.text
        return saved.json()

    return save


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


class Sent(NamedTuple):
    system: str
    skill_list: str
    loaded: dict[str, dict]
    offered: set[str]


def provider_request(
    admin, upstream, model: str, content: str = "what is at the dock?", **extra
) -> Sent:
    """What the provider got for one web chat on `model` about skills, and the tools offered."""
    upstream.queue(reply.text("done"))
    with admin.client() as client:
        ask(client, content, model=model, **extra)
    request = upstream.chat_requests()[-1]
    system = "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )
    offered = {tool["function"]["name"] for tool in request.get("tools") or []}
    return Sent(system, skill_list(request), viewed_skills(request), offered)


def test_skills_are_listed_and_viewable_by_default(admin, upstream):
    with skilled_model(admin, None) as setup:
        sent = provider_request(admin, upstream, setup["model"])

    assert "<available_skills>" in sent.skill_list, sent
    assert setup["skill"]["name"] in sent.skill_list, sent.skill_list
    assert setup["text"] not in sent.system + sent.skill_list and not sent.loaded, sent
    assert "view_skill" in sent.offered, sorted(sent.offered)


def test_the_skills_toggle_off_removes_the_manifest_and_the_viewer(admin, upstream):
    with skilled_model(admin, {"skills": False}) as setup:
        sent = provider_request(admin, upstream, setup["model"])

    assert "<available_skills>" not in sent.system, sent.system
    assert setup["skill"]["name"] not in sent.system, sent.system
    assert not sent.skill_list, sent.skill_list
    assert "view_skill" not in sent.offered, sorted(sent.offered)
    assert "get_current_timestamp" in sent.offered, sorted(sent.offered)


def test_with_skills_off_an_attached_skill_arrives_whole(admin, upstream):
    with skilled_model(admin, {"skills": False}, attach=True) as setup:
        sent = provider_request(admin, upstream, setup["model"])

    assert f'<skill id="{setup["skill"]["id"]}"' in sent.system, sent.system
    assert f'name="{setup["skill"]["name"]}">' in sent.system, sent.system
    assert setup["text"] in sent.system, sent.system
    assert "<available_skills>" not in sent.system and not sent.skill_list, sent
    assert "view_skill" not in sent.offered, sorted(sent.offered)


# 079fb19e7: "Load selected skills on demand"


def test_a_selected_skill_arrives_whole_with_lazy_loading_off(admin, upstream, lazy_loading):
    lazy_loading(False)
    with skilled_model(admin, None) as setup:
        selected = [setup["skill"]["id"]]
        sent = provider_request(admin, upstream, setup["model"], skill_ids=selected)

    loaded = sent.loaded.get(setup["skill"]["id"])
    assert loaded and setup["text"] in loaded["content"], sent


def test_with_lazy_loading_on_a_selected_skill_is_only_listed(admin, upstream, lazy_loading):
    lazy_loading(True)
    with skilled_model(admin, None) as setup:
        selected = [setup["skill"]["id"]]
        sent = provider_request(admin, upstream, setup["model"], skill_ids=selected)

    assert not sent.loaded, f"the selected skill was loaded whole: {sent.loaded}"
    assert setup["text"] not in sent.system + sent.skill_list, sent
    assert setup["skill"]["name"] in sent.skill_list, sent.skill_list
    assert "view_skill" in sent.offered, sorted(sent.offered)


def test_with_lazy_loading_on_a_mentioned_skill_still_arrives_whole(admin, upstream, lazy_loading):
    lazy_loading(True)
    with skilled_model(admin, None) as setup:
        skill = setup["skill"]
        mention = f"<${skill['id']}|{skill['name']}> what is at the dock?"
        sent = provider_request(admin, upstream, setup["model"], mention)

    loaded = sent.loaded.get(skill["id"])
    assert loaded and setup["text"] in loaded["content"], f"the skill was not loaded: {sent}"


def test_the_lazy_loading_setting_is_saved_and_kept_by_saves_without_it(admin, lazy_loading):
    assert lazy_loading(True).get(LAZY_LOADING) is True
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG).json()
        assert current.get(LAZY_LOADING) is True, f"the setting was not saved: {current}"
        without = {key: value for key, value in current.items() if key != LAZY_LOADING}
        resaved = client.post(ADMIN_CONFIG, json=without)
        assert resaved.status_code == 200, resaved.text
        kept = client.get(ADMIN_CONFIG).json().get(LAZY_LOADING)

    assert kept is True, "a save without the setting switched lazy skill loading off"
    assert lazy_loading(False).get(LAZY_LOADING) is False
