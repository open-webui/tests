"""Journey: the Visualizations builtin tool, its switch on the model and its admin prompt.

Since 0.13.4 a model can draw an interactive chart, diagram or calculator in the chat with the
Visualizations builtin tool (`render_visualization`). It is off by default: a model is offered it
only once Visualizations is ticked under its Builtin Tools (`meta.builtinTools.visualizations`).
The tool's description, which tells the model when and how to draw, is the built-in prompt the
chat config reports as `DEFAULT_VISUALIZATION_PROMPT_TEMPLATE` until the admin writes one under
Admin Settings > Interface > Visualization Prompt (`VISUALIZATION_PROMPT_TEMPLATE`); a blank one
brings the built-in prompt back and a save that leaves the field out keeps it. The tool checks
what the model drew and answers `{"status": "accepted"}`, or names what is wrong: an empty title
or html, an unknown library. The browser side is e2e/chat/test_visualizations.py.

Discriminates: passes on dev 485103cde. In a backend copy whose builtin tools offer
`render_visualization` to every model the off-by-default test fails; in one whose tool spec
keeps the docstring in place of the admin prompt the prompt test fails; in one whose tool
accepts every call the refused-call test fails.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from typing import Iterator

import pytest

from harness.python_tools import EVERYONE_READS
from harness.tool_calls import offered_tool_specs, run_tool
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
TOOL = "render_visualization"
TIDE_CHART = (
    '<svg viewBox="0 0 10 10" role="img" aria-label="Tide"><rect width="4" height="8"/></svg>'
)


@contextlib.contextmanager
def preset(owner, builtin_tools: dict | None) -> Iterator[str]:
    """A native function calling preset on the scripted model, with these Builtin Tools ticks."""
    model_id = f"charts-{uuid.uuid4().hex[:8]}"
    meta: dict = {"capabilities": {}}
    if builtin_tools is not None:
        meta["builtinTools"] = builtin_tools
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": model_id,
        "meta": meta,
        "params": {"function_calling": "native"},
        "access_grants": [EVERYONE_READS],
    }
    with owner.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        try:
            yield model_id
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def offered(actor, upstream, model_id: str) -> dict[str, dict]:
    with actor.client() as client:
        return offered_tool_specs(client, upstream, model=model_id)


def chat_config(admin) -> dict:
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0])
    assert current.status_code == 200, current.text
    return current.json()


def save_chat_config(admin, **changes) -> dict:
    current = chat_config(admin)
    form = {key: value for key, value in current.items() if not key.startswith("DEFAULT_")}
    with admin.client() as client:
        saved = client.post(CHAT_CONFIG[1], json={**form, **changes})
    assert saved.status_code == 200, saved.text
    return saved.json()


@pytest.fixture
def chat_config_restored(preserve):
    preserve(CHAT_CONFIG)


def test_a_model_is_offered_visualizations_only_once_they_are_ticked(admin, make_user, upstream):
    person = make_user()
    with preset(admin, None) as untouched, preset(admin, {"visualizations": True}) as ticked:
        by_default = offered(person, upstream, untouched)
        with_it = offered(person, upstream, ticked)

    assert "get_current_timestamp" in by_default, sorted(by_default)
    assert TOOL not in by_default, "a model nobody ticked Visualizations on was offered the tool"
    assert TOOL in with_it, f"the ticked model was not offered the tool: {sorted(with_it)}"
    parameters = with_it[TOOL]["parameters"]
    assert {"title", "html"} <= set(parameters.get("required") or []), parameters


def test_the_tool_is_described_by_the_built_in_prompt_until_the_admin_writes_one(
    admin, make_user, upstream, chat_config_restored
):
    person = make_user()
    built_in = chat_config(admin)["DEFAULT_VISUALIZATION_PROMPT_TEMPLATE"]
    assert "chart" in built_in, f"the built-in prompt is not about drawing: {built_in!r}"
    own = f"Draw only tide tables for the harbour {uuid.uuid4().hex[:8]}."

    with preset(admin, {"visualizations": True}) as model_id:
        save_chat_config(admin, VISUALIZATION_PROMPT_TEMPLATE="")
        default_description = offered(person, upstream, model_id)[TOOL]["description"]

        saved = save_chat_config(admin, VISUALIZATION_PROMPT_TEMPLATE=own)
        assert saved["VISUALIZATION_PROMPT_TEMPLATE"] == own, saved
        own_description = offered(person, upstream, model_id)[TOOL]["description"]

        save_chat_config(admin, VISUALIZATION_PROMPT_TEMPLATE="   ")
        blank_description = offered(person, upstream, model_id)[TOOL]["description"]

    assert default_description == built_in, (
        f"the tool did not carry the built-in prompt: {default_description[:200]!r}"
    )
    assert own_description == own, (
        f"the tool did not carry the admin's prompt: {own_description[:200]!r}"
    )
    assert blank_description == built_in, (
        f"a blank prompt did not bring the built-in one back: {blank_description[:200]!r}"
    )


def test_a_save_without_the_visualization_prompt_keeps_it(admin, chat_config_restored):
    own = f"Keep charts small {uuid.uuid4().hex[:8]}."
    save_chat_config(admin, VISUALIZATION_PROMPT_TEMPLATE=own)

    form = {
        key: value
        for key, value in chat_config(admin).items()
        if not key.startswith("DEFAULT_") and key != "VISUALIZATION_PROMPT_TEMPLATE"
    }
    with admin.client() as client:
        resaved = client.post(CHAT_CONFIG[1], json=form)
    assert resaved.status_code == 200, resaved.text

    assert chat_config(admin)["VISUALIZATION_PROMPT_TEMPLATE"] == own, (
        "a save from a form without the Visualization Prompt wiped it"
    )


def test_a_drawing_is_accepted(admin, make_user, upstream):
    person = make_user()
    with preset(admin, {"visualizations": True}) as model_id, person.client() as client:
        plain = run_tool(
            client, upstream, TOOL, {"title": "Tide", "html": TIDE_CHART}, model=model_id
        )
        chart = run_tool(
            client,
            upstream,
            TOOL,
            {"title": "Tide", "html": '<canvas id="c"></canvas>', "libraries": ["chartjs"]},
            model=model_id,
        )

    assert json.loads(plain) == {"status": "accepted"}, plain
    assert json.loads(chart) == {"status": "accepted"}, chart


@pytest.mark.parametrize(
    ("arguments", "named"),
    [
        ({"title": " ", "html": TIDE_CHART}, "title"),
        ({"title": "Tide", "html": "  "}, "html"),
        ({"title": "Tide", "html": TIDE_CHART, "libraries": ["d3"]}, "libraries"),
    ],
    ids=["blank title", "blank html", "unknown library"],
)
def test_a_drawing_the_tool_cannot_show_is_refused_with_the_reason(
    admin, make_user, upstream, arguments, named
):
    person = make_user()
    with preset(admin, {"visualizations": True}) as model_id, person.client() as client:
        result = run_tool(client, upstream, TOOL, arguments, model=model_id)

    answer = json.loads(result)
    assert "status" not in answer, f"the tool accepted a drawing it cannot show: {answer}"
    assert named in answer.get("error", ""), f"the refusal does not name the {named}: {answer}"
