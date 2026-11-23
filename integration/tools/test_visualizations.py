"""Journey: the Visualizations builtin tool, its switch on the model and its admin prompt.

Since 0.13.4 a model can draw an interactive chart, diagram or calculator in the chat with the
Visualizations builtin tool (`render_visualization`). It is off by default: a model is offered it
only once Visualizations is ticked under its Builtin Tools (`meta.builtinTools.visualizations`).
The tool's description, which tells the model when and how to draw, is a built-in prompt until
the admin writes one under Admin Settings > Interface > Visualization Prompt
(`VISUALIZATION_PROMPT_TEMPLATE`); a blank one brings the built-in prompt back and a save that
leaves the field out keeps it. Since b462a23a7 the chat config no longer reports the built-in
prompt, so the test reads it from the tool the model is offered. The tool checks
what the model drew and answers `{"status": "accepted"}`, or names what is wrong: an empty title
or html, an unknown library. A drawing whose script calls Chart.js (`new Chart(`) without
`libraries=["chartjs"]` cannot load it, so it is refused with an error naming the library, called
directly or through Tool Search's `call_tool`, and the model's retry with the library is accepted
and kept on the reply; the word, a chart class, own canvas drawing or Chart.js code shown as text
stays accepted. The browser side is e2e/chat/test_visualizations.py.

Discriminates: passes on dev 485103cde. In a backend copy whose builtin tools offer
`render_visualization` to every model the off-by-default test fails; in one whose tool spec
keeps the docstring in place of the admin prompt the prompt test fails; in one whose tool
accepts every call the refused-call test fails. The Chart.js retry test is red on dev 306f9be78,
where the chart is accepted and fails in the browser (open-webui/open-webui#34909); it passes on
the fix branch fix/visualizer-chart-needs-library (3619665e0).
"""

from __future__ import annotations

import contextlib
import json
import uuid
from typing import Iterator

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.python_tools import EVERYONE_READS
from harness.tool_calls import offered_tool_specs, run_tool, without_hidden_calls
from harness.upstream import MOCK_MODEL_ID, Reply

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
    own = f"Draw only tide tables for the harbour {uuid.uuid4().hex[:8]}."

    with preset(admin, {"visualizations": True}) as model_id:
        save_chat_config(admin, VISUALIZATION_PROMPT_TEMPLATE="")
        default_description = offered(person, upstream, model_id)[TOOL]["description"]

        saved = save_chat_config(admin, VISUALIZATION_PROMPT_TEMPLATE=own)
        assert saved["VISUALIZATION_PROMPT_TEMPLATE"] == own, saved
        own_description = offered(person, upstream, model_id)[TOOL]["description"]

        save_chat_config(admin, VISUALIZATION_PROMPT_TEMPLATE="   ")
        blank_description = offered(person, upstream, model_id)[TOOL]["description"]

    assert "chart" in default_description, (
        f"the tool did not carry the built-in prompt about drawing: {default_description[:200]!r}"
    )
    assert own_description == own, (
        f"the tool did not carry the admin's prompt: {own_description[:200]!r}"
    )
    assert blank_description == default_description, (
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


BAR_CHART = (
    '<div class="chart-container"><canvas id="tides"></canvas></div>'
    "<script>new Chart(document.getElementById('tides'), {type: 'bar', data: {labels: ['Mon',"
    " 'Tue'], datasets: [{label: 'High water (m)', data: [4.1, 4.6]}]}});</script>"
)


def visualizer_call(arguments: dict, call_id: str, through_call_tool: bool) -> dict:
    """The model's call of the tool, directly or dispatched through Tool Search's `call_tool`."""
    name, encoded = TOOL, json.dumps(arguments)
    if through_call_tool:
        name, encoded = "call_tool", json.dumps({"tool_name": TOOL, "params": encoded})
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": encoded}}


@pytest.mark.regression
@pytest.mark.parametrize("through_call_tool", [False, True], ids=["direct", "call_tool"])
def test_a_chart_drawn_without_requesting_chart_js_is_refused_and_the_retry_is_kept(
    admin, make_user, upstream, chat_config_restored, through_call_tool
):
    """A Chart.js chart only renders with libraries=["chartjs"] (open-webui/open-webui#34909)."""
    save_chat_config(
        admin,
        ENABLE_TOOL_SEARCH=through_call_tool,
        TOOL_SEARCH_DEFER_THRESHOLD=0,
        TOOL_SEARCH_ALWAYS_LOADED=[],
        TOOL_SEARCH_DEFER_BUILTIN_TOOLS=True,
    )
    person = make_user()
    question = f"chart the tide ({uuid.uuid4().hex[:8]})"
    answering = reply.answering(question)
    forgot = {"title": "Tide", "html": BAR_CHART}
    retried = {**forgot, "libraries": ["chartjs"]}
    upstream.queue(
        Reply(tool_calls=[visualizer_call(forgot, "call_1", through_call_tool)], match=answering),
        Reply(tool_calls=[visualizer_call(retried, "call_2", through_call_tool)], match=answering),
        Reply(content="There it is.", match=answering),
    )
    with preset(admin, {"visualizations": True}) as model_id, person.client() as client:
        _, message = ask(client, question, model=model_id)

    requests = [body for body in upstream.chat_requests() if answering(body)]
    assert len(requests) == 3, f"the model was asked {len(requests)} times, not 3"
    if through_call_tool:
        assert TOOL not in [tool["function"]["name"] for tool in requests[0]["tools"]]
    sent_back = {
        entry["tool_call_id"]: json.loads(entry["content"])
        for entry in reply.conversation(requests[2])
        if entry["role"] == "tool"
    }
    refusal = sent_back["call_1"]
    assert "status" not in refusal, f"the chart that cannot load Chart.js was accepted: {refusal}"
    assert "libraries" in refusal.get("error", "") and "chartjs" in refusal["error"], (
        f"the refusal does not tell the model to request chartjs: {refusal}"
    )
    assert sent_back["call_2"] == {"status": "accepted"}, sent_back["call_2"]

    output = without_hidden_calls(message.get("output") or [])
    calls = {item["call_id"]: item for item in output if item["type"] == "function_call"}
    results = {
        item["call_id"]: json.loads(item["output"][0]["text"])
        for item in output
        if item["type"] == "function_call_output"
    }
    assert calls["call_2"]["name"] == TOOL, calls
    assert json.loads(calls["call_2"]["arguments"]) == retried, calls["call_2"]
    assert results["call_2"] == {"status": "accepted"}, results
    assert "error" in results["call_1"], f"the stored reply shows the failed chart as {results}"
    assert "There it is." in str(output[-1]), output[-1]


@pytest.mark.regression
@pytest.mark.parametrize(
    "html",
    [
        "<h3>Tide Chart</h3><p>A new Chart (with spring tides) follows next week.</p>",
        '<div class="Chart chart-container"><svg viewBox="0 0 4 4"><rect width="2" height="3"/>'
        "</svg></div>",
        '<canvas id="c"></canvas><script>const pen = document.getElementById("c")'
        '.getContext("2d"); pen.fillRect(0, 0, 20, 40);</script>',
        "<p>Draw it with:</p><pre><code>new Chart(canvas, config)</code></pre>",
    ],
    ids=["chart in text", "chart class", "own canvas drawing", "chart code as text"],
)
def test_a_visual_without_chart_js_is_accepted_without_libraries(admin, make_user, upstream, html):
    person = make_user()
    with preset(admin, {"visualizations": True}) as model_id, person.client() as client:
        result = run_tool(client, upstream, TOOL, {"title": "Tide", "html": html}, model=model_id)

    assert json.loads(result) == {"status": "accepted"}, result


@pytest.mark.regression
@pytest.mark.parametrize(
    "html",
    [
        "<canvas id='c'></canvas><script>new window.Chart(document.body, {})</script>",
        "<canvas id='c'></canvas><script>\nnew Chart (document.getElementById('c'), {})",
    ],
    ids=["window.Chart", "unclosed script"],
)
def test_every_way_of_calling_chart_js_needs_the_library(admin, make_user, upstream, html):
    """open-webui/open-webui#34909: the prompt itself names window.Chart."""
    person = make_user()
    with preset(admin, {"visualizations": True}) as model_id, person.client() as client:
        result = run_tool(client, upstream, TOOL, {"title": "Tide", "html": html}, model=model_id)

    answer = json.loads(result)
    assert "status" not in answer, f"a chart that cannot load Chart.js was accepted: {answer}"
    assert "chartjs" in answer.get("error", ""), answer
