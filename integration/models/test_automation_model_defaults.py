"""Regression: an automation's first run on a cold model cache went out without its model's tools.

Fix `60ded561a` (open-webui/open-webui#30379, issue open-webui/open-webui#27694): an automation
read its model's tools, default features and filters from the in-memory model list before
building the request, and that list is empty after a restart or after an admin saves the
connection settings until something reloads it. The first run then reached the model without
the tools bound to it, and a channel automation introduced the model by its id instead of its
name. A run now loads the model list first when it is empty.

Saving the OpenAI connections unchanged empties the list, as a restart does; the run is started
with Run now right after.

`test_a_run_on_a_warm_cache_offers_the_models_tools` and
`test_the_first_run_on_a_cold_cache_offers_the_models_tools` are red on dev 62f70a844: since
de73bb830 a chat request whose reply message is already stored in the chat, the way automations,
sub-agents and timers prepare their reply, is refused with 409 and the reply is never written
(open-webui/open-webui#32066, fixed by 641c82df1).

Discriminates: passes on dev efe63bd34; with 60ded561a reverted in a backend copy both cold-cache
tests fail (no tools offered, and the channel run says "You are <model id>"). The warm-cache test
passes on both.

Default tools for all models (#33749, 870107b34): tools an admin sets in Model Defaults are the
tools of every model without a tool list of its own, in the model list the chat page reads and in
an automation run on it; a model whose own list is empty, every tool unticked, keeps it. Passes on
dev f659c0d12; in backend copies of it, leaving the default tools out of the model list fails the
first test, and letting the defaults replace an empty list fails the second.

Client IP placeholders (#33833, d39cb17b9) are meant to be empty for a run with no client. Red on
dev f659c0d12: a run sends 127.0.0.1, the address of the request automations build for
themselves (issue draft real-nf-admin-2). Reverting d39cb17b9 also turns it red, with the
placeholder sent as written.
"""

from __future__ import annotations

import time
import uuid

import pytest

from harness import upstream as reply
from harness.channel_quotes import enable_channels
from harness.python_tools import python_tool
from harness.second_provider import OPENAI_CONFIG
from harness.upstream import MOCK_MODEL_ID, conversation

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

SCHEDULE = "DTSTART:20990101T090000\nRRULE:FREQ=DAILY"
PRESET_NAME = "Forecast model"
RUN_WAIT = 30.0

FORECAST_TOOL = '''
class Tools:
    def get_forecast(self, city: str) -> str:
        """
        Tomorrow's weather for a city.

        :param city: The city.
        """
        return "sunny"
'''


@pytest.fixture
def scheduler(make_user):
    """A fresh admin, whose automations are deleted afterwards so none runs on its own later."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for automation in client.get("/api/v1/automations/list").json().get("items", []):
            client.delete(f"/api/v1/automations/{automation['id']}/delete")


@pytest.fixture
def forecast_model(admin, scheduler):
    """A preset on the scripted model with a workspace tool bound to it; yields its id."""
    with python_tool(admin, FORECAST_TOOL, name="Forecast") as tool_id:
        model_id = f"forecast-{uuid.uuid4().hex[:8]}"
        with scheduler.client() as client:
            created = client.post(
                "/api/v1/models/create",
                json={
                    "id": model_id,
                    "base_model_id": MOCK_MODEL_ID,
                    "name": PRESET_NAME,
                    "meta": {"toolIds": [tool_id]},
                    "params": {},
                },
            )
            assert created.status_code == 200, created.text
            try:
                yield model_id
            finally:
                client.post("/api/v1/models/model/delete", json={"id": model_id})


def _empty_model_cache(admin, preserve) -> None:
    """Save the OpenAI connections as they are, which drops the in-memory model list."""
    preserve(OPENAI_CONFIG)
    with admin.client() as client:
        current = client.get(OPENAI_CONFIG[0]).json()
        client.post(OPENAI_CONFIG[1], json=current).raise_for_status()


def _run_now(owner, model_id: str, prompt: str, target: dict | None = None) -> None:
    data = {"prompt": prompt, "model_id": model_id, "rrule": SCHEDULE}
    if target:
        data["target"] = target
    with owner.client() as client:
        created = client.post(
            "/api/v1/automations/create",
            json={"name": f"forecast {uuid.uuid4().hex[:6]}", "is_active": False, "data": data},
        )
        assert created.status_code == 200, created.text
        started = client.post(f"/api/v1/automations/{created.json()['id']}/run")
        assert started.status_code == 200, started.text


def _request_for(upstream, prompt: str) -> dict:
    deadline = time.monotonic() + RUN_WAIT
    while time.monotonic() < deadline:
        for request in upstream.chat_requests():
            if prompt in str(conversation(request)[-1].get("content")):
                return request
        time.sleep(0.2)
    raise AssertionError(f"the automation never reached the model with {prompt!r}")


def _offered(request: dict) -> set[str]:
    return {tool["function"]["name"] for tool in request.get("tools") or []}


def test_the_first_run_on_a_cold_cache_offers_the_models_tools(
    admin, preserve, scheduler, forecast_model, upstream
):
    prompt = f"Will it rain tomorrow? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("No rain.", match=reply.answering(prompt)))
    _empty_model_cache(admin, preserve)

    _run_now(scheduler, forecast_model, prompt)

    offered = _offered(_request_for(upstream, prompt))
    assert "get_forecast" in offered, (
        f"the run on an empty model list was offered {sorted(offered)}, without the tool bound "
        "to its model (#27694)"
    )


def test_a_channel_run_on_a_cold_cache_names_the_model(
    admin, preserve, scheduler, forecast_model, upstream
):
    preserve("admin_config")
    enable_channels(admin)
    with scheduler.client() as client:
        channel = client.post(
            "/api/v1/channels/create",
            json={"name": f"weather-{uuid.uuid4().hex[:6]}", "type": None},
        )
    assert channel.status_code == 200, channel.text
    prompt = f"Forecast for the team {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Sunny all day.", match=reply.answering(prompt)))
    _empty_model_cache(admin, preserve)

    _run_now(
        scheduler, forecast_model, prompt, {"type": "channel", "channel_id": channel.json()["id"]}
    )

    request = _request_for(upstream, prompt)
    system = request["messages"][0]["content"]
    assert f"You are {PRESET_NAME}" in system, (
        f"the channel run introduced the model by its id: {system!r} (#27694)"
    )
    assert "get_forecast" in _offered(request)


def test_a_run_on_a_warm_cache_offers_the_models_tools(scheduler, forecast_model, upstream):
    prompt = f"Is it windy? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("A breeze.", match=reply.answering(prompt)))
    with scheduler.client() as client:
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()

    _run_now(scheduler, forecast_model, prompt)

    assert "get_forecast" in _offered(_request_for(upstream, prompt))


# --------------------------------------------------------------------------- Model Defaults tools

MODELS_CONFIG = ("/api/v1/configs/models", "/api/v1/configs/models")


@pytest.fixture
def default_forecast_tool(admin, preserve):
    """The forecast tool set as a default tool for every model in Model Defaults; yields its id."""
    preserve(MODELS_CONFIG)
    with python_tool(admin, FORECAST_TOOL, name="Forecast default") as tool_id:
        with admin.client() as client:
            current = client.get(MODELS_CONFIG[0]).json()
            metadata = {**(current.get("DEFAULT_MODEL_METADATA") or {}), "toolIds": [tool_id]}
            saved = client.post(
                MODELS_CONFIG[1], json={**current, "DEFAULT_MODEL_METADATA": metadata}
            )
            assert saved.status_code == 200, saved.text
        yield tool_id


@pytest.fixture
def preset(scheduler):
    """`preset(meta)` makes a preset of the scheduler's on the scripted model; deleted after."""
    made = []

    def create(meta: dict) -> str:
        model_id = f"defaults-{uuid.uuid4().hex[:8]}"
        with scheduler.client() as client:
            created = client.post(
                "/api/v1/models/create",
                json={
                    "id": model_id,
                    "base_model_id": MOCK_MODEL_ID,
                    "name": f"Defaults {model_id}",
                    "meta": meta,
                    "params": {},
                },
            )
        assert created.status_code == 200, created.text
        made.append(model_id)
        return model_id

    yield create
    with scheduler.client() as client:
        for model_id in made:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def _listed_tool_ids(owner, model_id: str) -> list[str] | None:
    with owner.client() as client:
        models = client.get("/api/models", params={"refresh": "true"}).json()["data"]
    model = next(model for model in models if model["id"] == model_id)
    return model.get("info", {}).get("meta", {}).get("toolIds")


def test_a_model_without_tools_of_its_own_gets_the_default_tools(
    scheduler, preset, default_forecast_tool, upstream
):
    plain = preset({})
    assert _listed_tool_ids(scheduler, plain) == [default_forecast_tool], (
        "the model list does not give a model without tools the Model Defaults tools"
    )

    prompt = f"Frost tonight? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("No frost.", match=reply.answering(prompt)))
    _run_now(scheduler, plain, prompt)
    offered = _offered(_request_for(upstream, prompt))
    assert "get_forecast" in offered, (
        f"a run on a model without tools of its own was offered {sorted(offered)}, "
        "without the Model Defaults tool"
    )


def test_a_model_with_its_own_tool_list_keeps_it_over_the_defaults(
    scheduler, preset, default_forecast_tool, upstream
):
    # an empty list is a choice too: the admin unticked every tool for this model
    unticked = preset({"toolIds": []})
    assert _listed_tool_ids(scheduler, unticked) == [], (
        "the Model Defaults tools replaced a model's own empty tool list"
    )

    prompt = f"Fog at dawn? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Some fog.", match=reply.answering(prompt)))
    _run_now(scheduler, unticked, prompt)
    offered = _offered(_request_for(upstream, prompt))
    assert "get_forecast" not in offered, (
        f"a model whose own tool list is empty was offered the default tool: {sorted(offered)}"
    )


def test_a_run_has_no_client_address_for_the_ip_placeholders(
    admin, preserve, scheduler, forecast_model, upstream
):
    preserve(OPENAI_CONFIG)
    with admin.client() as client:
        config = client.get(OPENAI_CONFIG[0]).json()
        connection = config["OPENAI_API_CONFIGS"].get("0", {})
        headers = {"X-Real-IP": "{{USER_IP}}", "X-Client-Chain": "{{X_FORWARDED_FOR}}"}
        config["OPENAI_API_CONFIGS"] = {"0": {**connection, "headers": headers}}
        client.post(OPENAI_CONFIG[1], json=config).raise_for_status()
    prompt = f"Hail this week? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("No hail.", match=reply.answering(prompt)))

    _run_now(scheduler, forecast_model, prompt)

    body = _request_for(upstream, prompt)
    sent = next(entry for entry in upstream.requests_to("/chat/completions") if entry.body is body)
    received = {name.lower(): value for name, value in sent.headers.items()}
    # a scheduled run has no client, so no address may stand in for one (#33833)
    assert received.get("x-real-ip", "") == "", (
        f"a run sent {{{{USER_IP}}}} as {received.get('x-real-ip')!r}"
    )
    assert received.get("x-client-chain", "") == "", received.get("x-client-chain")
