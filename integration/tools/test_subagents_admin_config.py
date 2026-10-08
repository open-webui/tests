"""Journey: the sub-agent settings decide what the model is offered and how a sub-agent runs.

The delegation tool (and the timer that shares its switch) is offered to an ordinary account only
while sub-agents are on, and not to a model whose Sub-agents tick is off, whether the tick sits on
the model or in the models' default settings. Once on, the saved settings shape each sub-agent:
the saved system prompt follows the parent's own (the built-in prompt when it is empty), the max
output cuts a long answer short before it reaches the parent, and the max iterations stops a
sub-agent that keeps calling tools.

`test_a_larger_max_iterations_lets_the_sub_agent_finish`,
`test_an_answer_within_the_max_output_reaches_the_parent_whole`,
`test_an_empty_system_prompt_gives_the_sub_agent_the_built_in_one`,
`test_the_max_iterations_stops_a_sub_agent_that_keeps_calling_tools`,
`test_the_max_output_cuts_a_long_answer_before_it_reaches_the_parent` and
`test_the_saved_system_prompt_follows_the_parents_own_in_a_sub_agent` are red on dev 62f70a844:
since de73bb830 a chat request whose reply message is already stored in the chat, the way
automations, sub-agents and timers prepare their reply, is refused with 409 and the reply is never
written (open-webui/open-webui#32066).

Discriminates: passes on dev 176d31d1d; in a backend copy that offers the delegation tool whatever
the setting says, or whatever the model's tick says, or that ignores the default settings' tick,
the availability tests fail, and with the system prompt, the output cap or the iteration cap
replaced by a constant the matching test fails.
"""

from __future__ import annotations

import uuid
from typing import Iterator

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.python_tools import EVERYONE_READS
from harness.tool_calls import offered_tools
from harness.upstream import MOCK_MODEL_ID, Reply

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
MODELS_CONFIG = ("/api/v1/configs/models", "/api/v1/configs/models")
DELEGATION = {"delegate_task", "timer"}
PARENT_PROMPT = "You are the harbour master."
BUILT_IN_PROMPT = "You are a sub-agent working on a specific task"


def _configure(admin, **settings) -> None:
    with admin.client() as client:
        current = client.get(SUBAGENTS[0]).json()
        saved = client.post(SUBAGENTS[1], json={**current, **settings})
    assert saved.status_code == 200, saved.text


@pytest.fixture
def sub_agents_on(preserve, admin):
    preserve(SUBAGENTS)
    _configure(admin, ENABLE_SUBAGENTS=True)


@pytest.fixture
def helper_model(admin) -> Iterator[str]:
    """A model every account may use, whose builtin tools are left as they come."""
    model_id = f"helper-{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": model_id,
                "meta": {"builtinTools": {"subagents": False}},
                "params": {},
                "access_grants": [EVERYONE_READS],
            },
        )
        assert created.status_code == 200, created.text
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_an_ordinary_account_is_offered_delegation_while_sub_agents_are_on(
    sub_agents_on, make_user, upstream
):
    with make_user().client() as client:
        offered = offered_tools(client, upstream)

    assert DELEGATION <= offered


def test_an_ordinary_account_is_offered_no_delegation_while_sub_agents_are_off(
    preserve, admin, make_user, upstream
):
    preserve(SUBAGENTS)
    _configure(admin, ENABLE_SUBAGENTS=False)
    with make_user().client() as client:
        offered = offered_tools(client, upstream)

    assert offered, "the model was offered no tools at all, so this shows nothing"
    assert not DELEGATION & offered


def test_a_model_with_sub_agents_unticked_is_offered_no_delegation(
    sub_agents_on, helper_model, make_user, upstream
):
    with make_user().client() as client:
        without = offered_tools(client, upstream, model=helper_model)
        plain = offered_tools(client, upstream)

    assert without, "the model was offered no tools at all, so this shows nothing"
    assert not DELEGATION & without
    assert DELEGATION <= plain


@pytest.fixture
def default_settings_unticked(admin) -> Iterator[None]:
    """The models' default settings with Sub-agents unticked, put back afterwards."""
    with admin.client() as client:
        original = client.get(MODELS_CONFIG[0]).json()
        metadata = original["DEFAULT_MODEL_METADATA"] or {}
        untick = {"builtinTools": {**metadata.get("builtinTools", {}), "subagents": False}}
        changed = {**original, "DEFAULT_MODEL_METADATA": {**metadata, **untick}}
        assert client.post(MODELS_CONFIG[1], json=changed).status_code == 200
        # the next page load rebuilds the model list the chat reads
        client.get("/api/models").raise_for_status()
        yield
        assert client.post(MODELS_CONFIG[1], json=original).status_code == 200
        client.get("/api/models").raise_for_status()


def test_sub_agents_unticked_in_the_default_model_settings_withdraws_delegation_everywhere(
    sub_agents_on, default_settings_unticked, make_user, upstream
):
    with make_user().client() as client:
        offered = offered_tools(client, upstream)

    assert offered, "the model was offered no tools at all, so this shows nothing"
    assert not DELEGATION & offered


def _delegation(task: str) -> Reply:
    return reply.tool_call("delegate_task", {"task": task})


def _sub_agent_requests(upstream, task: str) -> list[dict]:
    return [request for request in upstream.chat_requests() if reply.answering(task)(request)]


def _tool_results(request: dict) -> list[str]:
    return [entry["content"] for entry in request["messages"] if entry["role"] == "tool"]


def _system_messages(request: dict) -> list[str]:
    return [entry["content"] for entry in request["messages"] if entry["role"] == "system"]


def test_the_saved_system_prompt_follows_the_parents_own_in_a_sub_agent(
    sub_agents_on, admin, make_user, upstream
):
    saved_prompt = f"Reply in one line, batch {uuid.uuid4().hex[:6]}."
    _configure(admin, SUBAGENTS_SYSTEM_PROMPT=saved_prompt)
    task = f"count the boats {uuid.uuid4().hex[:6]}"
    upstream.queue(
        _delegation(task),
        reply.text("Three boats.", match=reply.answering(task)),
        reply.text("The helper counted three."),
    )
    with make_user().client() as client:
        ask(client, "how many boats are in?", params={"system": PARENT_PROMPT})

    [sub_agent] = _sub_agent_requests(upstream, task)
    [system] = _system_messages(sub_agent)
    assert system == f"{PARENT_PROMPT}\n\n{saved_prompt}"


def test_an_empty_system_prompt_gives_the_sub_agent_the_built_in_one(
    sub_agents_on, admin, make_user, upstream
):
    _configure(admin, SUBAGENTS_SYSTEM_PROMPT="")
    task = f"count the buoys {uuid.uuid4().hex[:6]}"
    upstream.queue(
        _delegation(task),
        reply.text("Four buoys.", match=reply.answering(task)),
        reply.text("The helper counted four."),
    )
    with make_user().client() as client:
        ask(client, "how many buoys are out?", params={"system": PARENT_PROMPT})

    [sub_agent] = _sub_agent_requests(upstream, task)
    [system] = _system_messages(sub_agent)
    assert system.startswith(f"{PARENT_PROMPT}\n\n{BUILT_IN_PROMPT}")


def test_the_max_output_cuts_a_long_answer_before_it_reaches_the_parent(
    sub_agents_on, admin, make_user, upstream
):
    _configure(admin, SUBAGENTS_MAX_OUTPUT=1000)
    task = f"write the harbour log {uuid.uuid4().hex[:6]}"
    upstream.queue(
        _delegation(task),
        reply.text("L" * 1500, match=reply.answering(task)),
        reply.text("The log is in."),
    )
    with make_user().client() as client:
        ask(client, "have the log written")

    [result] = _tool_results(upstream.chat_requests()[-1])
    assert result == f"{'L' * 1000}\n\n[output truncated]"


def test_an_answer_within_the_max_output_reaches_the_parent_whole(
    sub_agents_on, admin, make_user, upstream
):
    _configure(admin, SUBAGENTS_MAX_OUTPUT=1000)
    task = f"write the short log {uuid.uuid4().hex[:6]}"
    upstream.queue(
        _delegation(task),
        reply.text("L" * 900, match=reply.answering(task)),
        reply.text("The log is in."),
    )
    with make_user().client() as client:
        ask(client, "have the short log written")

    assert _tool_results(upstream.chat_requests()[-1]) == ["L" * 900]


def test_the_max_iterations_stops_a_sub_agent_that_keeps_calling_tools(
    sub_agents_on, admin, make_user, upstream
):
    _configure(admin, SUBAGENTS_MAX_ITERATIONS=2)
    task = f"watch the tide {uuid.uuid4().hex[:6]}"
    looping = [
        reply.tool_call(
            "get_current_timestamp", {}, call_id=f"call_{n}", match=reply.answering(task)
        )
        for n in range(5)
    ]
    upstream.queue(_delegation(task), *looping, reply.text("The helper gave up."))
    with make_user().client() as client:
        ask(client, "have someone watch the tide")

    assert len(_sub_agent_requests(upstream, task)) == 3
    [result] = _tool_results(upstream.chat_requests()[-1])
    assert "Tool-call limit reached (2 iterations)" in result


def test_a_larger_max_iterations_lets_the_sub_agent_finish(
    sub_agents_on, admin, make_user, upstream
):
    _configure(admin, SUBAGENTS_MAX_ITERATIONS=5)
    task = f"read the tide gauge {uuid.uuid4().hex[:6]}"
    upstream.queue(
        _delegation(task),
        *[
            reply.tool_call(
                "get_current_timestamp", {}, call_id=f"call_{n}", match=reply.answering(task)
            )
            for n in range(3)
        ],
        reply.text("The tide is rising.", match=reply.answering(task)),
        reply.text("The helper reports a rising tide."),
    )
    with make_user().client() as client:
        ask(client, "have someone read the gauge")

    assert len(_sub_agent_requests(upstream, task)) == 4
    assert _tool_results(upstream.chat_requests()[-1]) == ["The tide is rising."]
