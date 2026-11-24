"""Journey: the sub-agent settings decide what the model is offered and how a sub-agent runs.

The delegation tool (and the timer that shares its switch) is offered to an ordinary account only
while sub-agents are on, and not to a model whose Sub-agents tick is off, whether the tick sits on
the model or in the models' default settings. Once on, the saved settings shape each sub-agent:
the saved system prompt follows the parent's own system message on the next line (the built-in
prompt when it is empty) and names the lead's chat, the max output cuts a long answer short
before it reaches the parent, and the max iterations stops a sub-agent that keeps calling tools.

Since dev 8522af2d6 a chat's sub-agent reports back after the reply that delegated has ended, as a
message of its own to the lead model, so the tests wait for that report and read the answer from
it; since then the sub-agent's system prompt also ends with the id of the lead's chat, which it
messages with `send_message`.

Since #34040 (2c8fa823f) the Sub-agents switch under Features in the user permissions decides who
gets them: with it off in the default permissions an ordinary account is offered neither the
delegation tool nor the timer, a group that switches it on gives them back to its members, an
admin keeps them, and USER_PERMISSIONS_FEATURES_SUBAGENTS=false starts an instance with it off.

From de73bb830 the sub-agent tests were red: a chat request whose reply message was already
stored in the chat, the way sub-agents prepare their reply, was refused with 409
(open-webui/open-webui#32066, fixed by 641c82df1).

Discriminates: passes on dev 79d376a1f; in a backend copy that offers the delegation tool whatever
the setting says, or whatever the model's tick says, or that ignores the default settings' tick,
the availability tests fail, and with the system prompt, the output cap or the iteration cap
replaced by a constant the matching test fails. The permission tests pass on dev 485103cde;
in a backend copy whose tool gate skips the subagents permission the default-off and env tests
fail, and in one whose permission check ignores groups the group test fails.
"""

from __future__ import annotations

import uuid
from typing import Iterator

import pytest

from harness import upstream as reply
from harness.access import make_group
from harness.actors import create_user
from harness.chat import ask
from harness.python_tools import EVERYONE_READS
from harness.subagents import delegated, report_content, wait_for_report
from harness.tool_calls import offered_tools
from harness.upstream import MOCK_MODEL_ID, Reply

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
MODELS_CONFIG = ("/api/v1/configs/models", "/api/v1/configs/models")
DEFAULT_PERMISSIONS = "/api/v1/users/default/permissions"
DELEGATION = {"delegate_task", "timer"}
PARENT_PROMPT = "You are the harbour master."
# the chat's system prompt, sent the way the web UI sends it
PARENT_SYSTEM = [{"role": "system", "content": PARENT_PROMPT}]
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
def sub_agents_withheld_by_default(sub_agents_on, preserve, admin) -> None:
    """Sub-agents on for the instance, the Sub-agents permission off in the defaults."""
    preserve("permissions")
    with admin.client() as client:
        current = client.get(DEFAULT_PERMISSIONS).json()
        features = {**current["features"], "subagents": False}
        saved = client.post(DEFAULT_PERMISSIONS, json={**current, "features": features})
    assert saved.status_code == 200, saved.text


def test_an_account_without_the_sub_agents_permission_is_offered_no_delegation(
    sub_agents_withheld_by_default, make_user, upstream
):
    with make_user().client() as client:
        offered = offered_tools(client, upstream)

    assert offered, "the model was offered no tools at all, so this shows nothing"
    assert not DELEGATION & offered, f"an account without the permission was offered {offered}"


def test_a_group_with_the_sub_agents_permission_gives_delegation_back(
    sub_agents_withheld_by_default, admin, make_user, upstream
):
    member = make_user()
    group_id = make_group(admin, [member], {"features": {"subagents": True}})
    try:
        with member.client() as client:
            offered = offered_tools(client, upstream)
    finally:
        with admin.client() as client:
            client.delete(f"/api/v1/groups/id/{group_id}/delete")

    assert DELEGATION <= offered, f"a member of a group allowed sub-agents was offered {offered}"


def test_an_admin_keeps_delegation_without_the_sub_agents_permission(
    sub_agents_withheld_by_default, make_user, upstream
):
    with make_user(role="admin").client() as client:
        offered = offered_tools(client, upstream)

    assert DELEGATION <= offered, f"an admin was offered {offered}"


@pytest.mark.slow
def test_the_environment_can_start_the_sub_agents_permission_off(instance_with):
    launched = instance_with(
        {"ENABLE_SUBAGENTS": "true", "USER_PERMISSIONS_FEATURES_SUBAGENTS": "false"}
    )
    account = create_user(launched)
    with account.client() as client:
        permissions = client.get("/api/v1/users/permissions").json()
        offered = offered_tools(client, launched.upstream)

    assert permissions["features"]["subagents"] is False, permissions["features"]
    assert offered, "the model was offered no tools at all, so this shows nothing"
    assert not DELEGATION & offered, f"an account without the permission was offered {offered}"


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
    return [request for request in upstream.chat_requests() if delegated(task)(request)]


def _reported_result(upstream, task: str) -> str:
    """The sub-agent's answer as its report hands it to the lead model."""
    report = report_content(wait_for_report(upstream, task))
    _, marker, result = report.partition("--- RESULT ---\n")
    assert marker, f"the report carries no result: {report}"
    return result


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
        reply.text("Three boats.", match=delegated(task)),
        reply.text("The helper counted three."),
    )
    with make_user().client() as client:
        turn, _ = ask(client, "how many boats are in?", history=PARENT_SYSTEM)
    wait_for_report(upstream, task)

    [sub_agent] = _sub_agent_requests(upstream, task)
    [system] = _system_messages(sub_agent)
    # skills other tests shared with everyone are listed after the prompt
    prompt, _, _ = system.partition("\n<available_skills>")
    own, _, lead = prompt.partition("\n\n")
    assert own == f"{PARENT_PROMPT}\n{saved_prompt}"
    assert turn.chat_id in lead, f"the sub-agent was not told the lead's chat: {prompt}"


def test_an_empty_system_prompt_gives_the_sub_agent_the_built_in_one(
    sub_agents_on, admin, make_user, upstream
):
    _configure(admin, SUBAGENTS_SYSTEM_PROMPT="")
    task = f"count the buoys {uuid.uuid4().hex[:6]}"
    upstream.queue(
        _delegation(task),
        reply.text("Four buoys.", match=delegated(task)),
        reply.text("The helper counted four."),
    )
    with make_user().client() as client:
        ask(client, "how many buoys are out?", history=PARENT_SYSTEM)
    wait_for_report(upstream, task)

    [sub_agent] = _sub_agent_requests(upstream, task)
    [system] = _system_messages(sub_agent)
    assert system.startswith(f"{PARENT_PROMPT}\n{BUILT_IN_PROMPT}")


def test_the_max_output_cuts_a_long_answer_before_it_reaches_the_parent(
    sub_agents_on, admin, make_user, upstream
):
    _configure(admin, SUBAGENTS_MAX_OUTPUT=1000)
    task = f"write the harbour log {uuid.uuid4().hex[:6]}"
    upstream.queue(
        _delegation(task),
        reply.text("L" * 1500, match=delegated(task)),
        reply.text("The log is in."),
    )
    with make_user().client() as client:
        ask(client, "have the log written")

    assert _reported_result(upstream, task) == f"{'L' * 1000}\n\n[output truncated]"


def test_an_answer_within_the_max_output_reaches_the_parent_whole(
    sub_agents_on, admin, make_user, upstream
):
    _configure(admin, SUBAGENTS_MAX_OUTPUT=1000)
    task = f"write the short log {uuid.uuid4().hex[:6]}"
    # exactly at the limit: a cut at `>=` would shorten it
    upstream.queue(
        _delegation(task),
        reply.text("L" * 1000, match=delegated(task)),
        reply.text("The log is in."),
    )
    with make_user().client() as client:
        ask(client, "have the short log written")

    assert _reported_result(upstream, task) == "L" * 1000


def test_the_max_iterations_stops_a_sub_agent_that_keeps_calling_tools(
    sub_agents_on, admin, make_user, upstream
):
    _configure(admin, SUBAGENTS_MAX_ITERATIONS=2)
    task = f"watch the tide {uuid.uuid4().hex[:6]}"
    looping = [
        reply.tool_call("get_current_timestamp", {}, call_id=f"call_{n}", match=delegated(task))
        for n in range(5)
    ]
    upstream.queue(_delegation(task), *looping, reply.text("The helper gave up."))
    with make_user().client() as client:
        ask(client, "have someone watch the tide")

    report = report_content(wait_for_report(upstream, task))
    assert len(_sub_agent_requests(upstream, task)) == 3
    # worded "Tool-call limit reached (2 iterations)" before dev d87837372
    assert "Status: error" in report and report.endswith("Tool-call iteration limit reached."), (
        report
    )


def test_a_larger_max_iterations_lets_the_sub_agent_finish(
    sub_agents_on, admin, make_user, upstream
):
    _configure(admin, SUBAGENTS_MAX_ITERATIONS=5)
    task = f"read the tide gauge {uuid.uuid4().hex[:6]}"
    upstream.queue(
        _delegation(task),
        *[
            reply.tool_call("get_current_timestamp", {}, call_id=f"call_{n}", match=delegated(task))
            for n in range(3)
        ],
        reply.text("The tide is rising.", match=delegated(task)),
        reply.text("The helper reports a rising tide."),
    )
    with make_user().client() as client:
        ask(client, "have someone read the gauge")

    assert _reported_result(upstream, task) == "The tide is rising."
    assert len(_sub_agent_requests(upstream, task)) == 4
