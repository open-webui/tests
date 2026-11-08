"""Journey: with Tool Call Labels on, the model labels each tool call and the reply keeps them.

New in 0.13.1 (2416b1fc0, renamed in e8c0ad85e, both titled `refac`). Admin Settings > Interface
has a Tool Call Labels switch, off by default and seeded by `ENABLE_TOOL_CALL_LABELS`. With it on,
every tool offered to the model gains a required `__labels__` argument asking for a short
`running` and `completed` label in the person's language. The tool runs without it; the labels go
onto the stored call as `display_labels`, trimmed, and a label that is empty or longer than 120
characters is dropped while the call still runs. With it off the tools are offered as they are.
A settings save that leaves the switch out keeps it. The model is sent its own call back with
the labels it wrote, in the same turn and the next (open-webui/open-webui#33952): before the fix
the replay carried the stripped arguments, so the model saw a call missing a required argument.

Discriminates: passes on dev e8c0ad85e. In a backend copy: never adding the labels argument fails
the offered, language and stored-labels tests, the language ignored fails the language test, the
labels left in the arguments fails the stored-labels and every dropped-labels case (the tool
refuses the extra argument), no length check
fails the long-label case, the keep-when-omitted rule dropped fails the omitted-save test and the
env default ignored fails the env test. The replay test passes on the fix branch
fix-replay-tool-call-labels and fails with labels on on dev b13d02e09 (the labels are missing
from the replayed call).
"""

from __future__ import annotations

import json
import uuid

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.python_tools import python_tool
from harness.tool_calls import offered_tool_specs, without_hidden_calls

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")

BUOY_TOOL = '''
class Tools:
    def check_buoy(self, buoy: str) -> str:
        """Read the light and battery of a harbour buoy.

        :param buoy: The buoy to check
        """
        return f"buoy {buoy}: light on, battery full"
'''

LABELS = {"running": "Checking the red buoy", "completed": "Checked the red buoy"}


@pytest.fixture
def labels_switch(admin, preserve):
    """`labels_switch(on)` sets the admin's Tool Call Labels switch for the rest of the test."""
    preserve(CHAT_CONFIG)

    def switch(on: bool) -> None:
        with admin.client() as client:
            current = client.get(CHAT_CONFIG[0]).json()
            saved = client.post(CHAT_CONFIG[1], json={**current, "ENABLE_TOOL_CALL_LABELS": on})
        assert saved.status_code == 200, saved.text
        assert saved.json()["ENABLE_TOOL_CALL_LABELS"] is on, saved.json()

    return switch


@pytest.fixture(scope="module")
def buoy_tool(admin):
    with python_tool(admin, BUOY_TOOL, name="Buoys") as tool_id:
        yield tool_id


def offered(make_user, upstream, buoy_tool: str, **options) -> dict[str, dict]:
    with make_user().client() as client:
        return offered_tool_specs(client, upstream, tool_ids=[buoy_tool], **options)


def labelled_call(make_user, upstream, buoy_tool: str, labels) -> tuple[dict, list[str]]:
    """The stored call and the tool results after the model checked a buoy with `labels`."""
    prompt = f"check the red buoy ({uuid.uuid4().hex[:8]})"
    answering = reply.answering(prompt)
    upstream.queue(
        reply.tool_call("check_buoy", {"__labels__": labels, "buoy": "red"}, match=answering),
        reply.text("The red buoy is fine.", match=answering),
    )
    with make_user().client() as client:
        _, message = ask(client, prompt, tool_ids=[buoy_tool])
    output = without_hidden_calls(message["output"])
    [call] = [item for item in output if item["type"] == "function_call"]
    results = [
        item["output"][0]["text"] for item in output if item["type"] == "function_call_output"
    ]
    return call, results


def test_with_labels_off_the_tools_are_offered_as_they_are(
    labels_switch, make_user, upstream, buoy_tool
):
    labels_switch(False)

    specs = offered(make_user, upstream, buoy_tool)

    assert {"check_buoy", "get_current_timestamp"} <= set(specs), set(specs)
    labelled = [name for name, spec in specs.items() if "__labels__" in json.dumps(spec)]
    assert labelled == [], f"tools were offered with labels while the switch is off: {labelled}"


def test_with_labels_on_every_offered_tool_asks_for_its_labels(
    labels_switch, make_user, upstream, buoy_tool
):
    labels_switch(True)

    specs = offered(make_user, upstream, buoy_tool)

    for name in ("check_buoy", "get_current_timestamp"):
        parameters = specs[name]["parameters"]
        asked = parameters["properties"].get("__labels__", {})
        assert asked.get("required") == ["running", "completed"], f"{name}: {parameters}"
        assert "__labels__" in parameters["required"], f"{name} does not require its labels"
    assert "buoy" in specs["check_buoy"]["parameters"]["properties"], "the tool lost its own input"
    assert (
        "in English" in specs["check_buoy"]["parameters"]["properties"]["__labels__"]["description"]
    )


def test_the_labels_are_asked_for_in_the_persons_language(
    labels_switch, make_user, upstream, buoy_tool
):
    labels_switch(True)

    specs = offered(make_user, upstream, buoy_tool, variables={"{{USER_LANGUAGE}}": "de-DE"})

    description = specs["check_buoy"]["parameters"]["properties"]["__labels__"]["description"]
    assert "in de-DE" in description, f"the labels were not asked for in German: {description}"


def test_the_tool_runs_without_its_labels_and_the_call_keeps_them(
    labels_switch, make_user, upstream, buoy_tool
):
    labels_switch(True)
    padded = {key: f"  {value}  " for key, value in LABELS.items()}

    call, results = labelled_call(make_user, upstream, buoy_tool, padded)

    assert results == ["buoy red: light on, battery full"], f"the tool did not run: {results}"
    assert call.get("display_labels") == LABELS, f"the stored call: {call}"
    assert "__labels__" not in call["arguments"], f"the labels stayed in the input: {call}"


@pytest.mark.parametrize(
    "labels",
    [
        {"running": "", "completed": "Checked the red buoy"},
        {"running": "Checking the red buoy", "completed": "x" * 121},
        "Checking the red buoy",
    ],
    ids=["empty", "too_long", "not_an_object"],
)
def test_unusable_labels_are_dropped_and_the_call_still_runs(
    labels, labels_switch, make_user, upstream, buoy_tool
):
    labels_switch(True)

    call, results = labelled_call(make_user, upstream, buoy_tool, labels)

    assert results == ["buoy red: light on, battery full"], f"the tool did not run: {results}"
    assert "display_labels" not in call, f"unusable labels were kept: {call}"


def test_a_save_without_the_switch_keeps_it(labels_switch, admin):
    labels_switch(True)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        current.pop("ENABLE_TOOL_CALL_LABELS")
        saved = client.post(CHAT_CONFIG[1], json=current)

    assert saved.status_code == 200, saved.text
    assert saved.json()["ENABLE_TOOL_CALL_LABELS"] is True, (
        "a save leaving the switch out cleared it"
    )


@pytest.mark.slow
def test_the_env_var_switches_labels_on_for_a_fresh_instance(instance_with):
    launched = instance_with({"ENABLE_TOOL_CALL_LABELS": "true"})
    with launched.client() as client:
        seeded = client.get(CHAT_CONFIG[0])

    assert seeded.status_code == 200, seeded.text
    assert seeded.json()["ENABLE_TOOL_CALL_LABELS"] is True


@pytest.mark.parametrize("labels_on", [True, False], ids=["labels_on", "labels_off"])
def test_the_model_is_sent_its_own_call_back_as_it_made_it(
    labels_on, labels_switch, make_user, upstream, buoy_tool
):
    """open-webui/open-webui#33952: the replayed call lost the required labels the model sent."""
    labels_switch(labels_on)
    arguments = {"__labels__": LABELS, "buoy": "red"} if labels_on else {"buoy": "red"}
    tag = uuid.uuid4().hex[:8]
    prompt, follow_up = f"check the red buoy ({tag})", f"and now? ({tag})"
    upstream.queue(
        reply.tool_call("check_buoy", arguments, match=reply.answering(prompt)),
        reply.text("The red buoy is fine.", match=reply.answering(prompt)),
        reply.text("Still fine.", match=reply.answering(follow_up)),
    )
    with make_user().client() as client:
        first, _ = ask(client, prompt, tool_ids=[buoy_tool])
        ask(
            client,
            follow_up,
            chat_id=first.chat_id,
            parent_id=first.assistant_message_id,
            tool_ids=[buoy_tool],
        )

    for asking in (prompt, follow_up):
        replayed = [
            json.loads(call["function"]["arguments"])
            for body in upstream.chat_requests()
            if reply.answering(asking)(body)
            for message in body["messages"]
            for call in message.get("tool_calls") or []
            if call["function"]["name"] == "check_buoy"
        ]
        assert replayed == [arguments], (
            f"after {asking!r} the model got its call back as {replayed}"
        )
