"""Journey: an admin turns Tool Call Labels on and replies name each tool call in plain words.

New in 0.13.1 (2416b1fc0, renamed in e8c0ad85e, both titled `refac`). The Tool Call Labels
switch in Admin Settings > Interface is off by default; switched on and saved it stays on after
a reload. A tool call the model labelled then shows its completed label where the reply would
say "View Result from <tool>", also after the chat is opened again, and the tool name stays on the
label as its hover text. Two labelled calls in a row show their labels, joined by a comma, in the
group summary. A call whose label is unusable shows as before.

Regression (open-webui/open-webui#33925): a group of three calls where the model left one call
unlabelled dropped every label from its summary, which read "Exploring 3 read_tide_gauge" while
the calls ran and "Explored 3 read_tide_gauge" after. The summary keeps the labels that exist,
running and completed, and names the unlabelled call by its tool. Red on dev until the fix merges.

Discriminates: passes on dev e8c0ad85e. In a backend copy that never keeps the labels on the
stored call the two label tests fail (the reply names the tool), and in one that never asks the
model for labels the admin test fails at the tools the model is offered, the label tests with it;
the unusable label control passes on both. The unlabelled call test fails on the dev 3c400952c
build (the running summary reads "Exploring 3 read_tide_gauge") and passes with the group summary
falling back to the tool name per unlabelled call.

The label tests run in the main chat and in a side chat, and the side variants are green on dev
c15f28649 and on the side chat rework build (open-webui/open-webui#34064). The admin settings
test stays on a page of its own: it is about the settings dialog.
"""

from __future__ import annotations

import json
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.python_tools import python_tool
from utils.cached_chat import turn_on_tool
from utils.chat_surface import ChatSurface
from utils.chat_ui import conversation, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
CHECKED = {"running": "Checking the harbour clock", "completed": "Checked the harbour clock"}
LOOKED = {"running": "Looking up the tide", "completed": "Looked up the tide"}
NORTH = {"running": "Reading the north gauge", "completed": "Read the north gauge"}
SOUTH = {"running": "Reading the south gauge", "completed": "Read the south gauge"}

GAUGE_TOOL = '''
import asyncio
import os


class Tools:
    async def read_tide_gauge(self, gauge: str) -> str:
        """Read the water level at a tide gauge.

        :param gauge: The gauge to read
        """
        for _ in range(600):
            if os.path.exists(RELEASE_FILE):
                break
            await asyncio.sleep(0.1)
        return f"gauge {gauge}: rising"
'''


@pytest.fixture
def labels_switch(admin, preserve):
    """`labels_switch(on)` sets the admin's Tool Call Labels switch over the API."""
    preserve(CHAT_CONFIG)

    def switch(on: bool) -> None:
        with admin.client() as client:
            current = client.get(CHAT_CONFIG[0]).json()
            saved = client.post(CHAT_CONFIG[1], json={**current, "ENABLE_TOOL_CALL_LABELS": on})
        assert saved.status_code == 200, saved.text

    return switch


def ask_for_the_time(chat: ChatSurface, upstream, *labels, answer: str = "It is noon.") -> str:
    """The model checks the time once per entry of `labels`, each call labelled with it."""
    question = f"what time is it? {uuid.uuid4().hex[:6]}"
    answering = reply.answering(question)
    calls = [
        {
            "id": f"call_{number}",
            "type": "function",
            "function": {
                "name": "get_current_timestamp",
                "arguments": json.dumps({"__labels__": label}),
            },
        }
        for number, label in enumerate(labels, start=1)
    ]
    upstream.queue(
        reply.Reply(tool_calls=calls, match=answering), reply.text(answer, match=answering)
    )
    send(chat, question)
    expect_reply(chat, answer)
    return question


def test_an_admin_switches_tool_call_labels_on_and_it_is_kept(
    labels_switch, page_for, admin, make_user, upstream
):
    labels_switch(False)
    page = page_for(admin, path="/admin/settings/interface")
    settings = page.get_by_role("dialog")
    switch = settings.get_by_role("switch", name="Tool Call Labels")
    expect(switch).not_to_be_checked()
    switch.click()
    expect(switch).to_be_checked()
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Settings saved successfully!")).to_be_visible()

    reopened = page_for(admin, path="/admin/settings/interface").get_by_role("dialog")
    expect(reopened.get_by_role("switch", name="Tool Call Labels")).to_be_checked()

    question = f"which tools are there? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Several.", match=reply.answering(question)))
    chat = page_for(make_user())
    send(chat, question)
    expect_reply(chat, "Several.")
    sent = next(body for body in upstream.chat_requests() if reply.answering(question)(body))
    labelled = [
        tool["function"]["name"]
        for tool in sent.get("tools") or []
        if "__labels__" in tool["function"]["parameters"].get("properties", {})
    ]
    assert labelled, f"the saved switch did not ask the model for labels: {sent.get('tools')}"


def test_a_labelled_call_shows_its_label_also_after_the_chat_is_opened_again(
    labels_switch, chat_for, make_user, upstream
):
    labels_switch(True)
    account = make_user()
    chat = chat_for(account)

    ask_for_the_time(chat, upstream, CHECKED)

    shown = conversation(chat).get_by_title("get_current_timestamp").locator("visible=true")
    expect(shown).to_have_text(CHECKED["completed"])
    expect(conversation(chat).get_by_text("View Result from get_current_timestamp")).to_have_count(
        0
    )

    reopened = chat.reload()
    reopened_label = conversation(reopened).get_by_title("get_current_timestamp")
    expect(reopened_label.locator("visible=true")).to_have_text(CHECKED["completed"])


def test_two_labelled_calls_in_a_row_show_both_labels_in_the_summary(
    labels_switch, chat_for, make_user, upstream
):
    labels_switch(True)
    chat = chat_for(make_user())

    ask_for_the_time(chat, upstream, CHECKED, LOOKED)

    summary = f"{CHECKED['completed']}, {LOOKED['completed']}"
    expect(conversation(chat).get_by_text(summary, exact=True)).to_be_visible()


@pytest.fixture
def held_gauges(admin, tmp_path):
    """A tide gauge tool whose reads all wait until the yielded release file exists."""
    release = tmp_path / "release"
    name = f"Tide gauges {uuid.uuid4().hex[:6]}"
    source = GAUGE_TOOL.replace("RELEASE_FILE", repr(str(release)))
    with python_tool(admin, source, name=name):
        yield name, release


def gauge_call(gauge: str, labels: dict | None) -> dict:
    arguments = {"gauge": gauge} if labels is None else {"__labels__": labels, "gauge": gauge}
    return {
        "id": f"call_{gauge}",
        "type": "function",
        "function": {"name": "read_tide_gauge", "arguments": json.dumps(arguments)},
    }


@pytest.mark.regression
def test_a_group_with_one_unlabelled_call_keeps_the_other_labels(
    labels_switch, chat_for, make_user, upstream, held_gauges
):
    tool_name, release = held_gauges
    labels_switch(True)
    chat = chat_for(make_user())
    turn_on_tool(chat, tool_name)
    question = f"read the tide gauges {uuid.uuid4().hex[:6]}"
    answering = reply.answering(question)
    calls = [gauge_call("north", NORTH), gauge_call("harbour", None), gauge_call("south", SOUTH)]
    upstream.queue(
        reply.Reply(tool_calls=calls, match=answering),
        reply.text("The tide is rising.", match=answering),
    )
    send(chat, question)

    summary = conversation(chat).get_by_role("button", name="Toggle details")
    try:
        expect(summary).to_contain_text(NORTH["running"])
        expect(summary).to_contain_text(SOUTH["running"])
        expect(summary).to_contain_text("read_tide_gauge")
    finally:
        release.touch()

    expect_reply(chat, "The tide is rising.")
    expect(summary).to_have_count(1)
    expect(summary).to_contain_text(NORTH["completed"])
    expect(summary).to_contain_text(SOUTH["completed"])
    expect(summary).to_contain_text("read_tide_gauge")


def test_a_call_with_an_unusable_label_shows_as_before(
    labels_switch, chat_for, make_user, upstream
):
    labels_switch(True)
    chat = chat_for(make_user())

    ask_for_the_time(chat, upstream, {"running": "", "completed": "x" * 121})

    expect(conversation(chat).get_by_text("View Result from get_current_timestamp")).to_be_visible()
