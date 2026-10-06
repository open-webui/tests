"""Journey: the status updates a tool sends show above its reply, with their history on a click.

A workspace tool reports what it is doing through its event emitter. The reply shows the latest
status above its text, the Toggle status history button opens every status the tool sent, in
order, and both are still there when the chat is opened again. A model with Status Updates
unticked in its capabilities shows none of them.

Discriminates: passes on dev 30f3f6a8f; in a backend copy that keeps no status updates with the
reply the reload test fails, in one that drops every status event the other two fail as well, and
in a frontend build that ignores the Status Updates capability the unticked test fails.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.python_tools import EVERYONE_READS, python_tool
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import conversation, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

GAUGE_TOOL = """class Tools:
    async def read_tide_gauge(self, harbour: str, __event_emitter__=None) -> str:
        \"\"\"Read the tide gauge of a harbour.

        :param harbour: the harbour
        \"\"\"
        await __event_emitter__(
            {"type": "status", "data": {"description": f"Walking to the gauge at {harbour}"}}
        )
        await __event_emitter__(
            {"type": "status", "data": {"description": "Gauge read: 3.2 metres", "done": True}}
        )
        return "3.2 metres and rising"
"""
FIRST_STATUS = "Walking to the gauge at Portree"
LAST_STATUS = "Gauge read: 3.2 metres"


@pytest.fixture
def gauge_tool(admin):
    with python_tool(admin, GAUGE_TOOL, name="Tide gauge") as tool_id:
        yield tool_id


def _ask_with_tool(page: Page, upstream, tool_id: str, model_id: str = MOCK_MODEL_ID) -> None:
    question = f"how high is the tide in Portree? {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.tool_call("read_tide_gauge", {"harbour": "Portree"}, match=reply.answering(question)),
        reply.text("The tide is at 3.2 metres.", match=reply.answering(question)),
    )
    page.goto(f"/?models={model_id}&tools={tool_id}")
    send(page, question)
    expect_reply(page, "The tide is at 3.2 metres.")


def _status_toggle(page: Page) -> Locator:
    return conversation(page).get_by_role("button", name="Toggle status history")


def _status_history_opens(page: Page) -> None:
    toggle = _status_toggle(page)
    expect(toggle).to_contain_text(LAST_STATUS)
    if toggle.get_attribute("aria-expanded") != "true":
        toggle.click()
    expect(conversation(page).get_by_text(FIRST_STATUS)).to_be_visible()
    expect(conversation(page).get_by_text(LAST_STATUS)).to_have_count(2)


def test_the_latest_status_shows_and_its_history_opens(page_for, make_user, upstream, gauge_tool):
    page = page_for(make_user())
    _ask_with_tool(page, upstream, gauge_tool)

    _status_history_opens(page)


def test_the_status_history_is_kept_with_the_reply(page_for, make_user, upstream, gauge_tool):
    page = page_for(make_user())
    _ask_with_tool(page, upstream, gauge_tool)

    page.reload()
    expect_reply(page, "The tide is at 3.2 metres.")
    _status_history_opens(page)


@pytest.fixture
def quiet_model(admin):
    model_id = f"quiet-{uuid.uuid4().hex[:6]}"
    form = {
        "id": model_id,
        "name": "Quiet harbour",
        "base_model_id": MOCK_MODEL_ID,
        "meta": {"capabilities": {"status_updates": False}},
        "params": {},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_a_model_with_status_updates_unticked_shows_none(
    page_for, make_user, upstream, gauge_tool, quiet_model
):
    page = page_for(make_user())
    _ask_with_tool(page, upstream, gauge_tool, quiet_model)

    expect(_status_toggle(page)).to_have_count(0)
    expect(conversation(page).get_by_text(LAST_STATUS)).to_have_count(0)
