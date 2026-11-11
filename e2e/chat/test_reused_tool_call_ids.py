"""Regression: with reused tool call ids the saved reply showed the second call's input twice.

Fix 88720c692 (PR #31887, issue #28305). Some providers number tool calls from zero again in
every round of one reply, and the second call then overwrote the first one with the same id, so
the saved chat showed the first call with the second call's input and result. Each call now keeps
its own. Twin of integration/chat/test_reused_tool_call_ids.py.

Discriminates: passes on dev b859124f9, fails with 88720c692 reverted (both calls read "second").

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`).
"""

from __future__ import annotations

import re
import uuid
from typing import Iterator

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.python_tools import EVERYONE_READS, python_tool
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import expect_reply, last_reply, send
from utils.model_selector import select_model

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

SHOUTING_TOOL = '''
class Tools:
    def shout(self, text: str) -> str:
        """Shout the text."""
        return text.upper()
'''
PROMPT = "shout both words"


@pytest.fixture
def shouter(admin) -> Iterator[str]:
    """A preset of the scripted model with the shouting tool attached; yields its name."""
    name = f"Shouter {uuid.uuid4().hex[:8]}"
    model_id = name.lower().replace(" ", "-")
    with python_tool(admin, SHOUTING_TOOL) as tool_id, admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": name,
                "meta": {"toolIds": [tool_id]},
                "params": {},
                "access_grants": [EVERYONE_READS],
            },
        )
        assert created.status_code == 200, created.text
        yield name
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def shout(text: str) -> reply.Reply:
    return reply.tool_call("shout", {"text": text}, call_id="call_0", match=reply.answering(PROMPT))


def test_each_call_shows_its_own_input_and_result_after_a_reload(
    chat_for, make_user, upstream, shouter
):
    upstream.queue(
        shout("first"), shout("second"), reply.text("all shouted", match=reply.answering(PROMPT))
    )
    chat = chat_for(make_user())
    select_model(chat, shouter)
    send(chat, PROMPT)
    expect_reply(chat, "all shouted")

    chat = chat.reload()
    last_reply(chat).get_by_text("Explored 2 shout").click()
    calls = last_reply(chat).get_by_text("View Result from shout")
    expect(calls).to_have_count(2)
    calls.first.click()
    calls.last.click()

    in_order = re.compile(r"first[\s\S]*FIRST[\s\S]*second[\s\S]*SECOND")
    expect(last_reply(chat)).to_contain_text(in_order)
