"""Journey: the status updates a tool sends show above its reply, with their history on a click.

A workspace tool reports what it is doing through its event emitter. The reply shows the latest
status above its text, the Toggle status history button opens every status the tool sent, in
order, by click or by keyboard, and both are still there when the chat is opened again. A model
with Status Updates unticked in its capabilities shows none of them.

The other events a tool sends to the page are covered at the end: a notification shows as a
toast, a confirmation and an input request show their message (and the input its field) and hand
the answer back to the tool, code the tool asks the page to run returns its result, and a code
execution shows on the reply.

The status tests run in the main chat and in a side chat (`utils/chat_surface.py`); the event
tests run in the main chat and in every way a side chat starts and every state it can be in. A
side chat drops the notification, shows a confirmation and an input request without their
message and field, never runs the code and shows no code execution
(open-webui/open-webui#34035, fix PR open-webui/open-webui#34064).

Discriminates: passes on dev 64796face; in a backend copy that keeps no status updates with the
reply the reload test fails, in one that drops every status event the other two fail as well, and
in a frontend build that ignores the Status Updates capability the unticked test fails. The event
tests fail in every side chat on dev c15f28649 (#34035) and pass on the build of #34064.
"""

from __future__ import annotations

import json
import uuid

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from harness.python_tools import EVERYONE_READS, python_tool
from harness.upstream import MOCK_MODEL_ID
from utils.chat_surface import EVERYWHERE, ChatSurface
from utils.chat_ui import REPLY_TIMEOUT_MS, conversation, expect_reply, send, start_from_link

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


def _ask_with_tool(
    chat: ChatSurface, upstream, tool_id: str, model_id: str = MOCK_MODEL_ID
) -> None:
    question = f"how high is the tide in Portree? {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.tool_call("read_tide_gauge", {"harbour": "Portree"}, match=reply.answering(question)),
        reply.text("The tide is at 3.2 metres.", match=reply.answering(question)),
    )
    start_from_link(chat, f"/?models={model_id}&tools={tool_id}")
    send(chat, question)
    expect_reply(chat, "The tide is at 3.2 metres.")


def _status_toggle(chat: ChatSurface) -> Locator:
    return conversation(chat).get_by_role("button", name="Toggle status history")


def _status_history_opens(chat: ChatSurface, by_keyboard: bool = False) -> None:
    toggle = _status_toggle(chat)
    expect(conversation(chat).get_by_text(LAST_STATUS)).to_be_visible()
    expect(conversation(chat).get_by_text(FIRST_STATUS)).to_have_count(0)
    expect(toggle).to_have_attribute("aria-expanded", "false")
    if by_keyboard:
        toggle.press("Enter")
    else:
        toggle.click()
    expect(toggle).to_have_attribute("aria-expanded", "true")
    expect(conversation(chat).get_by_text(FIRST_STATUS)).to_be_visible()
    expect(conversation(chat).get_by_text(LAST_STATUS)).to_have_count(2)


def test_the_latest_status_shows_and_its_history_opens(chat_for, make_user, upstream, gauge_tool):
    chat = chat_for(make_user())
    _ask_with_tool(chat, upstream, gauge_tool)

    _status_history_opens(chat)


def test_the_status_history_is_kept_with_the_reply(chat_for, make_user, upstream, gauge_tool):
    chat = chat_for(make_user())
    _ask_with_tool(chat, upstream, gauge_tool)

    chat = chat.reload()
    expect_reply(chat, "The tide is at 3.2 metres.")
    _status_history_opens(chat, by_keyboard=True)


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
    chat_for, make_user, upstream, gauge_tool, quiet_model
):
    chat = chat_for(make_user())
    _ask_with_tool(chat, upstream, gauge_tool, quiet_model)

    expect(_status_toggle(chat)).to_have_count(0)
    expect(conversation(chat).get_by_text(LAST_STATUS)).to_have_count(0)


HARBOUR_TOOL = '''
import json


class Tools:
    async def ring_the_harbour_bell(self, __event_emitter__=None) -> str:
        """Ring the harbour bell."""
        await __event_emitter__(
            {"type": "notification", "data": {"type": "info", "content": "The harbour bell rang"}}
        )
        return "rung"

    async def ask_the_harbour_master(self, __event_call__=None) -> str:
        """Ask the harbour master before opening the tide gate."""
        allowed = await __event_call__(
            {
                "type": "confirmation",
                "data": {"title": "Open the tide gate?", "message": "The tide is coming in fast."},
            }
        )
        keeper = await __event_call__(
            {
                "type": "input",
                "data": {
                    "title": "Gate keeper",
                    "message": "Who is on duty at the gate?",
                    "placeholder": "Keeper's name",
                },
            }
        )
        return json.dumps({"allowed": allowed, "keeper": keeper})

    async def count_in_the_browser(self, __event_call__=None) -> str:
        """Count the tide marks in the browser."""
        counted = await __event_call__({"type": "execute", "data": {"code": "return 6 * 7;"}})
        return json.dumps(counted)

    async def run_the_tide_model(self, __event_emitter__=None) -> str:
        """Run the tide model."""
        await __event_emitter__(
            {
                "type": "citation",
                "data": {
                    "type": "code_execution",
                    "id": "tide-model",
                    "name": "Tide model run",
                    "code": "print(3.2)",
                    "language": "python",
                    "result": {"output": "3.2 metres at noon"},
                },
            }
        )
        return "3.2 metres at noon"
'''


@pytest.fixture
def harbour_tool(admin):
    with python_tool(admin, HARBOUR_TOOL, name="Harbour master") as tool_id:
        yield tool_id


def _call(chat: ChatSurface, upstream, tool_id: str, tool: str, answer: str) -> str:
    """Have the model call `tool` and answer `answer` after it; returns the question asked."""
    question = f"please {tool.replace('_', ' ')} ({uuid.uuid4().hex[:6]})"
    upstream.queue(
        reply.tool_call(tool, {}, match=reply.answering(question)),
        reply.text(answer, match=reply.answering(question)),
    )
    start_from_link(chat, f"/?models={MOCK_MODEL_ID}&tools={tool_id}")
    send(chat, question)
    return question


def _tool_results(upstream, question: str) -> list[str]:
    final = [body for body in upstream.chat_requests() if reply.answering(question)(body)][-1]
    return [str(entry["content"]) for entry in reply.conversation(final) if entry["role"] == "tool"]


# several models picked: every answer's events carry its own chat and message, as with one model
@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_a_tool_notification_shows_as_a_toast(chat_for, make_user, upstream, harbour_tool):
    """In a side chat: open-webui/open-webui#34035."""
    chat = chat_for(make_user())
    _call(chat, upstream, harbour_tool, "ring_the_harbour_bell", "The bell has rung.")
    expect_reply(chat, "The bell has rung.")

    expect(
        chat.page.get_by_text("The harbour bell rang", exact=True),
        f"the tool's notification showed no toast in the {chat.place}",
    ).to_be_visible()


@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_a_tool_confirmation_and_input_request_show_their_message_and_return_the_answer(
    chat_for, make_user, upstream, harbour_tool
):
    """In a side chat: open-webui/open-webui#34035."""
    chat = chat_for(make_user())
    page = chat.page
    question = _call(chat, upstream, harbour_tool, "ask_the_harbour_master", "The gate opens.")

    confirmation = page.get_by_role("dialog", name="Open the tide gate?")
    expect(confirmation).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    expect(
        confirmation.get_by_text("The tide is coming in fast."),
        f"the confirmation in the {chat.place} does not show the tool's message",
    ).to_be_visible()
    confirmation.get_by_role("button", name="Confirm").click()

    request = page.get_by_role("dialog", name="Gate keeper")
    expect(request).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    expect(
        request.get_by_text("Who is on duty at the gate?"),
        f"the input request in the {chat.place} does not show the tool's message",
    ).to_be_visible()
    field = request.get_by_role("textbox", name="Keeper's name")
    expect(field, f"the input request in the {chat.place} has no field").to_be_visible()
    field.fill("Ada")
    request.get_by_role("button", name="Confirm").click()

    expect_reply(chat, "The gate opens.")
    assert _tool_results(upstream, question) == [json.dumps({"allowed": True, "keeper": "Ada"})]


@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_code_a_tool_asks_the_page_to_run_returns_its_result(
    chat_for, make_user, upstream, harbour_tool
):
    """In a side chat: open-webui/open-webui#34035."""
    chat = chat_for(make_user())
    question = _call(chat, upstream, harbour_tool, "count_in_the_browser", "There are 42 marks.")

    expect_reply(chat, "There are 42 marks.")
    results = _tool_results(upstream, question)
    assert results == ["42"], f"the tool got {results} back from the page, not 42"


@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_a_code_execution_a_tool_reports_shows_on_the_reply(
    chat_for, make_user, upstream, harbour_tool
):
    """In a side chat: open-webui/open-webui#34035."""
    chat = chat_for(make_user())
    _call(chat, upstream, harbour_tool, "run_the_tide_model", "The model says 3.2 metres.")
    expect_reply(chat, "The model says 3.2 metres.")

    execution = conversation(chat).get_by_role("button", name="Tide model run")
    expect(execution, f"the reply in the {chat.place} shows no code execution").to_be_visible()
    execution.click()
    details = chat.page.get_by_role("dialog", name="Tide model run")
    expect(details).to_contain_text("print(3.2)")
    expect(details).to_contain_text("3.2 metres at noon")
