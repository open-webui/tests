"""Journey: a model's tools, filter, action, knowledge and prompt suggestions stay with that model.

A fresh admin has two presets on the scripted model: the harbour master, with a workspace tool, a
filter, an action and a knowledge base attached and a prompt suggestion of its own, and the ferry
clerk with none of them. A chat on the harbour master offers its tool, runs its filter on the
message, shows its action under the reply, offers its knowledge and shows its suggestion on the new
chat. Switching the same chat to the ferry clerk, or a new chat started from it, carries none of
that over: the clerk is offered no harbour tool and no harbour knowledge, its messages go out
unfiltered, its replies carry no action and its new chat shows the instance's default suggestions.
Attaching each of them in the editor is covered in e2e/models/test_model_editor.py.

Discriminates: passes on the dev ebc6add67 build. In a frontend build whose chat keeps the chosen
tools when the model changes both tool and filter tests fail, and in one whose new chat shows the
suggestions of any model that has some the suggestion test fails. In a backend copy that runs
every active filter on every model both tool and filter tests fail, in one that lists every active
action on every model the action test fails and in one where a model without knowledge borrows
the last model's the knowledge test fails.
"""

from __future__ import annotations

import contextlib
import uuid
from dataclasses import dataclass

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.knowledge_bases import add_text_file, knowledge_base
from harness.plugins import installed_function
from harness.python_tools import EVERYONE_READS, python_tool
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import replies
from utils.model_editor import offered_tool_names, open_chat_on, sent_request
from utils.model_selector import select_model

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

LOCKER_TOOL = """class Tools:
    def lookup_locker(self, number: int) -> str:
        \"\"\"Look up who holds a locker.

        :param number: the locker number
        \"\"\"
        return f"Locker {number} belongs to Ada."
"""

GATE_FILTER = """class Filter:
    def inlet(self, body: dict, __user__=None) -> dict:
        body["messages"][-1]["content"] += " (checked at the harbour gate)"
        return body
"""

DESK_ACTION = """class Action:
    async def action(self, body: dict, __user__=None, __event_emitter__=None):
        await __event_emitter__(
            {"type": "notification", "data": {"type": "success", "content": "Logged at the desk"}}
        )
"""


@dataclass
class Harbour:
    master: dict
    clerk: dict
    action_id: str


def create_preset(client, name: str, meta: dict) -> dict:
    model = {"id": f"scope-{uuid.uuid4().hex[:8]}", "name": f"{name} {uuid.uuid4().hex[:6]}"}
    created = client.post(
        "/api/v1/models/create",
        json={
            **model,
            "base_model_id": MOCK_MODEL_ID,
            "meta": meta,
            "params": {},
            "access_grants": [EVERYONE_READS],
        },
    )
    assert created.status_code == 200, created.text
    return model


@pytest.fixture
def harbour(admin, make_user):
    """A fresh admin and its two presets; everything made is deleted afterwards."""
    builder = make_user(role="admin")
    with contextlib.ExitStack() as stack:
        tool_id = stack.enter_context(python_tool(admin, LOCKER_TOOL, name="Locker desk"))
        filter_id = stack.enter_context(installed_function(admin, GATE_FILTER))
        action_id = stack.enter_context(installed_function(admin, DESK_ACTION))
        client = stack.enter_context(builder.client())
        base_id = stack.enter_context(knowledge_base(client, name="Harbour log"))
        add_text_file(client, base_id, "keepers.txt", "The lighthouse keeper is Morag.")
        master_meta = {
            "toolIds": [tool_id],
            "filterIds": [filter_id],
            "actionIds": [action_id],
            "knowledge": [{"type": "collection", "id": base_id, "name": "Harbour log"}],
            "suggestion_prompts": [
                {"title": ["Lockers", "who holds which"], "content": "Who holds locker 7?"}
            ],
        }
        master = create_preset(client, "Harbour master", master_meta)
        clerk = create_preset(client, "Ferry clerk", {})
        stack.callback(client.post, "/api/v1/models/model/delete", json={"id": master["id"]})
        stack.callback(client.post, "/api/v1/models/model/delete", json={"id": clerk["id"]})
        yield builder, Harbour(master, clerk, action_id)


def switch_model(page: Page, current: dict, target: dict) -> None:
    page.get_by_role("button", name=f"Selected model: {current['name']}").click()
    select_model(page, target["name"])
    expect(page.get_by_role("button", name=f"Selected model: {target['name']}")).to_be_visible()


def whole_reply(page: Page, index: int) -> Locator:
    """The reply at `index` with the buttons under it."""
    return replies(page).nth(index).locator("xpath=ancestor::*[starts-with(@id, 'message-')][1]")


def test_the_tool_and_filter_do_not_follow_the_chat_to_another_model(page_for, harbour, upstream):
    builder, models = harbour
    page = page_for(builder)
    open_chat_on(page, models.master)
    on_master = sent_request(page, upstream, "who holds locker 7?")
    assert "lookup_locker" in offered_tool_names(on_master)
    assert (
        on_master["messages"][-1]["content"] == "who holds locker 7? (checked at the harbour gate)"
    )

    switch_model(page, models.master, models.clerk)
    on_clerk = sent_request(page, upstream, "who holds locker 8?")

    assert "lookup_locker" not in offered_tool_names(on_clerk)
    assert on_clerk["messages"][-1]["content"] == "who holds locker 8?"


def test_a_new_chat_switched_to_another_model_carries_no_tool_or_filter(
    page_for, harbour, upstream
):
    builder, models = harbour
    page = page_for(builder)
    open_chat_on(page, models.master)
    sent_request(page, upstream, "who holds locker 7?")

    page.get_by_role("link", name="New Chat").first.click()
    expect(replies(page)).to_have_count(0)
    switch_model(page, models.master, models.clerk)
    on_clerk = sent_request(page, upstream, "who holds locker 9?")

    assert "lookup_locker" not in offered_tool_names(on_clerk)
    assert on_clerk["messages"] == [{"role": "user", "content": "who holds locker 9?"}]


def test_the_action_shows_only_under_the_models_own_replies(page_for, harbour, upstream):
    builder, models = harbour
    page = page_for(builder)
    open_chat_on(page, models.master)
    sent_request(page, upstream, "log my arrival", answer="Arrival noted.")
    master_reply = whole_reply(page, 0)
    expect(master_reply.get_by_role("button", name=models.action_id, exact=True)).to_be_visible()

    switch_model(page, models.master, models.clerk)
    sent_request(page, upstream, "log my departure", answer="Departure noted.")

    expect(replies(page)).to_have_count(2)
    expect(whole_reply(page, 1).get_by_role("button", name=models.action_id)).to_have_count(0)
    expect(master_reply.get_by_role("button", name=models.action_id, exact=True)).to_be_visible()


def test_the_knowledge_is_offered_only_with_its_model(page_for, harbour, upstream):
    builder, models = harbour
    page = page_for(builder)
    open_chat_on(page, models.master)
    on_master = offered_tool_names(sent_request(page, upstream, "who keeps the lighthouse?"))
    assert "list_knowledge" in on_master, sorted(on_master)
    assert "list_knowledge_bases" not in on_master, sorted(on_master)

    switch_model(page, models.master, models.clerk)
    on_clerk = offered_tool_names(sent_request(page, upstream, "who keeps the light?"))

    assert "list_knowledge" not in on_clerk, sorted(on_clerk)
    assert "list_knowledge_bases" in on_clerk, sorted(on_clerk)


def default_suggestion_title(actor) -> str:
    """The first of the instance's default suggestions, or of the built-in ones when it has none."""
    with actor.client() as client:
        configured = client.get("/api/config").json().get("default_prompt_suggestions")
    return configured[0]["title"][0] if configured else "Help me study"


def test_the_prompt_suggestion_shows_only_on_its_models_new_chat(page_for, harbour):
    builder, models = harbour
    page = page_for(builder)
    open_chat_on(page, models.master)
    expect(page.get_by_role("listitem").filter(has_text="Lockers")).to_be_visible()

    open_chat_on(page, models.clerk)
    default_title = default_suggestion_title(builder)
    expect(page.get_by_role("listitem").filter(has_text=default_title)).to_be_visible()
    expect(page.get_by_role("listitem").filter(has_text="Lockers")).to_have_count(0)
