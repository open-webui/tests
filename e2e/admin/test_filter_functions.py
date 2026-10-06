"""Journey: filters as an admin sets them up and a person chatting meets them.

A filter attached to one model changes what that model is sent and leaves the base model alone,
until the admin switches Global on in its menu in Admin Panel > Functions; from then on the base
model is sent the change too. Two global filters that sign every reply sign it in the order of
their `priority` valve, and an admin who raises one filter's priority in its Valves dialog flips
the order on the next reply. A filter's user valve, set by each person in the chat's Controls
under Valves > Functions, signs that person's replies and nobody else's.

Discriminates: passes on dev ebc6add67. In a backend copy whose filter pipeline ignores a model's
own filters the scope test fails, in one that sorts filters by id alone the priority test fails,
and in one that hands every filter the default user valves the user valves test fails.
"""

from __future__ import annotations

import re
import uuid
from contextlib import contextmanager
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor
from harness.plugins import installed_function
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input, expect_reply, last_reply, send
from utils.valves import customise, valve

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

GATE_FILTER = """class Filter:
    def inlet(self, body: dict) -> dict:
        body["messages"][-1]["content"] += " (logged at the harbour gate)"
        return body
"""


def signing_filter(tag: str, priority: int) -> str:
    return f"""from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        priority: int = Field({priority}, description="Lower signs first")

    def __init__(self):
        self.valves = self.Valves()

    def outlet(self, body: dict) -> dict:
        reply = body["messages"][-1]
        reply["content"] += " [{tag}]"
        for item in reply.get("output") or []:
            if item.get("type") == "message":
                item["content"][-1]["text"] += " [{tag}]"
        return body
"""


USER_SIGNING_FILTER = """from pydantic import BaseModel, Field


class Filter:
    class UserValves(BaseModel):
        signature: str = Field("a guest", description="Who signs your replies")

    def outlet(self, body: dict, __user__: dict) -> dict:
        signed = f" [signed by {__user__['valves'].signature}]"
        reply = body["messages"][-1]
        reply["content"] += signed
        for item in reply.get("output") or []:
            if item.get("type") == "message":
                item["content"][-1]["text"] += signed
        return body
"""


@contextmanager
def model_with_filter(admin: Actor, filter_id: str) -> Iterator[dict]:
    """A preset on the scripted model, read by everyone, with `filter_id` attached to it."""
    model = {"id": f"gate-{uuid.uuid4().hex[:8]}", "name": f"Harbour gate {uuid.uuid4().hex[:6]}"}
    with admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                **model,
                "base_model_id": MOCK_MODEL_ID,
                "meta": {"filterIds": [filter_id]},
                "params": {},
                "access_grants": [EVERYONE_READS],
            },
        )
        assert created.status_code == 200, created.text
        try:
            yield model
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model["id"]})


def open_chat_on(page: Page, model_id: str, model_name: str) -> None:
    page.goto(f"/?model={model_id}")
    expect(page.get_by_role("button", name=f"Selected model: {model_name}")).to_be_visible()
    expect(chat_input(page)).to_be_visible()


def sent_text(page: Page, upstream, question: str) -> str:
    """Send `question` in the open chat; the last message the provider got for it."""
    upstream.queue(reply.text("Moor at berth four.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Moor at berth four.")
    request = next(filter(reply.answering(question), upstream.chat_requests()))
    return request["messages"][-1]["content"]


def function_card(page: Page, kind: str, function_id: str) -> Locator:
    page.goto("/admin/functions")
    page.get_by_placeholder("Search Functions").fill(function_id)
    card = page.get_by_role("main").get_by_role("button", name=re.compile(f"^{kind} {function_id}"))
    expect(card).to_be_visible()
    return card


def test_a_filter_on_one_model_reaches_the_others_once_made_global(
    page_for, admin, make_user, upstream
):
    with installed_function(admin, GATE_FILTER) as filter_id:
        with model_with_filter(admin, filter_id) as gated:
            page = page_for(make_user())
            open_chat_on(page, gated["id"], gated["name"])
            assert sent_text(page, upstream, "may I moor here?") == (
                "may I moor here? (logged at the harbour gate)"
            )
            open_chat_on(page, MOCK_MODEL_ID, MOCK_MODEL_ID)
            assert sent_text(page, upstream, "may I moor there?") == "may I moor there?"

            admin_page = page_for(admin)
            function_card(admin_page, "filter", filter_id).get_by_role(
                "button", name="Function Menu"
            ).first.click()
            admin_page.get_by_role("menu").get_by_role("switch").click()
            expect(admin_page.get_by_text("Filter is now globally enabled")).to_be_visible()

            open_chat_on(page, MOCK_MODEL_ID, MOCK_MODEL_ID)
            assert sent_text(page, upstream, "may I moor anywhere?") == (
                "may I moor anywhere? (logged at the harbour gate)"
            )


def expect_signed_reply(page: Page, upstream, question: str, signed: str) -> None:
    upstream.queue(reply.text("The tide turns at six.", match=reply.answering(question)))
    send(page, question)
    expect(last_reply(page)).to_contain_text(f"The tide turns at six.{signed}")


def test_filters_sign_the_reply_in_the_order_of_their_priority_valve(
    page_for, admin, make_user, upstream
):
    with (
        installed_function(admin, signing_filter("north", 1), is_global=True) as north,
        installed_function(admin, signing_filter("south", 2), is_global=True),
    ):
        page = page_for(make_user())
        page.goto("/")
        expect_signed_reply(page, upstream, "when does the tide turn?", " [north] [south]")

        admin_page = page_for(admin)
        function_card(admin_page, "filter", north).get_by_role("button", name="Valves").click()
        dialog = admin_page.get_by_role("dialog").filter(has_text="Valves")
        priority = valve(dialog, "Priority", "Lower signs first")
        customise(priority)
        priority.get_by_role("textbox").fill("3")
        dialog.get_by_role("button", name="Save").click()
        expect(admin_page.get_by_text("Valves updated successfully")).to_be_visible()

        page.goto("/")
        expect_signed_reply(page, upstream, "when does it turn again?", " [south] [north]")


def set_signature(page: Page, function_id: str, signature: str) -> None:
    """Set the filter's user valve in the open chat's Controls."""
    page.get_by_role("button", name="Controls").click()
    page.get_by_role("button", name="Valves").click()
    page.get_by_role("combobox").filter(
        has=page.get_by_role("option", name="Functions")
    ).select_option(label="Functions")
    function_picker = page.get_by_role("combobox").filter(
        has=page.get_by_role("option", name=function_id)
    )
    function_picker.select_option(label=function_id)
    panel = page.locator("form").filter(has=function_picker)
    signature_valve = valve(panel, "Signature", "Who signs your replies")
    customise(signature_valve)
    signature_valve.get_by_role("textbox").fill(signature)
    signature_valve.get_by_role("textbox").press("Tab")
    expect(page.get_by_text("Valves updated", exact=True)).to_be_visible()


def test_each_user_signs_their_replies_with_their_own_filter_user_valve(
    page_for, admin, make_user, upstream
):
    with installed_function(admin, USER_SIGNING_FILTER, is_global=True) as filter_id:
        first_page, second_page = page_for(make_user()), page_for(make_user())
        expect_signed_reply(first_page, upstream, "who signs this?", " [signed by a guest]")
        expect_signed_reply(second_page, upstream, "who signs mine?", " [signed by a guest]")

        set_signature(first_page, filter_id, "the harbour master")
        set_signature(second_page, filter_id, "the lighthouse keeper")

        expect_signed_reply(
            first_page, upstream, "who signs this now?", " [signed by the harbour master]"
        )
        expect_signed_reply(
            second_page, upstream, "who signs mine now?", " [signed by the lighthouse keeper]"
        )
