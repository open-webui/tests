"""Journey: a filter written in the admin's function editor signs the replies of every chat.

A fresh admin writes a filter in the Functions editor, switches it on and makes it global from
the function list, and sets its valve to a signature. The filter's outlet appends the valve to
the reply's text, so the next chat shows the reply signed, and still signed after a reload.

Discriminates: passes on dev ac00d40e3; in a backend copy, with
`POST /api/v1/functions/id/{id}/valves/update` answering without storing the valves the reply is
signed with the default `unsigned` instead of the saved valve.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from utils.chat_ui import expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SIGNING_FILTER = """from pydantic import BaseModel


class Filter:
    class Valves(BaseModel):
        signature: str = "unsigned"

    def __init__(self):
        self.valves = self.Valves()

    def outlet(self, body: dict) -> dict:
        signed = f" [{self.valves.signature}]"
        reply = body["messages"][-1]
        reply["content"] += signed
        for item in reply.get("output") or []:
            if item.get("type") == "message":
                item["content"][-1]["text"] += signed
        return body
"""


@pytest.fixture
def author(make_user, admin):
    """A fresh admin; the functions it made are deleted afterwards, so no other chat is signed."""
    account = make_user(role="admin")
    yield account
    with admin.client() as client:
        for function in client.get("/api/v1/functions/").json():
            if function["user_id"] == account.id:
                client.delete(f"/api/v1/functions/id/{function['id']}/delete")


def test_a_global_filter_from_the_editor_signs_the_reply_with_its_valve(page_for, author, upstream):
    name = f"Signer {uuid.uuid4().hex[:6]}"
    signature = f"checked by {name}"
    page = page_for(author)
    page.goto("/admin/functions/create")
    editor = page.get_by_role("main")
    editor.get_by_role("textbox", name="Function Name").fill(name)
    editor.get_by_role("textbox", name="Function Description").fill("signs every reply")
    editor.locator(".cm-content").click()
    page.keyboard.press("ControlOrMeta+A")
    page.keyboard.insert_text(SIGNING_FILTER)
    editor.get_by_role("button", name="Save & Create").click()
    page.get_by_role("dialog", name="Confirm your action").get_by_role(
        "button", name="Confirm"
    ).click()
    expect(page).to_have_url(re.compile(r"/admin/functions$"))

    card = page.get_by_role("main").get_by_role("button", name=re.compile(f"^filter {name}"))
    card.get_by_role("switch").click()
    expect(card.get_by_role("switch")).to_be_checked()
    card.get_by_role("button", name="Function Menu").first.click()
    page.get_by_role("menu").get_by_role("switch").click()
    expect(page.get_by_role("menu").get_by_role("switch")).to_be_checked()
    page.keyboard.press("Escape")

    card.get_by_role("button", name="Valves").click()
    valves = page.get_by_role("dialog").filter(has_text="Valves")
    valves.get_by_role("button", name="Default").click()
    valves.get_by_role("textbox", name="Signature").fill(signature)
    valves.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Valves updated successfully")).to_be_visible()

    question = "is the report ready?"
    upstream.queue(reply.text("The report is ready.", match=reply.answering(question)))
    page.goto("/")
    send(page, question)
    expect_reply(page, f"The report is ready. [{signature}]")

    page.reload()
    expect_reply(page, f"The report is ready. [{signature}]")
