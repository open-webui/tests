"""Journey: an admin switches, deletes, exports and imports functions from Admin Panel > Functions.

A global filter that signs every reply stops signing once its switch in the list is turned off,
and once Global is turned off in its menu. Delete in the menu, after a confirmation, removes the
function and its effect. Export JSON downloads every function with its source, and importing
that file, after the warning is confirmed, brings a deleted function back.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose list switch, Global toggle and
delete skip their requests and whose Export JSON saves an empty list, every test but the signed
baseline fails.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.plugins import installed_function
from utils.chat_ui import expect_reply, last_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SIGNATURE = "[filed by the harbour office]"
SIGNING_FILTER = f"""class Filter:
    def outlet(self, body: dict) -> dict:
        reply = body["messages"][-1]
        reply["content"] += " {SIGNATURE}"
        for item in reply.get("output") or []:
            if item.get("type") == "message":
                item["content"][-1]["text"] += " {SIGNATURE}"
        return body
"""


@pytest.fixture
def signing_filter(admin):
    with installed_function(admin, SIGNING_FILTER, is_global=True) as function_id:
        yield function_id


def function_card(page: Page, function_id: str) -> Locator:
    page.goto("/admin/functions")
    card = page.get_by_role("main").get_by_role("button", name=re.compile(f"^filter {function_id}"))
    expect(card).to_be_visible()
    return card


def open_menu(page: Page, card: Locator) -> Locator:
    card.get_by_role("button", name="Function Menu").first.click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("button", name="Edit")).to_be_visible()
    return menu


def reply_text(page: Page, upstream, question: str) -> str:
    upstream.queue(reply.text("The ferry leaves at noon.", match=reply.answering(question)))
    page.goto("/")
    send(page, question)
    expect_reply(page, "The ferry leaves at noon.")
    return last_reply(page).inner_text()


def test_a_global_filter_signs_a_users_reply(page_for, make_user, upstream, signing_filter):
    page = page_for(make_user())

    assert SIGNATURE in reply_text(page, upstream, "When does the ferry leave?")


def test_a_filter_switched_off_in_the_list_stops_signing(
    page_for, admin, make_user, upstream, signing_filter
):
    admin_page = page_for(admin)
    switch = function_card(admin_page, signing_filter).get_by_role("switch")
    switch.click()
    expect(switch).not_to_be_checked()

    page = page_for(make_user())

    assert SIGNATURE not in reply_text(page, upstream, "When does the ferry leave today?")


def test_a_filter_no_longer_global_stops_signing(
    page_for, admin, make_user, upstream, signing_filter
):
    admin_page = page_for(admin)
    menu = open_menu(admin_page, function_card(admin_page, signing_filter))
    menu.get_by_role("switch").click()
    expect(admin_page.get_by_text("Filter is now globally disabled")).to_be_visible()

    page = page_for(make_user())

    assert SIGNATURE not in reply_text(page, upstream, "When does the last ferry leave?")


def test_a_deleted_function_leaves_the_list_and_stops_signing(
    page_for, admin, make_user, upstream, signing_filter
):
    admin_page = page_for(admin)
    card = function_card(admin_page, signing_filter)
    open_menu(admin_page, card).get_by_role("button", name="Delete").click()
    admin_page.get_by_role("dialog", name="Delete function?").get_by_role(
        "button", name="Confirm"
    ).click()

    expect(admin_page.get_by_text("Function deleted successfully")).to_be_visible()
    expect(card).to_have_count(0)
    page = page_for(make_user())
    assert SIGNATURE not in reply_text(page, upstream, "When does the morning ferry leave?")


def test_an_exported_function_comes_back_through_import(page_for, admin, make_user, upstream):
    name = f"Signer {uuid.uuid4().hex[:6]}"
    with installed_function(admin, SIGNING_FILTER, is_global=True) as function_id:
        admin_page = page_for(admin)
        function_card(admin_page, function_id)
        with admin_page.expect_download() as downloaded:
            admin_page.get_by_label("Open create menu").click()
            admin_page.get_by_role("button", name="Export JSON").click()
        exported = json.loads(Path(downloaded.value.path()).read_text())
        [saved] = [entry for entry in exported if entry["id"] == function_id]
        assert saved["content"] == SIGNING_FILTER
        with admin.client() as client:
            client.delete(f"/api/v1/functions/id/{function_id}/delete").raise_for_status()

        admin_page.goto("/admin/functions")
        with admin_page.expect_file_chooser() as chooser:
            admin_page.get_by_label("Open create menu").click()
            admin_page.get_by_role("button", name="Import JSON").click()
        chooser.value.set_files(
            files=[
                {
                    "name": f"{name}.json",
                    "mimeType": "application/json",
                    "buffer": json.dumps([saved]).encode(),
                }
            ]
        )
        warning = admin_page.get_by_role("dialog", name="Confirm your action")
        expect(warning).to_contain_text("Functions allow arbitrary code execution.")
        warning.get_by_role("button", name="Confirm").click()

        expect(admin_page.get_by_text("Functions imported successfully")).to_be_visible()
        expect(function_card(admin_page, function_id)).to_be_visible()
        with admin.client() as client:
            restored = client.get(f"/api/v1/functions/id/{function_id}")
        assert restored.status_code == 200, restored.text
        assert restored.json()["content"] == SIGNING_FILTER
