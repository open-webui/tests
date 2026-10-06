"""Journey: the admin's function editor saves code, refuses code that does not parse, and imports.

Source with a syntax error is refused when a new function is saved: the editor shows where the
code fails to parse and an error, stays open and nothing is stored. The same mistake made while
editing an existing pipe shows the same errors, keeps the stored code and switches the pipe off,
as the functions docs describe; switched back on it answers with its old code. Code edited and
saved without a mistake answers the next chat. Import From Link fetches a function's source into
the editor, named after its file, and once saved and switched on it answers a chat.

Discriminates: passes on dev ebc6add67. In a backend copy whose function create and update store
the source without loading it, the two syntax error tests fail; in one whose update leaves the
running module in place the edit test fails; in one whose link import answers a placeholder in
place of the fetched source the import test fails.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.listener import text_answer
from harness.plugins import installed_function
from utils.chat_ui import expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def clock_pipe(answer: str) -> str:
    return f"""class Pipe:
    def pipe(self, body: dict) -> str:
        return "{answer}"
"""


MISSING_COLON = """class Pipe:
    def pipe(self, body: dict) -> str
        return "never"
"""

PARSE_ERROR = re.compile(r"Cannot parse.*2:\d+")


@pytest.fixture
def editor_admin(make_user):
    """A fresh admin; the functions it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for function in client.get("/api/v1/functions/").json():
            if function["user_id"] == account.id:
                client.delete(f"/api/v1/functions/id/{function['id']}/delete")


def replace_code(page: Page, source: str) -> None:
    page.get_by_role("main").locator(".cm-content").click()
    page.keyboard.press("ControlOrMeta+A")
    page.keyboard.insert_text(source)


def confirm(page: Page) -> None:
    page.get_by_role("dialog", name="Confirm your action").get_by_role(
        "button", name="Confirm"
    ).click()


def ask_pipe(page: Page, pipe_id: str, question: str, answer: str) -> None:
    page.goto(f"/?models={pipe_id}")
    send(page, question)
    expect_reply(page, answer)


def pipe_card(page: Page, pipe_id: str) -> Locator:
    page.goto("/admin/functions")
    page.get_by_placeholder("Search Functions").fill(pipe_id)
    card = page.get_by_role("main").get_by_role("button", name=re.compile(f"^pipe {pipe_id}"))
    expect(card).to_be_visible()
    return card


def test_new_code_that_does_not_parse_is_refused_with_the_error_shown(page_for, editor_admin):
    name = f"Tide clock {uuid.uuid4().hex[:6]}"
    page = page_for(editor_admin)
    page.goto("/admin/functions/create")
    editor = page.get_by_role("main")
    editor.get_by_role("textbox", name="Function Name").fill(name)
    editor.get_by_role("textbox", name="Function Description").fill("tells the tide")
    replace_code(page, MISSING_COLON)
    editor.get_by_role("button", name="Save & Create").click()
    confirm(page)

    expect(page.get_by_text(PARSE_ERROR)).to_be_visible()
    expect(page.get_by_text("Error creating function")).to_be_visible()
    expect(page).to_have_url(re.compile(r"/admin/functions/create$"))
    with editor_admin.client() as client:
        stored = [function["name"] for function in client.get("/api/v1/functions/").json()]
    assert name not in stored


def open_editor(page: Page, function_id: str) -> None:
    page.goto(f"/admin/functions/edit?id={function_id}")
    expect(page.get_by_role("main").locator(".cm-content")).to_contain_text("class Pipe")


def test_an_edit_that_does_not_parse_keeps_the_old_code_and_switches_the_pipe_off(page_for, admin):
    with installed_function(admin, clock_pipe("High tide at nine.")) as pipe_id:
        page = page_for(admin)
        open_editor(page, pipe_id)
        replace_code(page, MISSING_COLON)
        page.get_by_role("main").get_by_role("button", name="Save", exact=True).click()

        expect(page.get_by_text(PARSE_ERROR)).to_be_visible()
        expect(page.get_by_text("Error updating function")).to_be_visible()
        switch = pipe_card(page, pipe_id).get_by_role("switch")
        expect(switch).not_to_be_checked()
        switch.click()
        expect(switch).to_be_checked()
        ask_pipe(page, pipe_id, "when is high tide?", "High tide at nine.")


def test_edited_code_answers_the_next_chat(page_for, admin):
    with installed_function(admin, clock_pipe("High tide at nine.")) as pipe_id:
        page = page_for(admin)
        ask_pipe(page, pipe_id, "when is high tide?", "High tide at nine.")

        open_editor(page, pipe_id)
        replace_code(page, clock_pipe("High tide at ten."))
        page.get_by_role("main").get_by_role("button", name="Save", exact=True).click()
        expect(page.get_by_text("Function updated successfully")).to_be_visible()

        ask_pipe(page, pipe_id, "when is high tide today?", "High tide at ten.")


def test_a_function_imported_from_a_link_is_saved_and_answers(page_for, editor_admin, listener):
    file_name = f"tide_clock_{uuid.uuid4().hex[:6]}"
    source = clock_pipe("Low tide at three.")
    listener.route("GET", f"/functions/{file_name}.py", text_answer(source, "text/x-python"))
    page = page_for(editor_admin)
    page.goto("/admin/functions")
    page.get_by_label("Open create menu").click()
    page.get_by_role("button", name="Import From Link").click()
    importing = page.get_by_role("dialog")
    importing.get_by_placeholder("Enter the URL to import").fill(
        f"{listener.base_url}/functions/{file_name}.py"
    )
    importing.get_by_role("button", name="Import").click()

    expect(page).to_have_url(re.compile(r"/admin/functions/create$"))
    editor = page.get_by_role("main")
    expect(editor.get_by_role("textbox", name="Function Name")).to_have_value(file_name)
    editor.get_by_role("textbox", name="Function Description").fill("tells the tide")
    editor.get_by_role("button", name="Save & Create").click()
    confirm(page)
    expect(page).to_have_url(re.compile(r"/admin/functions$"))

    switch = pipe_card(page, file_name).get_by_role("switch")
    switch.click()
    expect(switch).to_be_checked()
    ask_pipe(page, file_name, "when is low tide?", "Low tide at three.")
