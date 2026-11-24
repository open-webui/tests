"""Regression: what a person picks in a new chat before the model list is in is kept.

Issue open-webui/open-webui#34919. Since dev 14892a184 a new chat shows its message box and its
More menu before the model list has arrived, and text typed there in that moment is kept. A tool
ticked under More > Tools in the same moment was unticked again once the list arrived and the chat
finished setting itself up, so the next message went out without it. The same holds for Web Search
and Image switched on there, a skill ticked under Skills and a file attached through Upload Files.
A tool ticked early on a model with a default tool of its own goes out beside that default, as a
tool ticked once the chat is ready does.

Each test holds the model list back, makes its pick while the chat still waits for it, then lets
the list through. The pick has to show as picked afterwards and reach the model with the message.

The rest pin what keeping the picks must not break: typed text stays, two files still uploading
stay two, a saved draft's file shows once and a file attached early joins it, a draft's tools give
way to the defaults of the model the chat reopens on, and a file whose upload fails while the chat
sets itself up does not come back.

Discriminates: the seven pick tests and the draft-join test fail on the dev dc9886799 build (the
pick is gone once the list is in) and pass with the fix. The others pass on dev; the failed-upload
test fails on a build that puts back every early file, failed ones included (the file comes back
stuck and blocks sending).
"""

from __future__ import annotations

import json
import re
import uuid

import pytest
from playwright.sync_api import Page, Route, expect

from harness import upstream as reply
from harness.channel_chat import serve_openai_images
from harness.image_engines import IMAGES_CONFIG, save_image_settings
from harness.python_tools import EVERYONE_READS, python_tool
from harness.upstream import MOCK_MODEL_ID
from harness.web_retrieval import RETRIEVAL_CONFIG, save_web_settings, serve_search_results
from utils.cached_chat import attach, turn_on_tool
from utils.chat_ui import (
    chat_input,
    close_menu,
    expect_picked,
    expect_reply,
    integration_switch,
    open_integrations_menu,
    send,
    turn_on_integration,
)

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PUFFIN_TOOL = '''
class Tools:
    def count_puffins(self, cliff: str) -> str:
        """Count the puffins nesting on a cliff."""
        return "forty"
'''
MODEL_LIST = re.compile(r"/api/models(\?|$)")
UPLOAD = re.compile(r"/api/v1/files/(\?|$)")


def model_selector(page: Page):
    return page.get_by_role("button", name=re.compile("^Selected model"))


def new_chat_waiting_for_models(page: Page) -> list[Route]:
    """Reload into a new chat whose model list is held back; returns the held requests."""
    held: list[Route] = []
    page.route(MODEL_LIST, lambda route: held.append(route), times=1)
    page.reload()
    expect(chat_input(page)).to_be_visible()
    expect(model_selector(page)).to_have_count(0)
    return held


def let_models_through(page: Page, held: list[Route]) -> None:
    assert held, "the chat never asked for the model list"
    # the chat is still waiting for its model list here, so the pick was made in that window
    expect(model_selector(page)).to_have_count(0)
    for route in held:
        route.continue_()
    expect(model_selector(page).first).to_be_visible()


def ask(page: Page, upstream) -> dict:
    """Send a question and return the request the model got for it."""
    question = f"How many puffins? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Forty.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Forty.")
    return next(filter(reply.answering(question), upstream.chat_requests()))


def offered_tools(request: dict) -> set[str]:
    return {tool["function"]["name"] for tool in request.get("tools") or []}


@pytest.fixture
def puffin_tool(admin) -> str:
    name = f"Puffin counter {uuid.uuid4().hex[:8]}"
    with python_tool(admin, PUFFIN_TOOL, name=name):
        yield name


def test_a_tool_ticked_before_the_model_list_is_in_stays_ticked(
    page_for, make_user, upstream, puffin_tool
):
    page = page_for(make_user())
    held = new_chat_waiting_for_models(page)
    turn_on_tool(page, puffin_tool)

    let_models_through(page, held)

    expect_picked(page, "Tools", 1)
    assert "count_puffins" in offered_tools(ask(page, upstream))


TIDE_TOOL = '''
class Tools:
    def read_tide(self, harbour: str) -> str:
        """Read the tide table of a harbour."""
        return "high at noon"
'''


@pytest.fixture
def tide_model(admin) -> str:
    """A model every account may use whose default tool reads the tide; its id."""
    with python_tool(admin, TIDE_TOOL, name=f"Tide default {uuid.uuid4().hex[:8]}") as tool_id:
        model_id = f"tide-model-{uuid.uuid4().hex[:8]}"
        form = {
            "id": model_id,
            "base_model_id": MOCK_MODEL_ID,
            "name": model_id,
            "meta": {"toolIds": [tool_id]},
            "params": {},
            "access_grants": [EVERYONE_READS],
        }
        with admin.client() as client:
            created = client.post("/api/v1/models/create", json=form)
            assert created.status_code == 200, created.text
            yield model_id
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_a_tool_ticked_before_the_model_list_is_in_joins_the_models_default_tool(
    page_for, make_user, upstream, puffin_tool, tide_model
):
    account = make_user()
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"models": [tide_model]}}
        )
    assert saved.status_code == 200, saved.text
    page = page_for(account)
    held = new_chat_waiting_for_models(page)
    turn_on_tool(page, puffin_tool)

    let_models_through(page, held)

    expect_picked(page, "Tools", 2)
    assert {"count_puffins", "read_tide"} <= offered_tools(ask(page, upstream))


@pytest.fixture
def web_search_on(admin, preserve, listener):
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        save_web_settings(client, **serve_search_results(listener, []))


@pytest.fixture
def image_engine_on(admin, preserve, listener):
    preserve(IMAGES_CONFIG)
    with admin.client() as client:
        save_image_settings(client, **serve_openai_images(listener))


@pytest.mark.parametrize(
    ("switch", "tool"),
    [("Web Search", "search_web"), ("Image", "generate_image")],
    ids=["web search", "image"],
)
def test_a_switch_turned_on_before_the_model_list_is_in_stays_on(
    page_for, make_user, upstream, web_search_on, image_engine_on, switch, tool
):
    page = page_for(make_user())
    held = new_chat_waiting_for_models(page)
    turn_on_integration(page, switch)

    let_models_through(page, held)

    expect(integration_switch(page, switch)).to_have_attribute("aria-checked", "true")
    close_menu(page)
    assert tool in offered_tools(ask(page, upstream))


def test_a_skill_ticked_before_the_model_list_is_in_stays_ticked(page_for, make_user, upstream):
    account = make_user(role="admin")
    name = f"Puffin lore {uuid.uuid4().hex[:6]}"
    lore = f"Puffins nest in burrows {uuid.uuid4().hex[:6]}."
    with account.client() as client:
        created = client.post(
            "/api/v1/skills/create",
            json={
                "id": f"skill_{uuid.uuid4().hex[:8]}",
                "name": name,
                "content": lore,
                "meta": {},
                "is_active": True,
                "access_grants": [EVERYONE_READS],
            },
        )
    assert created.status_code == 200, created.text
    page = page_for(account)
    held = new_chat_waiting_for_models(page)
    menu = open_integrations_menu(page)
    menu.get_by_role("menuitem", name=re.compile(r"^Skills")).click()
    skill = menu.get_by_role("menuitemcheckbox", name=re.compile(re.escape(name)))
    skill.click()
    expect(skill).to_have_attribute("aria-checked", "true")
    close_menu(page)

    let_models_through(page, held)

    expect_picked(page, "Skills", 1)
    assert lore in json.dumps(ask(page, upstream))


def attached_files(request: dict) -> str:
    """The attachment list the model got with the message."""
    [listing] = re.findall(r"<attached_files>.*?</attached_files>", json.dumps(request))
    return listing


def test_a_file_attached_before_the_model_list_is_in_stays_attached(page_for, make_user, upstream):
    page = page_for(make_user())
    name = f"puffins-{uuid.uuid4().hex[:6]}.txt"
    held = new_chat_waiting_for_models(page)
    attach(page, name, "The north cliff holds forty puffins.")
    form = page.locator("form")
    expect(form.get_by_text(name).first).to_be_visible()

    let_models_through(page, held)

    expect(form.get_by_text(name).first).to_be_visible()
    assert name in attached_files(ask(page, upstream))


def test_two_files_still_uploading_when_the_model_list_is_in_both_stay_attached(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    names = [f"cliff-{side}-{uuid.uuid4().hex[:6]}.txt" for side in ("north", "south")]
    uploads: list[Route] = []

    def hold_uploads(route: Route) -> None:
        if route.request.method == "POST" and len(uploads) < len(names):
            uploads.append(route)
        else:
            route.fallback()

    page.route(UPLOAD, hold_uploads)
    held = new_chat_waiting_for_models(page)
    form = page.locator("form")
    for name in names:
        attach(page, name, f"Puffins counted on {name}.")
        expect(form.get_by_text(name).first).to_be_visible()

    assert len(uploads) == 2, "the files did not start uploading"

    let_models_through(page, held)
    for name in names:
        expect(form.get_by_text(name).first).to_be_visible()
    for upload in uploads:
        upload.continue_()

    for name in names:
        expect(form.get_by_text(name).first).to_be_visible()
    listing = attached_files(ask(page, upstream))
    assert all(name in listing for name in names), listing


def test_text_typed_before_the_model_list_is_in_stays(page_for, make_user, upstream):
    page = page_for(make_user())
    held = new_chat_waiting_for_models(page)
    chat_input(page).click()
    page.keyboard.type("How many puffins nest here?")

    let_models_through(page, held)

    expect(chat_input(page)).to_have_text("How many puffins nest here?")


def saved_draft(page: Page, text: str) -> None:
    """Wait until the new chat's draft, kept for the tab, holds `text`."""
    page.wait_for_function(
        "text => (sessionStorage.getItem('chat-input') ?? '').includes(text)", arg=text
    )


def test_a_file_in_a_draft_shows_once_after_a_reload(page_for, make_user, upstream):
    page = page_for(make_user())
    name = f"draft-{uuid.uuid4().hex[:6]}.txt"
    attach(page, name, "Puffins nest on the north cliff.")
    form = page.locator("form")
    expect(form.get_by_text(name).first).to_be_visible()
    chat_input(page).click()
    page.keyboard.type("About the cliff")
    saved_draft(page, name)

    held = new_chat_waiting_for_models(page)
    let_models_through(page, held)

    expect(chat_input(page)).to_have_text("About the cliff")
    expect(form.get_by_text(name)).to_have_count(1)
    assert attached_files(ask(page, upstream)).count(name) == 1


def test_a_drafts_tools_give_way_to_the_defaults_of_the_model_it_reopens_on(
    page_for, make_user, upstream, tide_model
):
    account = make_user()
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"models": [tide_model]}}
        )
        assert saved.status_code == 200, saved.text
    page = page_for(account)
    expect_picked(page, "Tools", 1)
    chat_input(page).click()
    page.keyboard.type("Tides today?")
    saved_draft(page, "Tides today?")

    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"models": [MOCK_MODEL_ID]}}
        )
        assert saved.status_code == 200, saved.text
    page.evaluate("sessionStorage.removeItem('selectedModels')")
    held = new_chat_waiting_for_models(page)
    let_models_through(page, held)

    expect(model_selector(page).first).to_contain_text(MOCK_MODEL_ID)
    expect(chat_input(page)).to_have_text("Tides today?")
    row = open_integrations_menu(page).get_by_role("menuitem", name=re.compile(r"^Tools"))
    expect(row).not_to_contain_text("selected")
    close_menu(page)
    assert "read_tide" not in offered_tools(ask(page, upstream))


FUNCTIONS = re.compile(r"/api/v1/functions/(\?|$)")


def test_a_file_whose_upload_fails_while_the_chat_sets_up_does_not_come_back(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    name = f"broken-{uuid.uuid4().hex[:6]}.txt"
    functions_requests: list[Route] = []
    uploads: list[Route] = []

    def functions(route: Route) -> None:
        functions_requests.append(route)
        # the page's own load gets an error, so the chat's set-up asks again and waits there
        if len(functions_requests) == 1:
            route.fulfill(status=500, json={"detail": "busy"})

    page.route(FUNCTIONS, functions)
    page.route(
        UPLOAD,
        lambda route: uploads.append(route) if route.request.method == "POST" else route.fallback(),
    )
    held = new_chat_waiting_for_models(page)
    attach(page, name, "Puffins nest on the north cliff.")
    form = page.locator("form")
    expect(form.get_by_text(name).first).to_be_visible()
    assert len(uploads) == 1, "the file did not start uploading"

    let_models_through_to_setup(page, held, functions_requests)
    uploads[0].fulfill(status=500, json={"detail": "The upload failed."})
    expect(page.get_by_text("The upload failed.").first).to_be_visible()
    page.unroute(UPLOAD)
    functions_requests[1].continue_()

    expect(form.get_by_text(name)).to_have_count(0)
    assert "<attached_files>" not in json.dumps(ask(page, upstream))


def let_models_through_to_setup(page: Page, held: list[Route], functions_requests: list) -> None:
    """Let the model list through and wait until the chat's set-up is waiting on its functions."""
    for route in held:
        route.continue_()
    deadline = 100
    while len(functions_requests) < 2 and deadline:
        page.wait_for_timeout(100)
        deadline -= 1
    assert len(functions_requests) == 2, "the chat set itself up without asking for functions"


def test_a_file_attached_before_the_model_list_is_in_joins_a_saved_draft(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    chat_input(page).click()
    page.keyboard.type("Notes from the cliff")
    saved_draft(page, "Notes from the cliff")
    name = f"cliff-{uuid.uuid4().hex[:6]}.txt"

    held = new_chat_waiting_for_models(page)
    attach(page, name, "Puffins nest on the north cliff.")
    form = page.locator("form")
    expect(form.get_by_text(name).first).to_be_visible()
    let_models_through(page, held)

    expect(chat_input(page)).to_have_text("Notes from the cliff")
    expect(form.get_by_text(name)).to_have_count(1)
    assert name in attached_files(ask(page, upstream))
