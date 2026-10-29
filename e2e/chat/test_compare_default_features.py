"""A model's default Web Search was lost when comparing models, open-webui/open-webui#30310.

Fix commit `1b5a8ac5a` (PR open-webui/open-webui#30383). Changing the model selection clears the
feature toggles and then applies the model's default features, but only for a single model, so
adding a second model to the chat switched Web Search off even when both models have it as a
default feature. The same reset ran before the selection had settled, so for a single model the
toggle showed Web Search on while the request went out with it off. In compare mode a default
feature now stays on when every selected model has it, and the request matches the toggle.

Discriminates: passes on the efe63bd34 build, fails on it with `1b5a8ac5a` reverted (the toggle
and the request disagree).

The two journey tests pin switching models since v0.12.5 (dev 3aaecdd0e and 1c572c49b): the new
model's default tools and features are applied, but a tool the person switched on or off
themselves stays as they set it, and comparing two models turns on the default tools of both.
Discriminates: passes on dev c5981aaea; in a build that applies the new defaults over the
person's own choices the first fails (the switched-off tool comes back, the own tool goes), in one
that takes the default tools of the first model only the second fails.
"""

from __future__ import annotations

import contextlib
import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.python_tools import EVERYONE_READS, python_tool
from harness.upstream import MOCK_MODEL_ID
from harness.web_retrieval import RETRIEVAL_CONFIG, save_web_settings, serve_search_results
from utils.chat_ui import chat_input, close_menu, open_integrations_menu
from utils.model_selector import model_search

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def web_search_on(admin, preserve, listener) -> None:
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        save_web_settings(client, **serve_search_results(listener, []))


@pytest.fixture
def make_model(admin):
    """`make_model(searches_by_default, tool_ids)`: a preset able to search the web, named by its
    id, with `tool_ids` as its default tools."""
    created_ids: list[str] = []

    def create(searches_by_default: bool, tool_ids: tuple[str, ...] = ()) -> str:
        model_id = f"web-{uuid.uuid4().hex[:8]}"
        meta = {
            "capabilities": {"web_search": True},
            "defaultFeatureIds": ["web_search"] if searches_by_default else [],
            "toolIds": list(tool_ids),
        }
        form = {
            "id": model_id,
            "base_model_id": MOCK_MODEL_ID,
            "name": model_id,
            "meta": meta,
            "params": {},
            "access_grants": [EVERYONE_READS],
        }
        with admin.client() as client:
            saved = client.post("/api/v1/models/create", json=form)
        assert saved.status_code == 200, saved.text
        created_ids.append(model_id)
        return model_id

    yield create
    with admin.client() as client:
        for model_id in created_ids:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


LOOKUP_TOOL = '''class Tools:
    def lookup(self, term: str) -> str:
        """Look a term up."""
        return term
'''


@pytest.fixture
def make_tool(admin):
    """`make_tool()`: a tool every account may use; returns its id and its name."""
    with contextlib.ExitStack() as tools:

        def create() -> tuple[str, str]:
            name = f"Lookup {uuid.uuid4().hex[:8]}"
            return tools.enter_context(python_tool(admin, LOOKUP_TOOL, name=name)), name

        yield create


def open_on(page_for, make_user, model_id: str) -> Page:
    account = make_user()
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"models": [model_id]}}
        )
    saved.raise_for_status()
    page = page_for(account)
    expect(chat_input(page)).to_be_visible()
    expect(page.get_by_role("button", name=f"Selected model: {model_id}")).to_be_visible()
    return page


def pick_model(page: Page, current_id: str, model_id: str, *, compare: bool = False) -> None:
    page.get_by_role("button", name=f"Selected model: {current_id}").click()
    if compare:
        page.get_by_role("button", name="Compare").click()
    model_search(page).fill(model_id)
    page.get_by_role("option", name=f"Select {model_id} model").click()
    if compare:
        page.keyboard.press("Escape")
        shown = f"Selected model: {current_id} +1"
    else:
        shown = f"Selected model: {model_id}"
    expect(page.get_by_role("button", name=shown)).to_be_visible()


def web_search_shown_on(page: Page) -> bool:
    """What the Web Search toggle in the Integrations menu shows."""
    toggle = open_integrations_menu(page).get_by_role("menuitemcheckbox", name="Web Search")
    expect(toggle).to_be_visible()
    shown_on = toggle.get_attribute("aria-checked") == "true"
    page.keyboard.press("Escape")
    expect(toggle).to_be_hidden()
    return shown_on


def sent_request(page: Page, upstream, question: str) -> dict:
    """Send `question`; the first chat request the page made for it."""
    upstream.queue(reply.text("noted", match=reply.answering(question)))

    def is_chat_request(request) -> bool:
        return request.method == "POST" and request.url.endswith("/api/chat/completions")

    with page.expect_request(is_chat_request) as sent:
        chat_input(page).click()
        page.keyboard.type(question)
        page.keyboard.press("Enter")
    return sent.value.post_data_json


def web_search_sent_on(page: Page, upstream, question: str) -> bool:
    """Send `question`; whether the chat request the page made asked for web search."""
    features = sent_request(page, upstream, question).get("features") or {}
    return features.get("web_search") is True


def open_tools(page: Page) -> Locator:
    """The Tools list of the chat input's Integrations menu."""
    menu = open_integrations_menu(page)
    menu.get_by_role("menuitem", name=re.compile(r"^Tools")).click()
    expect(menu.get_by_placeholder("Search tools")).to_be_visible()
    return menu


def close_tools(page: Page, menu: Locator) -> None:
    """Go back to the menu's first list and close it; the menu reopens on the list it closed on."""
    menu.get_by_role("menuitem", name=re.compile(r"^Tools")).click()
    expect(menu.get_by_placeholder("Search tools")).to_be_hidden()
    close_menu(page)


def tool_switch(menu: Locator, name: str) -> Locator:
    return menu.get_by_role("menuitemcheckbox", name=name, exact=True)


def expect_tools(page: Page, *, on: list[str], off: list[str]) -> None:
    """Check in the Tools list which of the named tools show switched on and which off."""
    menu = open_tools(page)
    for name in on:
        expect(tool_switch(menu, name)).to_have_attribute("aria-checked", "true")
    for name in off:
        expect(tool_switch(menu, name)).to_have_attribute("aria-checked", "false")
    close_tools(page, menu)


def test_comparing_two_models_that_search_by_default_keeps_web_search_on(
    page_for, make_user, make_model, web_search_on, upstream
):
    first, second = make_model(True), make_model(True)
    page = open_on(page_for, make_user, first)

    pick_model(page, first, second, compare=True)

    assert web_search_shown_on(page), "the toggle went off when the second model was added"
    assert web_search_sent_on(page, upstream, "compare the two")


def test_switching_to_a_model_that_searches_by_default_sends_web_search_on(
    page_for, make_user, make_model, web_search_on, upstream
):
    searching = make_model(True)
    page = open_on(page_for, make_user, MOCK_MODEL_ID)

    pick_model(page, MOCK_MODEL_ID, searching)

    assert web_search_shown_on(page)
    assert web_search_sent_on(page, upstream, "just the one"), "the toggle is on, the request not"


def test_comparing_with_a_model_that_does_not_search_by_default_leaves_it_off(
    page_for, make_user, make_model, web_search_on, upstream
):
    first, second = make_model(True), make_model(False)
    page = open_on(page_for, make_user, first)

    pick_model(page, first, second, compare=True)

    assert not web_search_shown_on(page)
    assert not web_search_sent_on(page, upstream, "only one of us searches"), (
        "the toggle is off, the request asked for web search"
    )


def test_switching_to_a_model_that_does_not_search_by_default_turns_it_off(
    page_for, make_user, make_model, web_search_on, upstream
):
    searching, plain = make_model(True), make_model(False)
    page = open_on(page_for, make_user, searching)

    pick_model(page, searching, plain)

    assert not web_search_shown_on(page)
    assert not web_search_sent_on(page, upstream, "no search please")


@pytest.mark.journey
def test_switching_models_applies_the_new_defaults_but_keeps_the_tools_set_by_hand(
    page_for, make_user, make_model, make_tool, web_search_on, upstream
):
    (first_tool, first_name), (second_tool, second_name) = make_tool(), make_tool()
    (shared_tool, shared_name), (own_tool, own_name) = make_tool(), make_tool()
    first = make_model(False, (first_tool, shared_tool))
    second = make_model(True, (second_tool, shared_tool))
    page = open_on(page_for, make_user, first)
    expect_tools(page, on=[first_name, shared_name], off=[second_name, own_name])

    menu = open_tools(page)
    for name, checked in ((shared_name, "false"), (own_name, "true")):
        tool_switch(menu, name).click()
        expect(tool_switch(menu, name)).to_have_attribute("aria-checked", checked)
    close_tools(page, menu)
    pick_model(page, first, second)

    expect_tools(page, on=[second_name, own_name], off=[first_name, shared_name])
    assert web_search_shown_on(page), "the new model's default Web Search stayed off"
    sent = sent_request(page, upstream, "after the switch")
    sent_tools = sorted(sent.get("tool_ids") or [])
    assert sent_tools == sorted([second_tool, own_tool]), (
        f"sent tools {sent_tools}, not the new default {second_tool} and own {own_tool}"
    )
    assert (sent.get("features") or {}).get("web_search") is True, "Web Search was not sent on"


@pytest.mark.journey
def test_comparing_two_models_turns_on_the_default_tools_of_both(
    page_for, make_user, make_model, make_tool, upstream
):
    (first_tool, first_name), (second_tool, second_name) = make_tool(), make_tool()
    first = make_model(False, (first_tool,))
    second = make_model(False, (second_tool,))
    page = open_on(page_for, make_user, first)

    pick_model(page, first, second, compare=True)

    expect_tools(page, on=[first_name, second_name], off=[])
    sent = sent_request(page, upstream, "both of you")
    sent_tools = sorted(sent.get("tool_ids") or [])
    assert sent_tools == sorted([first_tool, second_tool]), (
        f"sent tools {sent_tools}, not the defaults {first_tool} and {second_tool} of both models"
    )
