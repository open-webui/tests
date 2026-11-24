"""Journey: an admin sets a router model up and a user is answered by the model each rule picks.

Admin Settings > Models > Router Models > Add Router Model fills the Model ID from the name, warns
until there is a default model, offers only real models as targets, shows the Decisions API (with
its default address, model, threshold and the Prompt box) once there is a condition, and is
private until the admin makes it public. Saved, it is listed under Router Models; reopened, every
field reads as saved, the API key masked, and an edit is kept after a reload. A user picks it in
the model selector; a message a condition fits goes to that condition's model, a message nothing
fits to the default model, and a message matching a pattern to that pattern's model without a
Decisions API call. Each reply names the router and the model that answered. The dialog refuses
an invalid pattern and a threshold above 1 and says why, and shows Strategy
once there are two default models. A router deleted in its dialog leaves the user's selector; its
chat still opens with its history, and the selector there shows the model that answers the next
message, as for any chat whose model is gone. A name translated in the dialog's Editing language
is what a user of that language sees in the selector.
Behaviour from docs/features/administration/router-models.md and the dev dialog.

The docs branch `docs-sync-router-automations` drops the Prompt box; this checks the box the dev
dialog shows (placeholder "Leave empty to use the default prompt, or enter a custom prompt").

One test is red on dev d599f7a08: an invalid pattern only shows "[object Object]"
(open-webui/open-webui#34897).

Discriminates (dev d599f7a08, one patched copy of the built frontend): a reply header without the
answering model turns the setup test red; Strategy shown for a single default model turns the
threshold test red; translations left out of the saved router turn the translation test red; a
Delete that saves the list unchanged turns the delete test red.
"""

from __future__ import annotations

import re
import uuid
from typing import Callable, Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor
from harness.router_models import (
    DECIDER_KEY,
    DECIDER_MODEL,
    EVERYONE_READS,
    ROUTER_MODELS_CONFIG,
    Decider,
    Target,
    create_target,
    default_routes,
    queue_target_answers,
    save_router_model,
)
from utils.access_control import choose_visibility
from utils.chat_ui import conversation, expect_reply, last_reply, send
from utils.model_selector import SELECTOR_BUTTON, model_options, select_model

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

LEGAL = "The user asks about contracts or other legal questions"
NO_DEFAULT_WARNING = "Add a default model, or requests that match nothing will fail."
PATTERN_PROBLEM = re.compile("pattern|regular expression", re.I)
PROMPT_PLACEHOLDER = "Leave empty to use the default prompt, or enter a custom prompt"


@pytest.fixture
def make_target(admin) -> Iterator[Callable[[str], Target]]:
    made: list[str] = []

    def create(label: str) -> Target:
        with admin.client() as client:
            target = create_target(client, label)
            client.get("/api/models", params={"refresh": True}).raise_for_status()
        made.append(target.id)
        return target

    yield create
    with admin.client() as client:
        for model_id in made:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def decider(listener) -> Decider:
    return Decider(listener)


def open_router_dialog(page: Page) -> Locator:
    page.get_by_role("button", name="Add Router Model").click()
    dialog = page.get_by_role("dialog", name="Add Router Model")
    expect(dialog).to_be_visible()
    return dialog


def open_section(dialog: Locator, label: str) -> None:
    """Open a closed section of the dialog (Patterns, Conditions, Default, Decisions API)."""
    section = dialog.locator("details").filter(
        has=dialog.page.locator("summary", has_text=re.compile(rf"^{re.escape(label)}"))
    )
    if section.get_attribute("open") is None:
        section.locator("summary").click()
    expect(section).to_have_attribute("open", "")


def model_selects(dialog: Locator) -> Locator:
    return dialog.get_by_role("combobox", name="Select a model")


def router_models_list(page: Page) -> Locator:
    """The Router Models panel of Admin Settings > Models, opened."""
    toggle = page.get_by_role("button", name="Router Models").and_(page.locator("[aria-expanded]"))
    if toggle.get_attribute("aria-expanded") == "false":
        toggle.click()
    expect(toggle).to_have_attribute("aria-expanded", "true")
    return toggle.locator("xpath=../..")


def toasts(page: Page) -> Locator:
    return page.locator("[data-sonner-toast]")


def saved_router(admin: Actor, router_id: str) -> dict | None:
    with admin.client() as client:
        saved = client.get(ROUTER_MODELS_CONFIG[0]).json()["ROUTER_MODELS"]
    return next((model for model in saved if model["id"] == router_id), None)


def reply_header(page: Page) -> Locator:
    """The last reply's model names and text."""
    return last_reply(page).locator("xpath=../..")


def test_an_admin_sets_up_a_router_in_its_dialog_and_a_user_gets_the_model_each_rule_picks(
    admin, make_user, page_for, preserve, upstream, decider, make_target
):
    preserve(ROUTER_MODELS_CONFIG)
    code, legal, general = make_target("code"), make_target("legal"), make_target("general")
    targets = [code, legal, general]
    tag = uuid.uuid4().hex[:6]
    other_router = f"Other Router {tag}"
    with admin.client() as client:
        save_router_model(
            client, f"other-router-{tag}", other_router, default_routes(general.id), []
        )
    name, router_id = f"Smart Router {tag}", f"smart-router-{tag}"
    router_admin = make_user(role="admin")
    page = page_for(router_admin, path="/admin/settings/models")

    dialog = open_router_dialog(page)
    dialog.get_by_placeholder("Model Name").fill(name)
    expect(dialog.get_by_role("textbox", name="Model ID")).to_have_value(router_id)
    expect(dialog.get_by_text(NO_DEFAULT_WARNING)).to_be_visible()

    open_section(dialog, "Patterns")
    dialog.get_by_role("button", name="Add pattern").click()
    pattern_target = model_selects(dialog).first
    expect(pattern_target.locator("option", has_text=code.name)).to_have_count(1)
    expect(pattern_target.locator("option", has_text=other_router)).to_have_count(0)
    pattern_target.select_option(label=code.name)
    dialog.get_by_role("textbox", name="Regular expression").fill("```")

    open_section(dialog, "Conditions")
    dialog.get_by_role("button", name="Add condition").click()
    model_selects(dialog).nth(1).select_option(label=legal.name)
    dialog.get_by_role("textbox", name="Condition").fill("Legal questions")

    dialog.get_by_role("button", name="Add model").click()
    model_selects(dialog).last.select_option(label=general.name)
    expect(dialog.get_by_text(NO_DEFAULT_WARNING)).to_have_count(0)

    open_section(dialog, "Decisions API")
    base_url = dialog.get_by_role("textbox", name="API Base URL")
    expect(base_url).to_have_value("https://api.openai.com/v1")
    expect(dialog.get_by_role("textbox", name="Model", exact=True)).to_have_value("gpt-6-luna")
    expect(dialog.get_by_role("spinbutton", name="Threshold")).to_have_value("0.5")
    prompt_box = dialog.get_by_role("textbox", name="Prompt")
    expect(prompt_box).to_have_attribute("placeholder", PROMPT_PLACEHOLDER)
    base_url.fill(decider.base_url)
    dialog.get_by_placeholder("API Key").fill(DECIDER_KEY)
    dialog.get_by_role("textbox", name="Model", exact=True).fill(DECIDER_MODEL)
    dialog.get_by_role("spinbutton", name="Threshold").fill("0.6")

    dialog.get_by_role("button", name="Access", exact=True).click()
    access = page.get_by_role("dialog", name="Access Control")
    expect(access.get_by_text("Private to you")).to_be_visible()
    choose_visibility(access, "Public")
    expect(access.get_by_text("Accessible to all users")).to_be_visible()
    page.keyboard.press("Escape")
    expect(access).to_be_hidden()

    with page.expect_response(lambda response: "/configs/router_models" in response.url):
        dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog).to_be_hidden()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()

    listed = router_models_list(page).get_by_role("button", name=re.compile(re.escape(name)))
    expect(listed).to_contain_text(router_id)

    listed.click()
    edit = page.get_by_role("dialog", name="Edit Router Model")
    expect(edit.get_by_placeholder("Model Name")).to_have_value(name)
    expect(edit.get_by_role("textbox", name="Model ID")).to_have_value(router_id)
    open_section(edit, "Patterns")
    open_section(edit, "Conditions")
    open_section(edit, "Default")
    open_section(edit, "Decisions API")
    expect(edit.get_by_role("textbox", name="Regular expression")).to_have_value("```")
    expect(model_selects(edit)).to_have_count(3)
    for select, target in zip(model_selects(edit).all(), targets):
        expect(select).to_have_value(target.id)
    expect(edit.get_by_role("textbox", name="API Base URL")).to_have_value(decider.base_url)
    key = edit.get_by_placeholder("API Key")
    assert key.evaluate("input => getComputedStyle(input).webkitTextSecurity") == "disc", (
        "the saved API key is shown in clear text"
    )
    expect(key).to_have_value(DECIDER_KEY)
    expect(edit.get_by_role("textbox", name="Model", exact=True)).to_have_value(DECIDER_MODEL)
    expect(edit.get_by_role("spinbutton", name="Threshold")).to_have_value("0.6")
    edit.get_by_role("textbox", name="Condition").fill(LEGAL)
    with page.expect_response(lambda response: "/configs/router_models" in response.url):
        edit.get_by_role("button", name="Save", exact=True).click()
    expect(edit).to_be_hidden()

    page = page_for(router_admin, path="/admin/settings/models")
    router_models_list(page).get_by_role("button", name=re.compile(re.escape(name))).click()
    edit = page.get_by_role("dialog", name="Edit Router Model")
    open_section(edit, "Conditions")
    expect(edit.get_by_role("textbox", name="Condition")).to_have_value(LEGAL)

    user = make_user()
    chat = page_for(user)
    select_model(chat, name)
    contract = f"Can I end my lease early? {tag}"
    decider.scores = {LEGAL: 0.9}
    queue_target_answers(upstream, contract, targets)
    send(chat, contract)
    expect_reply(chat, legal.answer)
    expect(reply_header(chat)).to_contain_text(name)
    expect(reply_header(chat)).to_contain_text(legal.name)
    assert len(decider.requests()) == 1

    weather = f"And what is the weather like? {tag}"
    decider.scores = {LEGAL: 0.1}
    queue_target_answers(upstream, weather, targets)
    send(chat, weather)
    expect_reply(chat, general.answer)
    expect(reply_header(chat)).to_contain_text(general.name)
    assert len(decider.requests()) == 2

    snippet = f"Why does ```print(1``` fail? {tag}"
    second_chat = page_for(user)
    select_model(second_chat, name)
    queue_target_answers(upstream, snippet, targets)
    send(second_chat, snippet)
    expect_reply(second_chat, code.answer)
    expect(reply_header(second_chat)).to_contain_text(code.name)
    assert len(decider.requests()) == 2, "a pattern match still asked the Decisions API"


@pytest.fixture
def strict_router(admin, make_user, page_for, preserve, make_target):
    """A new router in its dialog, with a default model and a pattern row for the code model.

    Also makes a spare model to add, as the dialog only offers the models there were on opening."""
    preserve(ROUTER_MODELS_CONFIG)
    code, general, spare = make_target("code"), make_target("general"), make_target("spare")
    tag = uuid.uuid4().hex[:6]
    page = page_for(make_user(role="admin"), path="/admin/settings/models")
    dialog = open_router_dialog(page)
    dialog.get_by_placeholder("Model Name").fill(f"Strict Router {tag}")
    dialog.get_by_role("button", name="Add model").click()
    model_selects(dialog).last.select_option(label=general.name)
    open_section(dialog, "Patterns")
    dialog.get_by_role("button", name="Add pattern").click()
    model_selects(dialog).first.select_option(label=code.name)
    return page, dialog, f"strict-router-{tag}", general, spare


def save_button(dialog: Locator) -> Locator:
    return dialog.get_by_role("button", name="Save", exact=True)


def test_saving_an_invalid_pattern_says_what_is_wrong(admin, strict_router):
    page, dialog, router_id, *_ = strict_router
    dialog.get_by_role("textbox", name="Regular expression").fill("([unclosed")

    with page.expect_response(lambda response: "/configs/router_models" in response.url):
        save_button(dialog).click()

    # dev shows "[object Object]" (issue not filed yet)
    expect(toasts(page).filter(has_text=PATTERN_PROBLEM).first).to_be_visible()
    expect(dialog).to_be_visible()
    assert saved_router(admin, router_id) is None, "a router with an invalid pattern was saved"


def test_the_dialog_refuses_a_threshold_above_one_and_offers_strategy_for_two_defaults(
    admin, strict_router
):
    page, dialog, router_id, general, spare = strict_router
    dialog.get_by_role("textbox", name="Regular expression").fill("```")
    expect(dialog.get_by_role("combobox", name="Strategy")).to_have_count(0)
    dialog.get_by_role("button", name="Add model").click()
    model_selects(dialog).last.select_option(label=spare.name)
    expect(dialog.get_by_role("combobox", name="Strategy")).to_have_value("first")

    open_section(dialog, "Conditions")
    dialog.get_by_role("button", name="Add condition").click()
    model_selects(dialog).nth(1).select_option(label=general.name)
    dialog.get_by_role("textbox", name="Condition").fill(LEGAL)
    open_section(dialog, "Decisions API")
    threshold = dialog.get_by_role("spinbutton", name="Threshold")
    threshold.fill("1.5")
    save_button(dialog).click()
    expect(dialog).to_be_visible()
    assert threshold.evaluate("input => input.validationMessage"), (
        "a threshold of 1.5 got no message"
    )
    assert saved_router(admin, router_id) is None, "a router with a threshold of 1.5 was saved"

    threshold.fill("0.5")
    with page.expect_response(lambda response: "/configs/router_models" in response.url):
        save_button(dialog).click()
    expect(dialog).to_be_hidden()
    saved = saved_router(admin, router_id)
    assert saved is not None, "the corrected router was not saved"
    assert [route.get("pattern") for route in saved["meta"]["routes"]][0] == "```"


def test_a_router_deleted_in_its_dialog_leaves_the_selector_and_its_chat_goes_on_with_another_model(
    admin, make_user, page_for, preserve, upstream, make_target
):
    preserve(ROUTER_MODELS_CONFIG)
    general = make_target("general")
    tag = uuid.uuid4().hex[:6]
    name = f"Doomed Router {tag}"
    with admin.client() as client:
        save_router_model(
            client,
            f"doomed-router-{tag}",
            name,
            default_routes(general.id),
            [EVERYONE_READS],
            strategy="first",
        )
    user = make_user()
    chat = page_for(user)
    select_model(chat, name)
    prompt = f"Remember the number seven {tag}"
    queue_target_answers(upstream, prompt, [general])
    send(chat, prompt)
    expect_reply(chat, general.answer)
    expect(chat).to_have_url(re.compile(r"/c/"))
    chat_path = re.sub(r"^https?://[^/]+", "", chat.url)

    page = page_for(make_user(role="admin"), path="/admin/settings/models")
    router_models_list(page).get_by_role("button", name=re.compile(re.escape(name))).click()
    edit = page.get_by_role("dialog", name="Edit Router Model")
    edit.get_by_role("button", name="Delete").click()
    with page.expect_response(lambda response: "/configs/router_models" in response.url):
        page.get_by_role("dialog").get_by_role("button", name="Confirm").click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()

    picker = page_for(user)
    expect(model_options(picker, general.name)).to_have_count(1)
    expect(model_options(picker, name)).to_have_count(0)

    reopened = page_for(user, path=chat_path)
    expect(conversation(reopened).get_by_text(prompt)).to_be_visible()
    expect(last_reply(reopened)).to_contain_text(general.answer)
    shown = reopened.get_by_role("button", name=SELECTOR_BUTTON)
    expect(shown).not_to_contain_text(name)
    answering = shown.inner_text().strip()
    assert answering, "the reopened chat shows no model to answer the next message"
    follow_up = f"Are you still there? {tag}"
    upstream.queue(reply.text("Still here.", match=reply.answering(follow_up)))
    send(reopened, follow_up)
    expect_reply(reopened, "Still here.")
    expect(reply_header(reopened)).to_contain_text(answering)


def offered_names(page: Page, search: str) -> Locator:
    """The selector's options after searching for `search`, in whatever language the page is."""
    page.locator("[id^='model-selector-'][id$='-button']").first.click()
    page.locator("#model-search-input").fill(search)
    return page.locator("#model-selector-list").get_by_role("option")


def in_german(page: Page) -> Page:
    """Pick Deutsch in Settings > General, then reload as a returning visit would."""
    page.goto("/?settings=general")
    page.locator("#tab-general").get_by_role("combobox", name="Language").select_option("de-DE")
    expect(page.locator("html")).to_have_attribute("lang", "de-DE")
    page.reload()
    expect(page.locator("html")).to_have_attribute("lang", "de-DE")
    return page


def test_a_router_name_translated_in_its_dialog_shows_in_the_users_language(
    make_user, page_for, preserve, make_target
):
    preserve(ROUTER_MODELS_CONFIG)
    general = make_target("general")
    tag = uuid.uuid4().hex[:6]
    english, german = f"Smart Router {tag}", f"Kluger Router {tag}"
    page = page_for(make_user(role="admin"), path="/admin/settings/models")
    dialog = open_router_dialog(page)
    dialog.get_by_role("textbox", name="Model Name").fill(english)
    dialog.get_by_role("button", name="Add model").click()
    model_selects(dialog).last.select_option(label=general.name)
    dialog.get_by_role("combobox", name="Editing language").select_option("de-DE")
    dialog.get_by_role("textbox", name="Model Name").fill(german)
    dialog.get_by_role("button", name="Access", exact=True).click()
    access = page.get_by_role("dialog", name="Access Control")
    choose_visibility(access, "Public")
    page.keyboard.press("Escape")
    expect(access).to_be_hidden()
    with page.expect_response(lambda response: "/configs/router_models" in response.url):
        dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog).to_be_hidden()

    english_user = page_for(make_user())
    expect(offered_names(english_user, tag)).to_have_text([english])

    german_user = in_german(page_for(make_user()))
    expect(offered_names(german_user, tag)).to_have_text([german])
