"""Journey: picking a model's controls in the chat input, and what the provider is sent.

An admin gives a model controls, each a set of approved options that send parameters. The chat
input of a chat on that model shows a Model controls button; its menu lists each control with the
option in use, and a submenu offers Default (naming the default option) and every option. A pick
is saved to the account's settings at once, so it is sent with the next message, kept after a
reload and in later chats, and only for the model it was made on: in a chat with two models the
menu groups the controls by model and each model is sent its own pick, and a model without
controls shows no button. On a phone a control's options open in place of the list, with a way
back, and the Chat Variables button beside it still opens its own form. The picked option's
parameters go out alongside what the chat's own Controls set. A user without the Chat Controls
or Advanced Params permission gets no button and the defaults still apply.

Four red tests pin picks the user can no longer see or clear. Saving Settings > General throws
every pick away (open-webui/open-webui#31997). A pick of a control the admin later removed fails
every message on that model ("Model control thinking: the selected option is no longer
available."), and the menu no longer lists the control to reset it (open-webui/open-webui#31996).
A pick made before the admin withdrew Chat Controls or Advanced Params gets every message to any
model refused ("You cannot change model parameters."): the button is gone and the server drops
the account's parameter settings, so the pick cannot be cleared (open-webui/open-webui#31995).

Discriminates: passes on the dev ebc6add67 build except the four red tests. In a frontend build
whose chat sends no picks, whose menu shows for every model and to every account and whose
submenu drops the description, every test fails but the default, other account and Chat Variables
tests; in a backend copy that never merges the picks, every test that reads the request fails;
in a frontend build without the Chat Variables button and whose pick is never saved to the
account, the Chat Variables, reload and new chat, and model switching tests fail.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.model_controls import LENGTH, THINKING, pick, preset_with_controls
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import REPLY_TIMEOUT_MS, chat_input, expect_reply, replies, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CONTROLS_BUTTON = "Model controls"


def _open_chat(page: Page, *model_ids: str) -> None:
    page.goto(f"/?models={','.join(model_ids)}")
    expect(chat_input(page)).to_be_visible()


def _controls_button(page: Page) -> Locator:
    return page.get_by_label(CONTROLS_BUTTON, exact=True)


def _open_menu(page: Page) -> Locator:
    _controls_button(page).click()
    menu = page.get_by_role("menu").first
    expect(menu).to_be_visible()
    return menu


def _control_row(page: Page, label: str) -> Locator:
    return page.get_by_role("button", name=label, exact=True)


def _choices(page: Page, control_label: str) -> Locator:
    """Open the submenu of `control_label` in the open menu; its option rows."""
    _control_row(page, control_label).click()
    options = page.get_by_role("menuitemradio")
    expect(options.first).to_be_visible()
    return options


def _choose(page: Page, control_label: str, option_label: str, row: int = 0) -> None:
    """Pick `option_label` of the `row`-th control named `control_label`, and wait for the save."""
    _open_menu(page)
    _control_row(page, control_label).nth(row).click()
    option = page.get_by_role("menuitemradio", name=option_label, exact=True)
    with page.expect_response(re.compile(r"/api/v1/users/user/settings/update")) as saved:
        option.click()
    assert saved.value.ok, saved.value.text()
    expect(option).to_have_attribute("aria-checked", "true")
    page.keyboard.press("Escape")
    page.keyboard.press("Escape")


def _ask(page: Page, upstream, answer: str = "Sure.") -> dict:
    question = f"what does the tide do tonight? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    send(page, question)
    expect_reply(page, answer)
    return next(filter(reply.answering(question), upstream.chat_requests()))


@pytest.fixture
def tuned(admin):
    with preset_with_controls(admin, {"thinking": THINKING, "length": LENGTH}) as model:
        yield model


# --------------------------------------------------------------------------- the menu


def test_the_menu_lists_each_control_with_the_option_in_use(page_for, make_user, tuned):
    page = page_for(make_user())
    _open_chat(page, tuned["id"])

    _open_menu(page)
    expect(_control_row(page, "Thinking")).to_contain_text("Medium")
    expect(_control_row(page, "Answer length")).to_contain_text("Default")

    options = _choices(page, "Thinking")
    expect(page.get_by_text(THINKING["description"])).to_be_visible()
    expect(options).to_have_text(["Default · Medium", "Low", "Medium", "High"])
    expect(options.first).to_have_attribute("aria-checked", "true")


def test_a_model_without_controls_shows_no_button(page_for, make_user, tuned):
    page = page_for(make_user())
    _open_chat(page, MOCK_MODEL_ID)
    expect(page.get_by_role("button", name="Voice mode")).to_be_visible()
    expect(_controls_button(page)).to_have_count(0)

    _open_chat(page, tuned["id"])
    expect(_controls_button(page)).to_be_visible()


def test_chat_variables_and_model_controls_each_open_their_own(page_for, make_user, admin):
    system = (
        'You write for {{chat.variables.audience | text:placeholder="Audience":default="crew"}}.'
    )
    with preset_with_controls(admin, {"thinking": THINKING}, params={"system": system}) as model:
        page = page_for(make_user())
        _open_chat(page, model["id"])

        page.get_by_role("button", name="Chat Variables").click()
        variables = page.get_by_role("dialog").filter(has_text="Chat Variables")
        expect(variables.get_by_placeholder("Audience")).to_be_visible()
        page.keyboard.press("Escape")
        expect(variables).to_be_hidden()

        _open_menu(page)
        expect(_control_row(page, "Thinking")).to_contain_text("Medium")


# --------------------------------------------------------------------------- what is sent


def test_the_default_option_is_sent_until_the_user_picks(page_for, make_user, tuned, upstream):
    page = page_for(make_user())
    _open_chat(page, tuned["id"])

    sent = _ask(page, upstream)

    assert sent.get("reasoning_effort") == "medium"
    assert "verbosity" not in sent and "max_tokens" not in sent


def test_a_pick_on_the_new_chat_screen_is_sent_and_kept(page_for, make_user, tuned, upstream):
    account = make_user()
    page = page_for(account)
    _open_chat(page, tuned["id"])
    _choose(page, "Thinking", "High")
    _choose(page, "Answer length", "Thorough")

    first = _ask(page, upstream)
    assert (first.get("reasoning_effort"), first.get("max_tokens")) == ("high", 4096)
    assert (first.get("verbosity"), first.get("text")) == ("high", {"format": {"type": "text"}})

    expect(page).to_have_url(re.compile(r"/c/"))
    page.reload()
    expect_reply(page, "Sure.")
    _open_menu(page)
    expect(_control_row(page, "Thinking")).to_contain_text("High")
    page.keyboard.press("Escape")
    after_reload = _ask(page, upstream, "Again.")
    assert after_reload.get("reasoning_effort") == "high"

    _open_chat(page, tuned["id"])
    in_a_new_chat = _ask(page, upstream)
    assert in_a_new_chat.get("reasoning_effort") == "high"
    with account.client() as client:
        stored = client.get("/api/v1/users/user/settings").json()
    picks = {"thinking": "high", "length": "thorough"}
    assert stored["ui"]["params"]["model_controls"] == {tuned["id"]: picks}


def test_a_new_pick_mid_chat_applies_from_the_next_message(page_for, make_user, tuned, upstream):
    page = page_for(make_user())
    _open_chat(page, tuned["id"])
    assert _ask(page, upstream, "One.").get("reasoning_effort") == "medium"

    _choose(page, "Thinking", "Low")
    assert _ask(page, upstream, "Two.").get("reasoning_effort") == "low"

    _choose(page, "Thinking", "Default · Medium")
    assert _ask(page, upstream, "Three.").get("reasoning_effort") == "medium"


def test_another_accounts_picks_are_its_own(page_for, make_user, tuned, upstream):
    picker = make_user()
    pick(picker, {tuned["id"]: {"thinking": "low"}})
    page = page_for(make_user())
    _open_chat(page, tuned["id"])

    _open_menu(page)
    expect(_control_row(page, "Thinking")).to_contain_text("Medium")
    page.keyboard.press("Escape")
    assert _ask(page, upstream).get("reasoning_effort") == "medium"


def test_the_chats_own_controls_are_sent_with_the_pick(page_for, make_user, tuned, upstream):
    page = page_for(make_user())
    _open_chat(page, tuned["id"])
    _choose(page, "Thinking", "Low")
    page.get_by_role("navigation").get_by_role("button", name="Controls").click()
    expect(page.get_by_text("Temperature", exact=True)).to_be_visible()
    page.get_by_text("Temperature", exact=True).locator("xpath=..").get_by_role(
        "button", name="Default"
    ).click()
    page.get_by_role("spinbutton", name="Temperature").fill("0.3")

    sent = _ask(page, upstream)

    assert (sent.get("reasoning_effort"), sent.get("temperature")) == ("low", 0.3)


def test_on_a_phone_the_options_open_in_place_with_a_way_back(page_for, make_user, tuned, upstream):
    page = page_for(make_user())
    page.set_viewport_size({"width": 400, "height": 800})
    _open_chat(page, tuned["id"])
    _open_menu(page)

    _control_row(page, "Thinking").click()
    expect(_control_row(page, "Answer length")).to_have_count(0)
    with page.expect_response(re.compile(r"/api/v1/users/user/settings/update")):
        page.get_by_role("menuitemradio", name="Low", exact=True).click()
    page.get_by_role("button", name="Back", exact=True).click()
    expect(_control_row(page, "Thinking")).to_contain_text("Low")
    expect(_control_row(page, "Answer length")).to_be_visible()
    page.keyboard.press("Escape")

    assert _ask(page, upstream).get("reasoning_effort") == "low"


# --------------------------------------------------------------------------- several models


def test_each_model_keeps_its_own_pick_when_switching(page_for, make_user, admin, upstream):
    with (
        preset_with_controls(admin, {"thinking": THINKING}, name="Harbour pilot") as pilot,
        preset_with_controls(admin, {"thinking": THINKING}, name="Lighthouse keeper") as keeper,
    ):
        page = page_for(make_user())
        _open_chat(page, pilot["id"])
        _choose(page, "Thinking", "High")
        assert _ask(page, upstream, "Pilot.").get("reasoning_effort") == "high"

        _open_chat(page, keeper["id"])
        _open_menu(page)
        expect(_control_row(page, "Thinking")).to_contain_text("Medium")
        page.keyboard.press("Escape")
        assert _ask(page, upstream, "Keeper.").get("reasoning_effort") == "medium"

        _open_chat(page, pilot["id"])
        assert _ask(page, upstream, "Pilot again.").get("reasoning_effort") == "high"


def test_two_models_in_one_chat_are_each_sent_their_own_pick(page_for, make_user, admin, upstream):
    pilot_prompt, keeper_prompt = "You steer ships.", "You tend the light."
    controls = {"thinking": THINKING}
    with (
        preset_with_controls(
            admin, controls, name="Harbour pilot", params={"system": pilot_prompt}
        ) as pilot,
        preset_with_controls(
            admin, controls, name="Lighthouse keeper", params={"system": keeper_prompt}
        ) as keeper,
    ):
        page = page_for(make_user())
        _open_chat(page, pilot["id"], keeper["id"])
        menu = _open_menu(page)
        expect(menu.get_by_text("Harbour pilot", exact=True)).to_be_visible()
        expect(menu.get_by_text("Lighthouse keeper", exact=True)).to_be_visible()
        expect(_control_row(page, "Thinking")).to_have_count(2)
        page.keyboard.press("Escape")
        _choose(page, "Thinking", "Low", row=1)

        question = "who keeps watch tonight?"
        upstream.queue(
            reply.text("The pilot.", match=reply.answering(question)),
            reply.text("The keeper.", match=reply.answering(question)),
        )
        send(page, question)
        expect(replies(page)).to_have_count(2, timeout=REPLY_TIMEOUT_MS)
        expect(replies(page).nth(1)).not_to_have_text("", timeout=REPLY_TIMEOUT_MS)
        sent = [body for body in upstream.chat_requests() if reply.answering(question)(body)]

    efforts = {
        body["messages"][0]["content"]: body.get("reasoning_effort")
        for body in sent
        if body["messages"][0]["role"] == "system"
    }
    assert efforts == {pilot_prompt: "medium", keeper_prompt: "low"}


# --------------------------------------------------------------------------- permissions


def _set_chat_permissions(admin, **flags: bool) -> None:
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"].update(flags)
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text


WITHDRAWN = pytest.mark.parametrize("withheld", ["controls", "params"])


@WITHDRAWN
def test_without_the_permissions_the_defaults_apply_unseen(
    page_for, make_user, admin, tuned, preserve, upstream, withheld
):
    preserve("permissions")
    _set_chat_permissions(admin, **{withheld: False})
    page = page_for(make_user())
    _open_chat(page, tuned["id"])
    expect(page.get_by_role("button", name="Voice mode")).to_be_visible()

    expect(_controls_button(page)).to_have_count(0)
    assert _ask(page, upstream).get("reasoning_effort") == "medium"


# --------------------------------------------------------------------------- picks left behind


def _open_general_settings(page: Page) -> Locator:
    page.get_by_role("navigation", name="Chat history").get_by_label("User menu").click()
    page.get_by_role("button", name="Settings").click()
    page.get_by_role("tab", name="General").click()
    return page.get_by_role("dialog")


def test_saving_general_settings_keeps_the_picks(page_for, make_user, tuned, upstream):
    account = make_user()
    page = page_for(account)
    _open_chat(page, tuned["id"])
    _choose(page, "Thinking", "High")

    settings = _open_general_settings(page)
    with page.expect_response(re.compile(r"/api/v1/users/user/settings/update")):
        settings.get_by_role("button", name="Save", exact=True).click()
    page.keyboard.press("Escape")

    with account.client() as client:
        stored = client.get("/api/v1/users/user/settings").json()
    picks = stored["ui"].get("params", {}).get("model_controls")
    assert picks == {tuned["id"]: {"thinking": "high"}}, (
        "saving Settings > General threw away the account's model control picks "
        "(open-webui/open-webui#31997)"
    )
    assert _ask(page, upstream).get("reasoning_effort") == "high"


def _remove_control(admin, model: dict, key: str) -> None:
    with admin.client() as client:
        stored = client.get("/api/v1/models/model", params={"id": model["id"]}).json()
        stored["params"]["model_controls"].pop(key)
        saved = client.post("/api/v1/models/model/update", json=stored)
    assert saved.status_code == 200, saved.text


def _answers(page: Page, upstream, question: str, answer: str, failure: str) -> None:
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    send(page, question)
    expect(replies(page).last, failure).to_contain_text(answer, timeout=REPLY_TIMEOUT_MS)


def test_a_pick_of_a_removed_control_does_not_block_the_model(
    page_for, make_user, admin, tuned, upstream
):
    page = page_for(make_user())
    _open_chat(page, tuned["id"])
    _choose(page, "Thinking", "High")
    _remove_control(admin, tuned, "thinking")

    _open_chat(page, tuned["id"])
    _open_menu(page)
    expect(_control_row(page, "Thinking")).to_have_count(0)
    page.keyboard.press("Escape")

    _answers(
        page,
        upstream,
        "still there?",
        "Still here.",
        "a pick of a removed control fails every message on the model, and nothing clears it "
        "(open-webui/open-webui#31996)",
    )


@WITHDRAWN
def test_a_pick_made_before_the_permission_was_withdrawn_does_not_block_chats(
    page_for, make_user, admin, tuned, preserve, upstream, withheld
):
    preserve("permissions")
    page = page_for(make_user())
    _open_chat(page, tuned["id"])
    _choose(page, "Thinking", "High")
    _set_chat_permissions(admin, **{withheld: False})

    page.reload()
    _open_chat(page, MOCK_MODEL_ID)
    _answers(
        page,
        upstream,
        "can I still chat?",
        "You can.",
        "a pick kept from before the permission was withdrawn refuses every message to any model "
        "(open-webui/open-webui#31995)",
    )
