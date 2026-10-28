"""Journey: parameters set in a chat's Controls reach the model and stay with that chat only.

Under Controls, Advanced Params switches a parameter from Default to a value of the chat's own:
a temperature and a seed set there are sent to the provider with the next message and with the
messages after a reload, and so are top_p, max_tokens, min_p, the two penalties, the stop
sequences (comma separated, `\\n` read as a line break, a `%` kept as typed), the reasoning
effort and the logit bias (`token:bias` pairs sent as a map). A custom parameter an admin adds
there is sent under its name, its value read as JSON when it is JSON and kept as text when it is
not. The chat's own system prompt is sent in place of none, again after a reload. A new chat
starts from the defaults again, and going back and forth between two chats in the sidebar keeps
each chat's values to itself. Two tabs open on one chat that each change a different setting
keep both, and the chat opened again shows and sends both (#32755, b4a58be19).

The parameters an account saves under Settings > General > Advanced Parameters are sent with
every chat that sets none of its own; a value set in the chat's Controls wins over them, and a
chat value switched back to Default leaves the account's value in force, as on a chat that never
changed it.

The switched-back test is red on dev ebc6add67: Default stores the chat's value as null, and that
null replaces the account's value in the request, so no temperature is sent at all
(open-webui/open-webui#32013, fixed by c2a5c7a24).

Discriminates: every other test passes on dev ebc6add67. On a build that sends only the account's
parameters every chat-value test fails (and the switched-back test passes); on one that ignores a
loaded chat's stored parameters the reload, sidebar and system prompt tests fail; on one that
sends the stop field as one word the sampling test fails. In the pre-fix build of 0.12.4
(fdb304be2 reverted) a stop sequence holding `%` keeps the message from sending, so the percent
sign test gets no reply. In a backend copy that keeps custom
values as text both custom parameter tests fail, and one that drops the parameters from a saved
account setting fails the three account tests. On dev 948432b43, a backend copy whose param save
replaces the chat's stored settings with the sent one fails the two tabs test (the system prompt
is gone).
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor
from harness.chat import ask
from utils.chat_ui import REPLY_TIMEOUT_MS, chat_input, conversation, expect_reply, send
from utils.setting_rows import setting_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT_PROMPT = "You answer like a lighthouse keeper."


def _advanced_params(page: Page) -> Locator:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("navigation").get_by_role("button", name="Controls").click()
    # the section starts open
    expect(page.get_by_role("button", name="Advanced Params")).to_have_attribute(
        "aria-expanded", "true"
    )
    expect(page.get_by_text("Temperature", exact=True)).to_be_visible()
    return page.locator("body")


def _set(panel: Locator, label: str, value: str) -> None:
    setting_button(panel, label).click()
    panel.get_by_role("spinbutton", name=label).fill(value)


def _set_text(panel: Locator, label: str, value: str) -> None:
    setting_button(panel, label).click()
    panel.get_by_role("textbox", name=label).fill(value)


def _ask(page: Page, upstream, topic: str = "how cold is the sea today?") -> dict:
    question = f"{topic} {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Brisk.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Brisk.")
    return next(filter(reply.answering(question), upstream.chat_requests()))


def _system_prompts(request: dict) -> list[str]:
    return [entry["content"] for entry in request["messages"] if entry["role"] == "system"]


def _sidebar_chat(page: Page, title: str) -> Locator:
    open_sidebar = page.get_by_role("button", name="Open Sidebar", exact=True)
    if open_sidebar.is_visible():
        open_sidebar.click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    return sidebar.get_by_role("button", name=title).first


def test_a_temperature_and_seed_set_in_the_chat_are_sent_and_kept(page_for, make_user, upstream):
    page = page_for(make_user())
    panel = _advanced_params(page)
    _set(panel, "Temperature", "0.3")
    _set(panel, "Seed", "4242")

    first = _ask(page, upstream)
    assert (first.get("temperature"), first.get("seed")) == (0.3, 4242)

    expect(page).to_have_url(re.compile(r"/c/"))
    page.reload()
    expect_reply(page, "Brisk.")
    later = _ask(page, upstream)
    assert (later.get("temperature"), later.get("seed")) == (0.3, 4242)


def test_every_sampling_parameter_set_in_the_chat_reaches_the_provider(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    panel = _advanced_params(page)
    _set(panel, "top_p", "0.4")
    _set(panel, "max_tokens", "321")
    _set(panel, "min_p", "0.05")
    _set(panel, "frequency_penalty", "0.5")
    _set(panel, "presence_penalty", "0.25")
    _set_text(panel, "Stop Sequence", "END,HALT,\\n")
    _set_text(panel, "Reasoning Effort", "high")
    _set_text(panel, "logit_bias", "5432:100, 413:-100")

    request = _ask(page, upstream)

    sent = {key: request.get(key) for key in EXPECTED_SAMPLING}
    assert sent == EXPECTED_SAMPLING


EXPECTED_SAMPLING = {
    "top_p": 0.4,
    "max_tokens": 321,
    "min_p": 0.05,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.25,
    "stop": ["END", "HALT", "\n"],
    "reasoning_effort": "high",
    "logit_bias": {"5432": 100, "413": -100},
}


def test_a_stop_sequence_with_a_percent_sign_is_sent_as_typed(page_for, make_user, upstream):
    """#32649 (fix fdb304be2): a `%` in a stop sequence kept every message from sending."""
    page = page_for(make_user())
    _set_text(_advanced_params(page), "Stop Sequence", "100%,END")

    request = _ask(page, upstream)

    assert request.get("stop") == ["100%", "END"]


def test_an_admins_custom_parameter_is_sent_under_its_name(page_for, make_user, upstream):
    page = page_for(make_user(role="admin"))
    panel = _advanced_params(page)
    panel.get_by_role("button", name="Add Custom Parameter").click()
    panel.get_by_placeholder("Custom Parameter Name").fill("top_k")
    panel.get_by_placeholder("Custom Parameter Value").fill("7")

    request = _ask(page, upstream)
    assert request.get("top_k") == 7


def test_custom_parameters_are_sent_as_json_or_as_text(page_for, make_user, upstream):
    page = page_for(make_user(role="admin"))
    panel = _advanced_params(page)
    panel.get_by_role("button", name="Add Custom Parameter").click()
    panel.get_by_placeholder("Custom Parameter Name").last.fill("chat_template_kwargs")
    panel.get_by_placeholder("Custom Parameter Value").last.fill(
        '{"enable_thinking": false, "levels": [1, 2]}'
    )
    panel.get_by_role("button", name="Add Custom Parameter").click()
    panel.get_by_placeholder("Custom Parameter Name").last.fill("verbosity")
    panel.get_by_placeholder("Custom Parameter Value").last.fill("low and slow")

    request = _ask(page, upstream)

    assert request.get("chat_template_kwargs") == {"enable_thinking": False, "levels": [1, 2]}
    assert request.get("verbosity") == "low and slow"


def test_a_new_chat_starts_from_the_defaults(page_for, make_user, upstream):
    page = page_for(make_user())
    panel = _advanced_params(page)
    _set(panel, "Temperature", "0.3")
    assert _ask(page, upstream).get("temperature") == 0.3

    page.get_by_role("link", name="New Chat").first.click()
    expect(page).not_to_have_url(re.compile(r"/c/"))
    fresh = _ask(page, upstream)
    assert "temperature" not in fresh


def test_each_chat_keeps_its_own_values_when_switching_in_the_sidebar(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    panel = _advanced_params(page)
    _set(panel, "Temperature", "0.3")
    assert _ask(page, upstream, "tuned chat").get("temperature") == 0.3

    page.get_by_role("link", name="New Chat").first.click()
    expect(page).not_to_have_url(re.compile(r"/c/"))
    assert "temperature" not in _ask(page, upstream, "plain chat")

    _sidebar_chat(page, "tuned chat").click()
    expect(conversation(page).get_by_text("tuned chat")).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    expect(setting_button(panel, "Temperature")).to_have_text("Custom")
    expect(panel.get_by_role("spinbutton", name="Temperature")).to_have_value("0.3")
    assert _ask(page, upstream, "tuned again").get("temperature") == 0.3

    _sidebar_chat(page, "plain chat").click()
    expect(conversation(page).get_by_text("plain chat")).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    expect(setting_button(panel, "Temperature")).to_have_text("Default")
    assert "temperature" not in _ask(page, upstream, "plain again")


def test_a_chats_system_prompt_is_kept_after_a_reload_and_stays_in_that_chat(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    _advanced_params(page)
    page.get_by_role("textbox", name="Enter system prompt").fill(CHAT_PROMPT)
    assert _system_prompts(_ask(page, upstream)) == [CHAT_PROMPT]

    expect(page).to_have_url(re.compile(r"/c/"))
    page.reload()
    expect_reply(page, "Brisk.")
    assert _system_prompts(_ask(page, upstream)) == [CHAT_PROMPT]

    page.get_by_role("link", name="New Chat").first.click()
    expect(page).not_to_have_url(re.compile(r"/c/"))
    assert _system_prompts(_ask(page, upstream)) == []


@contextmanager
def _saved_setting(page: Page) -> Iterator[None]:
    """Waits until the chat's Controls change the block makes is stored."""
    with page.expect_response(
        lambda response: response.url.endswith("/params") and response.request.method == "PATCH"
    ) as saved:
        yield
    assert saved.value.ok, saved.value.text()


def test_two_tabs_changing_different_settings_keep_both(page_for, make_user, upstream):
    account = make_user()
    opener = f"good morning {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Fog again.", match=reply.answering(opener)))
    with account.client() as client:
        turn, _ = ask(client, opener)
    first = page_for(account, path=f"/c/{turn.chat_id}")
    second = page_for(account, path=f"/c/{turn.chat_id}")
    expect_reply(first, "Fog again.")
    expect_reply(second, "Fog again.")

    _advanced_params(first)
    prompt_box = first.get_by_role("textbox", name="Enter system prompt")
    with _saved_setting(first):
        prompt_box.fill(CHAT_PROMPT)
        prompt_box.blur()
    panel = _advanced_params(second)
    with _saved_setting(second):
        setting_button(panel, "Temperature").click()

    with account.client() as client:
        stored = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"].get("params", {})
    assert (stored.get("system"), stored.get("temperature")) == (CHAT_PROMPT, 0.8), (
        f"one tab's change wiped the other's: {stored} (#32755)"
    )

    reopened = page_for(account, path=f"/c/{turn.chat_id}")
    expect_reply(reopened, "Fog again.")
    panel = _advanced_params(reopened)
    expect(reopened.get_by_role("textbox", name="Enter system prompt")).to_have_value(CHAT_PROMPT)
    expect(panel.get_by_role("spinbutton", name="Temperature")).to_have_value("0.8")
    request = _ask(reopened, upstream)
    assert (_system_prompts(request), request.get("temperature")) == ([CHAT_PROMPT], 0.8)


# --------------------------------------------------------------------------- account defaults


def _save_account_parameters(page: Page, values: dict[str, str]) -> None:
    """Sets each spinbutton parameter under Settings > General and saves."""
    page.goto("/?settings=general")
    settings = page.get_by_role("dialog")
    setting_button(settings, "Model parameters").click()
    for label, value in values.items():
        _set(settings, label, value)
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        settings.get_by_role("button", name="Save", exact=True).click()
    page.goto("/")


@pytest.fixture
def tuned_account(make_user) -> Actor:
    return make_user()


def test_the_accounts_saved_parameters_reach_a_new_chat(page_for, tuned_account, upstream):
    page = page_for(tuned_account)
    _save_account_parameters(page, {"Temperature": "0.6", "max_tokens": "200"})

    request = _ask(page, upstream)

    assert (request.get("temperature"), request.get("max_tokens")) == (0.6, 200)


def test_a_chat_value_wins_over_the_accounts_saved_one(page_for, tuned_account, upstream):
    page = page_for(tuned_account)
    _save_account_parameters(page, {"Temperature": "0.6", "max_tokens": "200"})
    _set(_advanced_params(page), "Temperature", "0.2")

    request = _ask(page, upstream)

    assert (request.get("temperature"), request.get("max_tokens")) == (0.2, 200)


def test_a_chat_value_switched_back_to_default_leaves_the_accounts_one(
    page_for, tuned_account, upstream
):
    page = page_for(tuned_account)
    _save_account_parameters(page, {"Temperature": "0.6"})
    panel = _advanced_params(page)
    _set(panel, "Temperature", "0.2")
    setting_button(panel, "Temperature").click()
    expect(setting_button(panel, "Temperature")).to_have_text("Default")

    request = _ask(page, upstream)

    assert request.get("temperature") == 0.6, (
        "Temperature reads Default in the chat's Controls, yet the account's saved 0.6 was not "
        "sent (open-webui/open-webui#32013)"
    )
