"""Journey: parameters set in a chat's Controls reach the model and stay with that chat only.

Under Controls, Advanced Params switches a parameter from Default to a value of the chat's own:
a temperature and a seed set there are sent to the provider with the next message and with the
messages after a reload, and a custom parameter an admin adds there is sent under its name, its
value read as JSON. A new chat starts from the defaults again and sends none of them.

Discriminates: passes on dev 30f3f6a8f; in a backend copy that drops the parameters a chat sends
with its messages all three tests fail.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


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
    panel.get_by_text(label, exact=True).locator("xpath=..").get_by_role(
        "button", name="Default"
    ).click()
    panel.get_by_role("spinbutton", name=label).fill(value)


def _ask(page: Page, upstream) -> dict:
    question = f"how cold is the sea today? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Brisk.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Brisk.")
    return next(filter(reply.answering(question), upstream.chat_requests()))


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


def test_an_admins_custom_parameter_is_sent_under_its_name(page_for, make_user, upstream):
    page = page_for(make_user(role="admin"))
    panel = _advanced_params(page)
    panel.get_by_role("button", name="Add Custom Parameter").click()
    panel.get_by_placeholder("Custom Parameter Name").fill("top_k")
    panel.get_by_placeholder("Custom Parameter Value").fill("7")

    request = _ask(page, upstream)
    assert request.get("top_k") == 7


def test_a_new_chat_starts_from_the_defaults(page_for, make_user, upstream):
    page = page_for(make_user())
    panel = _advanced_params(page)
    _set(panel, "Temperature", "0.3")
    assert _ask(page, upstream).get("temperature") == 0.3

    page.get_by_role("link", name="New Chat").first.click()
    expect(page).not_to_have_url(re.compile(r"/c/"))
    fresh = _ask(page, upstream)
    assert "temperature" not in fresh
