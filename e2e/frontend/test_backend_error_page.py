"""Journey: a frontend that cannot reach its backend shows the error page and recovers.

On load the app asks the backend for its configuration. When that request fails it sends the user
to `/error`, which says the backend is required. Once the backend answers again, Check Again
reloads the app from the start and the signed-in user lands in the chat. A user who opens `/error`
while the backend is fine is sent straight to the chat instead (the last step of the recovery test).

Discriminates: passes on dev 176d31d1d; in a frontend copy, with the redirect on a failed
configuration request removed the three tests that need the error page fail, with Check Again doing
nothing the recovery test fails, and with the page's own redirect removed the recovery test
fails.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CONFIG_URL = "**/api/config"
ERROR_HEADING = re.compile(r"Backend Required$")


def test_unreachable_backend_lands_on_the_error_page(authenticated_page):
    page = authenticated_page
    page.route(CONFIG_URL, lambda route: route.abort())
    page.goto("/")

    expect(page).to_have_url(re.compile(r"/error$"))
    expect(page.get_by_text(ERROR_HEADING)).to_be_visible()
    expect(page.get_by_text("Please serve the WebUI from the backend.")).to_be_visible()
    expect(page.get_by_role("button", name="Check Again")).to_be_visible()


def test_a_failing_config_answer_also_lands_on_the_error_page(authenticated_page):
    page = authenticated_page
    page.route(
        CONFIG_URL,
        lambda route: route.fulfill(status=502, json={"detail": "bad gateway"}),
    )
    page.goto("/notes")

    expect(page).to_have_url(re.compile(r"/error$"))
    expect(page.get_by_text(ERROR_HEADING)).to_be_visible()


def test_check_again_recovers_once_the_backend_answers(authenticated_page):
    page = authenticated_page
    page.route(CONFIG_URL, lambda route: route.abort())
    page.goto("/")
    expect(page.get_by_text(ERROR_HEADING)).to_be_visible()

    page.unroute(CONFIG_URL)
    page.get_by_role("button", name="Check Again").click()

    expect(page).not_to_have_url(re.compile(r"/error$"))
    expect(chat_input(page)).to_be_visible()

    page.goto("/error")

    expect(page).not_to_have_url(re.compile(r"/error$"))
    expect(chat_input(page)).to_be_visible()
