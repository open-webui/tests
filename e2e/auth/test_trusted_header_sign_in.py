"""Journey: a browser behind an authenticating reverse proxy is signed in by the headers it sends.

With `WEBUI_AUTH_TRUSTED_EMAIL_HEADER` set, the auth page signs in by itself and the proxy's
headers say who the person is. The browser context here adds those headers to every request it
makes, which is what such a proxy does. A first visit creates the account under the name header
and, with the default role, shows the activation screen; a role header of `user` lands in the
chat under that name. A browser that reaches the instance without the email header is told the
provider sent no trusted header and gets no chat. When the proxy starts naming someone else, the
next page load is that other person, never the first one.

Discriminates: on dev ebc6add67, a frontend copy whose auth page waits for the form instead of
signing in by itself turns all four tests red, and a backend copy that ignores the name header
turns the three tests that sign in red (the account is named after the email).
"""

from __future__ import annotations

import re
import secrets
import urllib.parse
from typing import Callable, Generator

import pytest
from playwright.sync_api import Browser, BrowserContext, Page, expect

from harness.instance import LaunchedInstance
from utils.chat_ui import chat_input

pytestmark = [
    pytest.mark.journey,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

EMAIL_HEADER = "X-Forwarded-Email"
NAME_HEADER = "X-Forwarded-Name"
ROLE_HEADER = "X-Forwarded-Role"
TRUSTED_HEADERS = {
    "WEBUI_AUTH_TRUSTED_EMAIL_HEADER": EMAIL_HEADER,
    "WEBUI_AUTH_TRUSTED_NAME_HEADER": NAME_HEADER,
    "WEBUI_AUTH_TRUSTED_ROLE_HEADER": ROLE_HEADER,
}
PAGE_TIMEOUT_MS = 30_000


@pytest.fixture
def proxied(instance_with) -> LaunchedInstance:
    launched = instance_with(TRUSTED_HEADERS)
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    return launched


@pytest.fixture
def behind_proxy(
    browser: Browser, proxied: LaunchedInstance
) -> Generator[Callable[[dict[str, str]], Page], None, None]:
    """`behind_proxy(headers)` opens a page whose every request carries those headers."""
    contexts: list[BrowserContext] = []

    def open_page(headers: dict[str, str]) -> Page:
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            base_url=proxied.base_url,
            extra_http_headers=headers,
        )
        context.set_default_timeout(PAGE_TIMEOUT_MS)
        contexts.append(context)
        return context.new_page()

    yield open_page
    for context in contexts:
        context.close()


def _person(name: str) -> dict[str, str]:
    email = f"proxied-{secrets.token_hex(4)}@example.com"
    return {EMAIL_HEADER: email, NAME_HEADER: urllib.parse.quote(name), ROLE_HEADER: "user"}


def _account(instance: LaunchedInstance, email: str) -> dict:
    with instance.client() as client:
        found = client.get("/api/v1/users/", params={"query": email}).json()["users"]
    assert len(found) == 1, f"expected one account for {email}, found {found}"
    return found[0]


def _signed_in_name(page: Page) -> str:
    return page.evaluate(
        "() => fetch('/api/v1/auths/', {headers: {Authorization: `Bearer ${localStorage.token}`}})"
        ".then((answer) => answer.json()).then((session) => session.name)"
    )


def test_a_first_visit_creates_the_named_account_and_shows_the_activation_screen(
    proxied, behind_proxy
):
    headers = _person("Jörg Østergaard")
    del headers[ROLE_HEADER]
    page = behind_proxy(headers)

    page.goto("/")

    expect(page.get_by_text("Account Activation Pending")).to_be_visible(timeout=PAGE_TIMEOUT_MS)
    expect(chat_input(page)).to_be_hidden()
    account = _account(proxied, headers[EMAIL_HEADER])
    assert (account["name"], account["role"]) == ("Jörg Østergaard", "pending")


def test_a_user_role_header_lands_in_the_chat_under_the_forwarded_name(behind_proxy):
    page = behind_proxy(_person("Harbour Pilot"))

    page.goto("/")

    expect(chat_input(page)).to_be_visible(timeout=PAGE_TIMEOUT_MS)
    page.get_by_role("navigation", name="Chat history").get_by_label("User menu").click()
    expect(page.get_by_role("button", name="Harbour Pilot")).to_be_visible()


def test_a_page_reached_without_the_email_header_says_so_and_offers_no_chat(behind_proxy):
    page = behind_proxy({})

    page.goto("/")

    expect(page.get_by_text("Your provider has not provided a trusted header.")).to_be_visible(
        timeout=PAGE_TIMEOUT_MS
    )
    expect(page).to_have_url(re.compile(r"/auth"))
    expect(chat_input(page)).to_be_hidden()


def test_when_the_proxy_names_another_person_the_next_page_load_is_that_person(behind_proxy):
    first, second = _person("First Watch"), _person("Second Watch")
    page = behind_proxy(first)
    page.goto("/")
    expect(chat_input(page)).to_be_visible(timeout=PAGE_TIMEOUT_MS)

    page.context.set_extra_http_headers(second)
    page.reload()

    expect(chat_input(page)).to_be_visible(timeout=PAGE_TIMEOUT_MS)
    page.get_by_role("navigation", name="Chat history").get_by_label("User menu").click()
    expect(page.get_by_role("button", name="Second Watch")).to_be_visible()
    assert _signed_in_name(page) == "Second Watch"
