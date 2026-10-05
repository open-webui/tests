"""Journey: signing in with an authenticator app in the browser while the admin requires one.

A first sign-in shows "Set up your authenticator" with the QR code and, folded away, the key to type
by hand; the code from that key leads to the ten recovery codes, which can only be left once "I have
saved my recovery codes" is ticked, and then to the chat. A later sign-in shows "Verify your sign-
in": a wrong code is named as such, a current one signs in, and "Use a recovery code" takes one of
the saved codes instead. After an operator's reset the sign-in asks for the operator's recovery
token and then sets up a new authenticator. An SSO sign-in lands on the same second step. Under
Settings > Account the authenticator shows as configured with the recovery codes left, and
generating new ones shows them once and signs the browser out.

Discriminates: in a backend copy, making `is_mfa_required` always answer False turns every test
here red (the password alone opens the chat, SSO included).

The two tests of new codes from Settings > Account fail on dev b859124f9
(open-webui/open-webui#31954): the change signs the account out, the browser's socket reconnects
with the ended session and the app sends it to the sign-in page within a second, so the codes, shown
nowhere else, are gone before anyone can save them (after generating, the old codes no longer work
either).
"""

from __future__ import annotations

import json
import re
from typing import Callable, Iterator

import pytest
from playwright.sync_api import Browser, Page, expect

from harness.mfa import (
    Authenticator,
    MfaAccount,
    add_account,
    enrolled_account,
    operator_reset,
    require_mfa,
)
from harness.oidc_provider import shared_provider, sso_env
from utils.chat_ui import chat_input

pytestmark = [
    pytest.mark.journey,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

PAGE_TIMEOUT_MS = 30_000


@pytest.fixture(scope="module")
def idp():
    return shared_provider()


@pytest.fixture(scope="module")
def secured(instance_with, idp):
    instance = instance_with({**sso_env(idp), "DEFAULT_USER_ROLE": "user"})
    if not instance.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    require_mfa(instance, ENABLE_MFA=True)
    return instance


@pytest.fixture
def open_page(browser: Browser, secured) -> Iterator[Callable[[str | None], Page]]:
    """`open_page(token)` opens the app in a browser of its own, signed in when given a token."""
    contexts = []

    def opened(token: str | None = None) -> Page:
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080}, base_url=secured.base_url
        )
        context.set_default_timeout(PAGE_TIMEOUT_MS)
        contexts.append(context)
        page = context.new_page()
        if token:
            page.add_init_script(
                f"try {{ localStorage.setItem('token', {json.dumps(token)}); }} catch (e) {{}}"
            )
            page.goto("/")
        else:
            page.goto("/auth")
        return page

    yield opened
    for context in contexts:
        context.close()


def quiet_changelog(secured, token: str) -> None:
    with secured.client(token) as client:
        client.post("/api/v1/users/user/settings/update", json={"ui": {"showChangelog": False}})


def enter_password(page: Page, email: str, password: str) -> None:
    page.get_by_label("Email").fill(email)
    page.get_by_label("Password", exact=True).fill(password)
    page.get_by_role("button", name="Sign in", exact=True).click()


def enter_code(page: Page, label: str, code: str) -> None:
    page.get_by_label(label).fill(code)
    page.get_by_role("button", name="Continue").click()


def at_second_step(page: Page, account: MfaAccount) -> None:
    enter_password(page, account.email, account.password)
    expect(page.get_by_role("heading", name="Verify your sign-in")).to_be_visible()


def test_a_first_sign_in_sets_up_the_authenticator_and_shows_the_codes_once(secured, open_page):
    email, password, _ = add_account(secured)
    page = open_page()
    enter_password(page, email, password)

    expect(page.get_by_role("heading", name="Set up your authenticator")).to_be_visible()
    expect(page.get_by_role("img", name="Authenticator setup QR code")).to_be_visible()
    page.get_by_text("Enter the key manually").click()
    manual_key = page.locator("details code").inner_text().strip()
    assert re.fullmatch(r"[A-Z2-7]{16,}", manual_key), manual_key
    enter_code(page, "Authenticator code", Authenticator(manual_key).code())

    expect(page.get_by_role("heading", name="Save your recovery codes")).to_be_visible()
    expect(page.get_by_role("listitem")).to_have_count(10)
    finish = page.get_by_role("button", name="Continue")
    expect(finish).to_be_disabled()
    page.get_by_label("I have saved my recovery codes").check()
    finish.click()
    expect(chat_input(page)).to_be_visible()

    page.reload()
    expect(chat_input(page)).to_be_visible()
    expect(page.get_by_role("heading", name="Save your recovery codes")).to_have_count(0)


def test_a_later_sign_in_refuses_a_wrong_code_and_takes_a_current_one(secured, open_page):
    account = enrolled_account(secured)
    quiet_changelog(secured, account.token)
    page = open_page()
    at_second_step(page, account)

    enter_code(page, "Authenticator code", account.authenticator.wrong_code())
    expect(page.get_by_role("alert")).to_have_text("Invalid or already used code.")
    enter_code(page, "Authenticator code", account.authenticator.code())

    expect(chat_input(page)).to_be_visible()


def test_a_recovery_code_signs_in_from_the_second_step(secured, open_page):
    account = enrolled_account(secured)
    quiet_changelog(secured, account.token)
    page = open_page()
    at_second_step(page, account)

    page.get_by_role("button", name="Use a recovery code").click()
    enter_code(page, "Recovery code", account.recovery_codes[0])

    expect(chat_input(page)).to_be_visible()


def test_an_operator_reset_leads_through_the_recovery_token_to_a_new_setup(secured, open_page):
    account = enrolled_account(secured)
    reset = operator_reset(secured, account.email, "lost phone, identity checked")
    assert reset.returncode == 0, reset.stderr[-3000:]
    page = open_page()
    enter_password(page, account.email, account.password)

    expect(page.get_by_role("heading", name="Recover your authenticator")).to_be_visible()
    enter_code(page, "Operator recovery token", reset.stdout.strip().splitlines()[-1])

    expect(page.get_by_role("heading", name="Set up your authenticator")).to_be_visible()
    expect(page.get_by_role("img", name="Authenticator setup QR code")).to_be_visible()


def test_an_sso_sign_in_lands_on_the_second_step(secured, idp, open_page):
    idp.sign_in_as()
    page = open_page()
    page.get_by_role("button", name="Continue with SSO").click()

    expect(page.get_by_role("heading", name="Set up your authenticator")).to_be_visible()
    expect(chat_input(page)).to_be_hidden()


def open_account_settings(page: Page) -> None:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("navigation", name="Chat history").get_by_label("User menu").click()
    page.get_by_role("button", name="Settings").click()
    page.get_by_role("tab", name="Account").click()


def manage_with_code(page: Page, account: MfaAccount, action: str) -> None:
    page.get_by_role("button", name="Manage").click()
    page.get_by_label("Authenticator code").fill(account.authenticator.code())
    page.get_by_role("button", name=action).click()


def expect_codes_kept_on_screen(page: Page) -> None:
    """The codes are shown once, so they must wait for "I have saved my recovery codes"."""
    expect(page.get_by_role("heading", name="Save your recovery codes")).to_be_visible()
    # bounded: the socket the change disconnected reconnects within this time
    page.wait_for_timeout(3_000)
    assert "/auth" not in page.url, (
        "the new recovery codes were taken off the screen before they could be saved: the "
        "browser was sent to the sign-in page as soon as the change signed the account out (#31954)"
    )
    expect(page.get_by_role("listitem")).to_have_count(10)
    page.get_by_label("I have saved my recovery codes").check()
    page.get_by_role("button", name="Continue").click()
    expect(page).to_have_url(re.compile(r"/auth"))


def test_account_settings_show_the_authenticator_and_the_codes_left(secured, open_page):
    account = enrolled_account(secured)
    quiet_changelog(secured, account.token)
    page = open_page(account.token)
    open_account_settings(page)

    expect(page.get_by_text("Authenticator configured")).to_be_visible()
    expect(page.get_by_text("10 recovery codes remaining")).to_be_visible()
    page.get_by_role("button", name="Manage").click()
    expect(page.get_by_role("button", name="Replace authenticator")).to_be_disabled()
    expect(page.get_by_role("button", name="Generate recovery codes")).to_be_disabled()


def test_new_recovery_codes_stay_on_screen_until_saved(secured, open_page):
    account = enrolled_account(secured)
    quiet_changelog(secured, account.token)
    page = open_page(account.token)
    open_account_settings(page)

    manage_with_code(page, account, "Generate recovery codes")

    expect_codes_kept_on_screen(page)


def test_a_replaced_authenticator_shows_its_codes_until_saved(secured, open_page):
    account = enrolled_account(secured)
    quiet_changelog(secured, account.token)
    page = open_page(account.token)
    open_account_settings(page)

    manage_with_code(page, account, "Replace authenticator")
    expect(page.get_by_role("heading", name="Set up your authenticator")).to_be_visible()
    page.get_by_text("Enter the key manually").click()
    new_key = page.locator("details code").inner_text().strip()
    assert new_key != account.authenticator.secret
    enter_code(page, "Authenticator code", Authenticator(new_key).code())

    expect_codes_kept_on_screen(page)
