"""Journey: signing in with an authenticator app in the browser while the admin requires one.

A first sign-in shows "Set up your authenticator" with the QR code and, folded away, the key to type
by hand; the code from that key leads to the ten recovery codes, which can only be left once "I have
saved my recovery codes" is ticked, and then to the chat. A later sign-in shows "Verify your sign-
in": a wrong code is named as such, a current one signs in, and "Use a recovery code" takes one of
the saved codes instead. After an operator's reset the sign-in asks for the operator's recovery
token and then sets up a new authenticator. An SSO or LDAP sign-in sets up an authenticator the
same way and asks for its code at the next sign-in. Under Settings > Account the authenticator shows
as configured with the recovery codes left, and generating new ones shows them once and signs the
browser out.

On the second step, switching between an authenticator code and a recovery code empties the field
and clears the last error; a used recovery code is refused at the next sign-in. Five wrong codes end
the sign-in ("Too many codes. Please sign in again."), so even a current code is refused until "Back
to sign in" and the password again. The keyboard alone signs in, with the code pasted and Enter to
submit, and a second tab of the same browser has no session until the code is in. The setup QR code
holds the key shown for manual entry, and setup and the second step fit a phone screen without
scrolling sideways. The admin's "Sign out all devices" and a new password set in Edit User send the
account back through its code, never into a new setup.

Discriminates: in a backend copy, making `is_mfa_required` always answer False turns every test
here red (the password alone opens the chat, SSO included). In a backend copy with the five-code
limit raised, recovery codes kept after use, the QR code made from another key, `is_mfa_required`
answering False for LDAP and SSO, the admin's sign-out revoking nothing and a password change
wiping the authenticator, the matching tests go red. In a frontend build where switching keeps the
field and its error, Enter does nothing, the QR code is 480 pixels wide and "Back to sign in" does
nothing, the switching, keyboard, phone and too-many-codes tests go red.

The two tests of new codes from Settings > Account fail on dev b859124f9
(open-webui/open-webui#31954): the change signs the account out, the browser's socket reconnects
with the ended session and the app sends it to the sign-in page within a second, so the codes, shown
nowhere else, are gone before anyone can save them (after generating, the old codes no longer work
either).
"""

from __future__ import annotations

import json
import re
import urllib.parse
import uuid
from typing import Callable, Iterator

import pytest
from playwright.sync_api import Browser, Locator, Page, expect

from harness.ldap_server import LDAP_CONFIG, save_ldap_settings, serve_directory
from harness.mfa import (
    Authenticator,
    MfaAccount,
    add_account,
    enrolled_account,
    new_address,
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
def open_page(browser: Browser, secured) -> Iterator[Callable[..., Page]]:
    """`open_page(token)` opens the app in a browser of its own, signed in when given a token.

    Each browser signs in from an address of its own; further keywords go to the browser context.
    """
    contexts = []

    def opened(token: str | None = None, **context_options) -> Page:
        options = {
            "viewport": {"width": 1920, "height": 1080},
            "extra_http_headers": {"X-Forwarded-For": new_address()},
            **context_options,
        }
        context = browser.new_context(base_url=secured.base_url, **options)
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


def set_up_from_the_key(page: Page) -> Authenticator:
    """Type the setup key into an authenticator app, confirm it and save the codes."""
    expect(page.get_by_role("heading", name="Set up your authenticator")).to_be_visible()
    page.get_by_text("Enter the key manually").click()
    authenticator = Authenticator(page.locator("details code").inner_text().strip())
    enter_code(page, "Authenticator code", authenticator.code())
    page.get_by_label("I have saved my recovery codes").check()
    page.get_by_role("button", name="Continue").click()
    expect(chat_input(page)).to_be_visible()
    return authenticator


def sign_out(page: Page, secured) -> None:
    quiet_changelog(secured, page.evaluate("localStorage.token"))
    page.reload()
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("navigation", name="Chat history").get_by_label("User menu").click()
    page.get_by_role("button", name="Sign Out").click()


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


def test_an_sso_sign_in_sets_up_an_authenticator_and_asks_for_it_next_time(secured, idp, open_page):
    person = idp.sign_in_as()
    page = open_page()
    page.get_by_role("button", name="Continue with SSO").click()

    expect(page.get_by_role("heading", name="Set up your authenticator")).to_be_visible()
    expect(chat_input(page)).to_be_hidden()
    authenticator = set_up_from_the_key(page)
    sign_out(page, secured)
    # the stand-in provider has no sign-out page to send the browser back from
    expect(page).to_have_url(re.compile(r"/logout\?.*post_logout_redirect_uri"))
    page.goto("/auth")

    idp.sign_in_as(sub=person["sub"], email=person["email"], name=person["name"])
    page.get_by_role("button", name="Continue with SSO").click()
    expect(page.get_by_role("heading", name="Verify your sign-in")).to_be_visible()
    expect(chat_input(page)).to_be_hidden()
    enter_code(page, "Authenticator code", authenticator.code())
    expect(chat_input(page)).to_be_visible()


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


# --------------------------------------------------------------------------- the second step


def test_switching_between_authenticator_and_recovery_code_starts_the_field_afresh(
    secured, open_page
):
    account = enrolled_account(secured)
    quiet_changelog(secured, account.token)
    page = open_page()
    at_second_step(page, account)
    enter_code(page, "Authenticator code", account.authenticator.wrong_code())
    expect(page.get_by_role("alert")).to_have_text("Invalid or already used code.")

    page.get_by_role("button", name="Use a recovery code").click()
    expect(page.get_by_text("Enter one of your saved recovery codes.")).to_be_visible()
    expect(page.get_by_label("Recovery code")).to_have_value("")
    expect(page.get_by_role("alert")).to_have_count(0)
    page.get_by_label("Recovery code").fill(account.recovery_codes[0][:8])

    page.get_by_role("button", name="Use an authenticator code").click()
    expect(
        page.get_by_text("Enter the six-digit code from your authenticator app.")
    ).to_be_visible()
    expect(page.get_by_label("Authenticator code")).to_have_value("")
    enter_code(page, "Authenticator code", account.authenticator.code())
    expect(chat_input(page)).to_be_visible()


def test_a_used_recovery_code_is_refused_at_the_next_sign_in(secured, open_page):
    account = enrolled_account(secured)
    quiet_changelog(secured, account.token)
    page = open_page()
    at_second_step(page, account)
    page.get_by_role("button", name="Use a recovery code").click()
    enter_code(page, "Recovery code", account.recovery_codes[0])
    sign_out(page, secured)
    expect(page).to_have_url(re.compile(r"/auth"))

    at_second_step(page, account)
    page.get_by_role("button", name="Use a recovery code").click()
    enter_code(page, "Recovery code", account.recovery_codes[0])
    expect(page.get_by_role("alert")).to_have_text("Invalid or already used code.")
    expect(chat_input(page)).to_be_hidden()
    enter_code(page, "Recovery code", account.recovery_codes[1].upper())

    open_account_settings(page)
    expect(page.get_by_text("8 recovery codes remaining")).to_be_visible()


def test_too_many_wrong_codes_end_the_sign_in_until_the_password_is_entered_again(
    secured, open_page
):
    account = enrolled_account(secured)
    quiet_changelog(secured, account.token)
    page = open_page()
    at_second_step(page, account)
    for _ in range(5):
        enter_code(page, "Authenticator code", account.authenticator.wrong_code())
        expect(page.get_by_role("alert")).to_have_text("Invalid or already used code.")
        expect(page.get_by_role("button", name="Continue")).to_be_enabled()

    enter_code(page, "Authenticator code", account.authenticator.code())
    expect(page.get_by_role("alert")).to_have_text("Too many codes. Please sign in again.")
    expect(chat_input(page)).to_be_hidden()

    page.get_by_role("button", name="Back to sign in").click()
    at_second_step(page, account)
    enter_code(page, "Authenticator code", account.authenticator.code())
    expect(chat_input(page)).to_be_visible()


def test_the_keyboard_alone_signs_in_with_a_pasted_code(secured, open_page):
    account = enrolled_account(secured)
    quiet_changelog(secured, account.token)
    page = open_page(permissions=["clipboard-read", "clipboard-write"])
    page.get_by_label("Email").focus()
    page.keyboard.type(account.email)
    page.keyboard.press("Tab")
    page.keyboard.type(account.password)
    page.keyboard.press("Enter")
    expect(page.get_by_role("heading", name="Verify your sign-in")).to_be_visible()

    code_field = page.get_by_label("Authenticator code")
    for _ in range(10):
        if code_field.evaluate("field => field === document.activeElement"):
            break
        page.keyboard.press("Tab")
    expect(code_field).to_be_focused()
    page.evaluate("code => navigator.clipboard.writeText(code)", account.authenticator.code())
    page.keyboard.press("Control+V")
    page.keyboard.press("Enter")

    expect(chat_input(page)).to_be_visible()


def test_a_second_tab_has_no_session_until_the_code_is_entered(secured, open_page):
    account = enrolled_account(secured)
    quiet_changelog(secured, account.token)
    first_tab = open_page()
    at_second_step(first_tab, account)

    second_tab = first_tab.context.new_page()
    second_tab.goto("/")
    expect(second_tab.get_by_label("Email")).to_be_visible()
    expect(second_tab).to_have_url(re.compile(r"/auth"))

    enter_code(first_tab, "Authenticator code", account.authenticator.code())
    expect(chat_input(first_tab)).to_be_visible()
    second_tab.goto("/")
    expect(chat_input(second_tab)).to_be_visible()


# --------------------------------------------------------------------------- the setup screen


def test_the_qr_code_holds_the_same_key_as_the_manual_entry(secured, open_page):
    email, password, _ = add_account(secured)
    cv2 = pytest.importorskip("cv2", reason="reading the QR code takes opencv-python")
    numpy = pytest.importorskip("numpy")
    page = open_page(device_scale_factor=3)
    enter_password(page, email, password)

    qr_code = page.get_by_role("img", name="Authenticator setup QR code")
    expect(qr_code).to_be_visible()
    pixels = cv2.imdecode(numpy.frombuffer(qr_code.screenshot(), numpy.uint8), cv2.IMREAD_COLOR)
    uri, _, _ = cv2.QRCodeDetector().detectAndDecode(pixels)
    page.get_by_text("Enter the key manually").click()
    manual_key = page.locator("details code").inner_text().strip()

    scanned = urllib.parse.urlsplit(uri)
    query = urllib.parse.parse_qs(scanned.query)
    assert (scanned.scheme, scanned.netloc) == ("otpauth", "totp"), uri
    assert query["secret"] == [manual_key], (
        f"the QR code holds another key than the page shows: {uri}"
    )
    assert query["issuer"] == ["Open WebUI"], uri
    assert urllib.parse.unquote(scanned.path) == f"/Open WebUI:{email}", uri


PHONE = {"width": 390, "height": 844}


def fits_the_screen(page: Page, element) -> None:
    box = element.bounding_box()
    assert box and box["x"] >= 0 and box["x"] + box["width"] <= PHONE["width"], box
    scroll_width = page.evaluate("document.documentElement.scrollWidth")
    assert scroll_width <= PHONE["width"], f"the page scrolls sideways ({scroll_width}px wide)"


def test_setup_and_the_second_step_fit_a_phone(secured, open_page):
    email, password, _ = add_account(secured)
    phone = {"viewport": PHONE, "is_mobile": True, "has_touch": True}
    page = open_page(**phone)
    enter_password(page, email, password)

    qr_code = page.get_by_role("img", name="Authenticator setup QR code")
    expect(qr_code).to_be_visible()
    fits_the_screen(page, qr_code)
    page.get_by_text("Enter the key manually").tap()
    manual_key = page.locator("details code")
    fits_the_screen(page, manual_key)
    authenticator = Authenticator(manual_key.inner_text().strip())
    enter_code(page, "Authenticator code", authenticator.code())
    expect(page.get_by_role("heading", name="Save your recovery codes")).to_be_visible()
    fits_the_screen(page, page.get_by_role("list"))
    page.get_by_label("I have saved my recovery codes").check()
    page.get_by_role("button", name="Continue").click()
    expect(chat_input(page)).to_be_visible()

    later = open_page(**phone)
    enter_password(later, email, password)
    expect(later.get_by_role("heading", name="Verify your sign-in")).to_be_visible()
    fits_the_screen(later, later.get_by_role("button", name="Continue"))
    fits_the_screen(later, later.get_by_role("button", name="Use a recovery code"))
    enter_code(later, "Authenticator code", authenticator.code())
    expect(chat_input(later)).to_be_visible()


# --------------------------------------------------------------------------- other sign-ins


@pytest.fixture
def directory(secured, preserve):
    preserve(LDAP_CONFIG, on=secured)
    with serve_directory() as served, secured.client() as client:
        save_ldap_settings(client, served)
        yield served


def ldap_sign_in(page: Page, username: str, password: str) -> None:
    page.get_by_label("Username").fill(username)
    page.get_by_label("Password", exact=True).fill(password)
    page.get_by_role("button", name="Authenticate").click()


def test_an_ldap_sign_in_sets_up_an_authenticator_and_asks_for_it_next_time(
    secured, directory, open_page
):
    username = f"mfa-ldap-{uuid.uuid4().hex[:8]}"
    directory.add_person(username, "directory-pass-1", cn="Directory Person")
    page = open_page()
    ldap_sign_in(page, username, "directory-pass-1")

    authenticator = set_up_from_the_key(page)
    sign_out(page, secured)
    expect(page).to_have_url(re.compile(r"/auth"))

    ldap_sign_in(page, username, "directory-pass-1")
    expect(page.get_by_role("heading", name="Verify your sign-in")).to_be_visible()
    expect(chat_input(page)).to_be_hidden()
    enter_code(page, "Authenticator code", authenticator.code())
    expect(chat_input(page)).to_be_visible()


# --------------------------------------------------------------------------- the admin's controls


def edit_user(admin_page: Page, email: str) -> Locator:
    admin_page.goto("/admin/users")
    users = admin_page.get_by_role("main")
    users.get_by_role("textbox", name="Search").fill(email)
    users.get_by_role("row").filter(has_text=email).get_by_role("button", name="Edit User").click()
    return admin_page.get_by_role("dialog").filter(has_text="Edit User")


def test_the_admins_new_password_and_sign_out_still_leave_the_second_step(secured, open_page):
    account = enrolled_account(secured)
    quiet_changelog(secured, account.token)
    quiet_changelog(secured, secured.admin_token)
    accounts_page = open_page(account.token)
    expect(chat_input(accounts_page)).to_be_visible()
    admin_page = open_page(secured.admin_token)
    expect(chat_input(admin_page)).to_be_visible()

    editing = edit_user(admin_page, account.email)
    editing.get_by_role("button", name="Sign out all devices").click()
    confirming = admin_page.get_by_role("dialog").filter(has_text="Sign out all devices?")
    confirming.get_by_role("button", name="Confirm").click()
    expect(admin_page.get_by_text("All sessions revoked")).to_be_visible()
    accounts_page.reload()
    expect(accounts_page).to_have_url(re.compile(r"/auth"))

    editing.get_by_label("New Password").fill("set-by-the-admin-1")
    editing.get_by_role("button", name="Save").click()
    expect(editing).to_be_hidden()

    enter_password(accounts_page, account.email, "set-by-the-admin-1")
    expect(accounts_page.get_by_role("heading", name="Verify your sign-in")).to_be_visible()
    expect(chat_input(accounts_page)).to_be_hidden()
    enter_code(accounts_page, "Authenticator code", account.authenticator.code())
    expect(chat_input(accounts_page)).to_be_visible()
