"""Journey: an admin's banners reach users on the new-chat screen and users can close them.

The admin adds a banner in the General settings (type, content, remembered dismissal), saves and
removes it again; a user sees its type label and rendered content on a new chat and not in an
open one. Every banner has a close button (the docs still say otherwise). A dismissible one
stays closed after a reload (the browser remembers its id), one saved without "Remember
Dismissal" comes back on the next load.

Discriminates: passes on the 176d31d1d frontend; in frontend copies, with the admin's Save not
sending the banner list the add and remove tests fail, with dismissals never stored the reload test
fails (the banner is back), with every close stored the session-only test fails (the banner stays
gone), with any stored dismissal hiding all banners the replaced-banner test fails, with the type
list or the dismissal switch unbound the add and save tests fail, with the label fixed to Info the
other three type tests fail, with the content shown as plain text the markdown test fails and with
the new-chat condition dropped the open-chat test fails (the banner shows above a running chat).
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from utils.admin_settings import save
from utils.chat_ui import expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

BANNERS = "/api/v1/configs/banners"
SAVED = "Settings saved successfully!"


def _banner(content: str, type: str = "info", dismissible: bool = True) -> dict:
    return {
        "id": uuid.uuid4().hex,
        "type": type,
        "title": "",
        "content": content,
        "dismissible": dismissible,
        "timestamp": 1767225600,
    }


def _saved_banners(admin) -> list[dict]:
    with admin.client() as client:
        listed = client.get(BANNERS)
    listed.raise_for_status()
    return listed.json()


def _publish(admin, banners: list[dict]) -> None:
    with admin.client() as client:
        client.post(BANNERS, json={"banners": banners}).raise_for_status()


@pytest.fixture
def no_banners(admin):
    """Starts each test with no banners and puts the instance's own back afterwards."""
    before = _saved_banners(admin)
    _publish(admin, [])
    yield
    _publish(admin, before)


@pytest.fixture
def admin_page(make_user, page_for) -> Page:
    """A fresh admin's page, so the shared admin's browser is left alone."""
    return page_for(make_user(role="admin"))


def _open_general_settings(page: Page):
    page.goto("/admin/settings/general")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("button", name="Add banner")).to_be_visible()
    return settings


def _close_button(page: Page):
    return page.get_by_role("button", name="Close Banner")


def test_an_admin_adds_a_banner_and_a_user_sees_its_type_and_text(
    admin_page, page_for, make_user, admin, no_banners
):
    text = f"Maintenance tonight {uuid.uuid4().hex[:6]}"
    settings = _open_general_settings(admin_page)
    settings.get_by_role("button", name="Add banner").click()
    settings.get_by_label("Type").select_option("Warning")
    settings.get_by_label("Content").fill(text)
    save(settings)

    saved = _saved_banners(admin)
    assert [(b["type"], b["content"], b["dismissible"]) for b in saved] == [("warning", text, True)]
    user_page = page_for(make_user())
    expect(user_page.get_by_text(text)).to_be_visible()
    expect(user_page.get_by_text("Warning", exact=True)).to_be_visible()


@pytest.mark.parametrize("kind", ["info", "success", "warning", "error"])
def test_each_banner_type_shows_its_own_label(kind, admin, page_for, make_user, no_banners):
    _publish(admin, [_banner(f"a {kind} notice", type=kind)])
    page = page_for(make_user())

    expect(page.get_by_text(f"a {kind} notice")).to_be_visible()
    expect(page.get_by_text(kind.capitalize(), exact=True)).to_be_visible()


def test_banner_content_is_rendered_as_markdown_and_sanitized(
    admin, page_for, make_user, no_banners
):
    content = (
        "**Read this** first, see [the status page](https://status.example.com)"
        '<img src="/nothing.png" onerror="window.pwned = 1">'
    )
    _publish(admin, [_banner(content)])
    page = page_for(make_user())

    expect(page.locator("strong", has_text="Read this")).to_be_visible()
    link = page.get_by_role("link", name="the status page")
    expect(link).to_have_attribute("href", "https://status.example.com")
    expect(page.locator("[onerror]")).to_have_count(0)


def test_an_admin_can_save_a_banner_without_remembered_dismissal(admin_page, admin, no_banners):
    settings = _open_general_settings(admin_page)
    settings.get_by_role("button", name="Add banner").click()
    settings.get_by_label("Type").select_option("Error")
    settings.get_by_label("Content").fill("Outage in progress")
    settings.get_by_role("switch", name="Remember Dismissal").click()
    save(settings)

    assert [b["dismissible"] for b in _saved_banners(admin)] == [False]


def test_a_dismissed_banner_stays_dismissed_after_a_reload(admin, page_for, make_user, no_banners):
    _publish(
        admin, [_banner("Please read the new policy"), _banner("Always here", dismissible=False)]
    )
    page = page_for(make_user())
    expect(page.get_by_text("Please read the new policy")).to_be_visible()

    _close_button(page).first.click()
    expect(page.get_by_text("Please read the new policy")).to_be_hidden()
    page.reload()

    expect(page.get_by_text("Always here")).to_be_visible()
    expect(page.get_by_text("Please read the new policy")).to_be_hidden()
    expect(_close_button(page)).to_have_count(1)


def test_a_banner_without_remembered_dismissal_closes_only_until_the_next_load(
    admin, page_for, make_user, no_banners
):
    _publish(admin, [_banner("Planned downtime", dismissible=False)])
    page = page_for(make_user())
    expect(page.get_by_text("Planned downtime")).to_be_visible()

    _close_button(page).click()
    expect(page.get_by_text("Planned downtime")).to_be_hidden()
    page.reload()

    expect(page.get_by_text("Planned downtime")).to_be_visible()


def test_a_replaced_banner_is_new_to_a_user_who_closed_the_old_one(
    admin, page_for, make_user, no_banners
):
    _publish(admin, [_banner("Version one of the notice")])
    page = page_for(make_user())
    _close_button(page).click()
    expect(page.get_by_text("Version one of the notice")).to_be_hidden()

    _publish(admin, [_banner("Version two of the notice")])
    page.reload()

    expect(page.get_by_text("Version two of the notice")).to_be_visible()


def test_banners_are_not_shown_above_an_open_chat(admin, page_for, make_user, upstream, no_banners):
    _publish(admin, [_banner("Only for new chats")])
    page = page_for(make_user())
    expect(page.get_by_text("Only for new chats")).to_be_visible()

    upstream.queue(reply.text("hello there", match=reply.answering("start a chat")))
    send(page, "start a chat")
    expect_reply(page, "hello there")

    expect(page.get_by_text("Only for new chats")).to_be_hidden()


def test_an_admin_removes_a_banner_and_users_stop_seeing_it(
    admin_page, admin, page_for, make_user, no_banners
):
    _publish(admin, [_banner("About to be removed"), _banner("Stays for everyone")])
    user_page = page_for(make_user())
    expect(user_page.get_by_text("About to be removed")).to_be_visible()

    settings = _open_general_settings(admin_page)
    settings.get_by_role("button", name="Delete").first.click()
    save(settings)

    assert [b["content"] for b in _saved_banners(admin)] == ["Stays for everyone"]
    user_page.reload()
    expect(user_page.get_by_text("Stays for everyone")).to_be_visible()
    expect(user_page.get_by_text("About to be removed")).to_be_hidden()
