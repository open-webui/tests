"""Regression: the bars at the top of the chat and the other pages sit cleanly over the page.

Three 0.13.1 fixes. The chat's tab bar painted a solid strip over the chat background (PR
open-webui/open-webui#33849, fix 86b49034b); it now lets the background show through. At a UI
Scale above 1 the tab bar grows with the text while the tab content kept its unscaled offset, so
the top of the conversation slid under the bar (PR #33830, fix 03361c968, issue #33829). The top
bars of Workspace, Admin Panel, Playground and Calendar blurred a page background image behind
them (PR #33799, fix 8ebc96d3d, issue #33798).

Discriminates: passes on the dev e8c0ad85e build and fails in a frontend copy with 86b49034b,
03361c968 and 8ebc96d3d reverted (the tab bar is white, the conversation starts under the bar,
and each page's top bar blurs).
"""

from __future__ import annotations

import base64

import pytest
from playwright.sync_api import Page, expect

from utils.chat_ui import chat_input, chat_tab_bar

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

BACKGROUND = (
    "data:image/svg+xml;base64,"
    + base64.b64encode(
        b'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8">'
        b'<rect width="8" height="8" fill="#2a6f97"/></svg>'
    ).decode()
)
PAGES = ["/workspace/models", "/admin/users", "/playground", "/calendar"]


def account_with(make_user, **ui_settings):
    account = make_user(role="admin")
    with account.client() as client:
        saved = client.post("/api/v1/users/user/settings/update", json={"ui": ui_settings})
    assert saved.status_code == 200, saved.text
    return account


def test_the_tab_bar_lets_the_chat_background_show_through(page_for, make_user):
    page = page_for(account_with(make_user, backgroundImageUrl=BACKGROUND))
    expect(chat_input(page)).to_be_visible()
    bar = chat_tab_bar(page)
    expect(bar).to_be_visible()

    painted = bar.evaluate("bar => getComputedStyle(bar).backgroundColor")

    assert painted == "rgba(0, 0, 0, 0)", f"the tab bar covers the chat background with {painted}"


def test_at_a_larger_ui_scale_the_conversation_starts_below_the_tab_bar(page_for, make_user):
    page = page_for(account_with(make_user, textScale=1.5))
    expect(chat_input(page)).to_be_visible()
    bar = chat_tab_bar(page)
    expect(bar).to_be_visible()
    conversation_tab = page.get_by_role("tabpanel").filter(has=chat_input(page))

    bar_bottom = bar.evaluate("bar => bar.getBoundingClientRect().bottom")
    content_top = conversation_tab.evaluate("panel => panel.getBoundingClientRect().top")

    assert content_top >= bar_bottom - 1, (
        f"at UI Scale 1.5 the conversation starts at {content_top}px, under the tab bar that "
        f"ends at {bar_bottom}px"
    )


@pytest.mark.parametrize("path", PAGES)
def test_the_page_top_bar_does_not_blur_the_background(path, page_for, make_user):
    page: Page = page_for(account_with(make_user, backgroundImageUrl=BACKGROUND), path=path)
    # pages open as tabs beside the chat since dev da850248d, whose own top bar is then hidden
    navigation = page.get_by_role("main").locator("nav").filter(visible=True).first
    expect(navigation).to_be_visible()

    blurs = page.locator("nav").evaluate_all(
        "navs => navs.map(nav => getComputedStyle(nav).backdropFilter)"
    )

    assert blurs and all(blur == "none" for blur in blurs), (
        f"a top bar on {path} blurs the page behind it: {blurs}"
    )
