"""Regression: a comment's Copy link puts the link on the clipboard over plain HTTP by address.

Issue open-webui/open-webui#33927. Opened at an address such as http://192.168.1.20:8080, the
page is not a secure context and the browser offers no `navigator.clipboard`. The comment menu's
Copy link called it directly, so over plain HTTP by address the click threw "Cannot read
properties of undefined (reading 'writeText')" and nothing was copied, while the app's other copy
buttons fall back to the older copy command. The browser here reaches the instance by a host name
that is neither `localhost` nor a loopback address, as a phone or another computer on the network
does. The second test shows the same entry still copies on a secure page.

Discriminates: the plain-HTTP test fails on 0.13.1 and dev b6534c58f (the clipboard keeps what it
held before the click) and passes with the fix for #33927; the secure-page test passes on both.
"""

from __future__ import annotations

from urllib.parse import urlsplit

import pytest
from playwright.sync_api import Page, expect

from harness.actors import Actor
from harness.comments import comment, shared_item, switch_comments
from utils.clipboard import NOTHING_COPIED, copied, loopback_clipboard

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

CLIPBOARD = ["clipboard-read", "clipboard-write"]
COMMENT = "Shorten the second paragraph."


@pytest.fixture
def commented_prompt(admin, make_user, preserve) -> tuple[Actor, str, str]:
    """A fresh account's prompt with one comment of its own: the account, prompt id, comment id."""
    preserve("admin_config")
    switch_comments(admin, True)
    owner = make_user()
    prompt_id = shared_item("prompt", owner, admin, [])
    posted = comment(owner, "prompt", prompt_id, COMMENT)
    assert posted.status_code == 200, f"posting the comment failed: {posted.text}"
    return owner, prompt_id, posted.json()["id"]


def _copy_comment_link(page: Page, prompt_id: str) -> None:
    page.goto(f"/workspace/prompts/{prompt_id}")
    page.get_by_role("button", name="Comments (1)").click()
    posted = page.get_by_text(COMMENT)
    expect(posted).to_be_visible()
    posted.hover()
    page.get_by_role("button", name="More", exact=True).click()
    page.get_by_role("menuitem", name="Copy link").click()


def _link(page: Page, owner: Actor, prompt_id: str, comment_id: str) -> str:
    host = urlsplit(page.url).hostname
    port = urlsplit(owner.base_url).port
    return f"http://{host}:{port}/workspace/prompts/{prompt_id}?comment={comment_id}"


def test_copy_link_copies_the_comment_link_over_plain_http(lan_page_for, commented_prompt):
    owner, prompt_id, comment_id = commented_prompt
    page = lan_page_for(owner)
    reader = loopback_clipboard(page, owner)

    _copy_comment_link(page, prompt_id)

    clipboard = copied(reader)
    assert clipboard == _link(page, owner, prompt_id, comment_id), (
        f"Copy link over plain HTTP left {clipboard!r} on the clipboard, not the comment's link"
    )


@pytest.mark.secure_context
def test_copy_link_copies_the_comment_link_on_a_secure_page(page_for, commented_prompt):
    owner, prompt_id, comment_id = commented_prompt
    page = page_for(owner, permissions=CLIPBOARD)
    page.evaluate("text => navigator.clipboard.writeText(text)", NOTHING_COPIED)

    _copy_comment_link(page, prompt_id)

    clipboard = copied(page)
    assert clipboard == _link(page, owner, prompt_id, comment_id), (
        f"Copy link on a secure page left {clipboard!r} on the clipboard, not the comment's link"
    )
