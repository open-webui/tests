"""Journey: a comment, a reply and an edit written in the comment panel are saved.

The comment panel's box makes a fresh id for every comment it starts. Up to 0.13.1 it made that id
with `crypto.randomUUID`, which the browser offers only in a secure context, so over plain HTTP by
address (a LAN address, Docker without TLS) the box failed with "crypto.randomUUID is not a
function" and no comment, reply or edit could be written. Fixed in 0.13.2 by 3c400952c. Every
page of the suite opens over plain HTTP by such an address, so this walks the fixed path as a
person on the LAN does: type into the box, send with Enter, and read back what was stored.

Discriminates: passes on dev 485103cde; fails on a build of it with the comment box's id made by
`crypto.randomUUID` again (3c400952c undone for the box), where the panel offers no box to type in.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from harness.comments import replies, shared_item, switch_comments, thread

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

FIRST = "The second step needs an example."
ANSWER = "Added one under the list."
EDITED = "The second step needs a shorter example."


@pytest.fixture
def owned_prompt(admin, make_user, preserve) -> tuple[Actor, str]:
    """A fresh account and a prompt of its own, with comments switched on."""
    preserve("admin_config")
    switch_comments(admin, True)
    owner = make_user()
    return owner, shared_item("prompt", owner, admin, [])


def _panel(page: Page) -> Locator:
    return page.get_by_role("complementary", name="Comments")


def _write(page: Page, box: Locator, text: str, sent: str) -> None:
    """Types `text` into the comment box and sends it with Enter, waiting for the saved answer."""
    box.click()
    expect(box).to_be_focused()
    box.press_sequentially(text)
    with page.expect_response(
        lambda response: "/api/v1/comments" in response.url and response.request.method == sent
    ) as saving:
        box.press("Enter")
    assert saving.value.ok, f"saving the comment answered {saving.value.status}"


def _stored_thread(owner: Actor, prompt_id: str) -> list[dict]:
    listed = thread(owner, "prompt", prompt_id)
    assert listed.status_code == 200, listed.text
    return listed.json()["items"]


def test_a_comment_its_reply_and_an_edit_are_written_in_the_panel(page_for, owned_prompt):
    owner, prompt_id = owned_prompt
    page = page_for(owner, path=f"/workspace/prompts/{prompt_id}")
    page.get_by_role("button", name="Comments", exact=True).click()
    panel = _panel(page)
    expect(panel.get_by_text("No comments yet")).to_be_visible()

    _write(page, panel.locator('[contenteditable="true"]'), FIRST, "POST")
    posted = panel.get_by_text(FIRST)
    expect(posted).to_be_visible()
    stored = _stored_thread(owner, prompt_id)
    assert [item["content"] for item in stored] == [FIRST], (
        f"the comment typed in the panel was stored as {stored!r}"
    )
    comment_id = stored[0]["id"]

    panel.get_by_role("button", name="Reply", exact=True).click()
    reply_box = (
        panel.locator(f"#comment-{comment_id}").locator("..").locator('[contenteditable="true"]')
    )
    _write(page, reply_box.first, ANSWER, "POST")
    expect(panel.get_by_text(ANSWER)).to_be_visible()
    answered = replies(owner, comment_id)
    assert answered.status_code == 200, answered.text
    assert [item["content"] for item in answered.json()["items"]] == [ANSWER], (
        f"the reply typed in the panel was stored as {answered.json()!r}"
    )

    posted.hover()
    panel.locator(f"#comment-{comment_id}").get_by_role("button", name="More", exact=True).click()
    page.get_by_role("menuitem", name="Edit").click()
    edit_box = panel.locator(f"#comment-{comment_id}").locator('[contenteditable="true"]')
    expect(edit_box).to_contain_text(FIRST)
    edit_box.click()
    edit_box.press("ControlOrMeta+a")
    edit_box.press_sequentially(EDITED)
    with page.expect_response(
        lambda response: (
            f"/api/v1/comments/{comment_id}" in response.url and response.request.method == "PATCH"
        )
    ) as saving:
        panel.get_by_role("button", name="Save", exact=True).click()
    assert saving.value.ok, f"saving the edit answered {saving.value.status}"
    expect(panel.get_by_text(EDITED)).to_be_visible()
    stored = _stored_thread(owner, prompt_id)
    assert [item["content"] for item in stored] == [EDITED], (
        f"the edited comment was stored as {stored!r}"
    )
