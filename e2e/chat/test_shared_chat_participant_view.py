"""Journey: what the owner and the people replying see of each other in an "Allow replies" chat.

While one of them types in the message box, the other's chat shows "<name> is typing..." above
theirs, and nobody sees their own name there. In the chat bubble layout each person's own
messages sit on the right and everybody else's on the left. Someone replying may branch off any
prompt with Edit (Send only, no Save in place) and Regenerate any reply, as 2a6e0da05 and
788a61180 (v0.12.3) allow, but rates no reply and continues only their own; the owner rates and
continues only their own replies and regenerates everyone's. Stop shows to the person whose reply
is running.

The Stop test is red on dev 226dca18c: while the owner's reply runs, the page of someone replying
shows Stop as well (the composer counts any unfinished reply as theirs), and clicking it does
nothing, since the page refuses to stop a reply it did not start. Present since the shared replies
landed in de73bb830 (v0.12.0).

Discriminates: the other four pass on dev 226dca18c. One frontend build of it with four edits
fails each of them: the chat page ignoring typing events fails the typing test, every prompt
bubble set on the right fails the bubble test, Edit shown only in one's own chats (as before
2a6e0da05) fails the replying test, and rating and Continue offered on every reply in one's own
chat fails the owner test.
"""

from __future__ import annotations

import re
import threading
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.access import grant
from utils.chat_ui import chat_input, conversation, expect_reply, send, stop_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

QUESTION = "is the lighthouse open?"
ANSWER = "Open until dusk."
MEMBER_QUESTION = "can we climb it?"
MEMBER_ANSWER = "Yes, all the way up."


def _accounts(make_user):
    suffix = uuid.uuid4().hex[:6]
    return make_user(name=f"Ada Owner {suffix}"), make_user(name=f"Ben Crew {suffix}")


def _ask(page: Page, upstream, question: str, answer: str, hold=None) -> None:
    upstream.queue(reply.text(answer, match=reply.answering(question), hold=hold))
    send(page, question)
    if hold is None:
        expect_reply(page, answer)


def _share_continue(owner, chat_id: str, member) -> None:
    """Share the chat the way the dialog does with Allow replies: the link, then the grant."""
    with owner.client() as client:
        shared = client.post(f"/api/v1/chats/{chat_id}/share", json={"share_mode": None})
        assert shared.status_code == 200, shared.text
        saved = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update",
            json={"access_grants": [grant("user", member.id, "read")], "share_mode": "continue"},
        )
    assert saved.status_code == 200, saved.text


def _shared_chat(page_for, make_user, upstream) -> tuple[object, object, Page, Page]:
    """The owner's answered chat, shared with Allow replies and open in both accounts' pages."""
    owner, member = _accounts(make_user)
    owner_page = page_for(owner)
    _ask(owner_page, upstream, QUESTION, ANSWER)
    expect(owner_page).to_have_url(re.compile(r"/c/"))
    chat_id = owner_page.url.rsplit("/", 1)[-1]
    _share_continue(owner, chat_id, member)
    member_page = page_for(member, path=f"/c/{chat_id}")
    expect(conversation(member_page).get_by_text(ANSWER)).to_be_visible()
    return owner, member, owner_page, member_page


def _with_both_turns(page_for, make_user, upstream):
    """As `_shared_chat`, with a turn of the member's that shows in the owner's open page."""
    owner, member, owner_page, member_page = _shared_chat(page_for, make_user, upstream)
    _ask(member_page, upstream, MEMBER_QUESTION, MEMBER_ANSWER)
    expect(_message(owner_page, MEMBER_ANSWER)).to_be_visible()
    return owner, member, owner_page, member_page


def _message(page: Page, text: str) -> Locator:
    """The message that holds `text`, with its action buttons."""
    return conversation(page).get_by_role("listitem").filter(has_text=text)


def _actions(page: Page, text: str) -> dict[str, int]:
    """How many of each action button the message holding `text` offers."""
    message = _message(page, text)
    expect(message).to_have_count(1)
    labels = ["Edit", "Delete", "Good Response", "Bad Response", "Continue Response", "Regenerate"]
    return {label: message.get_by_label(label, exact=True).count() for label in labels}


def _sees_present(page: Page, other) -> None:
    """The navbar shows `other` with the chat open, so both pages have joined it live."""
    navbar = page.get_by_role("navigation").filter(has=page.get_by_role("button", name="Controls"))
    # presence avatars carry no name, only the person's picture
    expect(navbar.locator(f'img[src$="/users/{other.id}/profile/image"]')).to_be_visible()


def _typing_line(page: Page) -> Locator:
    return page.get_by_text(re.compile(r"is typing\.\.\.$"))


def test_each_person_sees_who_else_is_typing(page_for, make_user, upstream):
    owner, member, owner_page, member_page = _shared_chat(page_for, make_user, upstream)
    _sees_present(owner_page, member)
    _sees_present(member_page, owner)

    chat_input(member_page).click()
    member_page.keyboard.type("half a thought")

    expect(_typing_line(owner_page)).to_have_text(f"{member.name} is typing...")
    expect(_typing_line(member_page)).to_have_count(0)

    chat_input(owner_page).click()
    owner_page.keyboard.type("me too")

    expect(_typing_line(member_page)).to_have_text(f"{owner.name} is typing...")


def test_a_person_replying_edits_any_prompt_and_regenerates_any_reply_but_rates_none(
    page_for, make_user, upstream
):
    owner, member, owner_page, member_page = _with_both_turns(page_for, make_user, upstream)

    owner_prompt = _actions(member_page, QUESTION)
    assert owner_prompt == {
        **dict.fromkeys(["Delete", "Good Response", "Bad Response"], 0),
        **{"Continue Response": 0, "Regenerate": 0, "Edit": 1},
    }, f"the member's actions on the owner's prompt: {owner_prompt}"
    assert _actions(member_page, MEMBER_QUESTION)["Edit"] == 1
    assert _actions(member_page, MEMBER_QUESTION)["Delete"] == 0
    owner_reply = _actions(member_page, ANSWER)
    assert (owner_reply["Regenerate"], owner_reply["Good Response"]) == (1, 0), owner_reply
    assert owner_reply["Bad Response"] == owner_reply["Continue Response"] == 0, owner_reply
    member_reply = _actions(member_page, MEMBER_ANSWER)
    assert (member_reply["Regenerate"], member_reply["Continue Response"]) == (1, 1), member_reply
    assert member_reply["Good Response"] == member_reply["Bad Response"] == 0, member_reply

    # the first prompt sits under the chat's header, which takes a pointer click
    edit = _message(member_page, QUESTION).get_by_label("Edit", exact=True)
    edit.focus()
    member_page.keyboard.press("Enter")
    expect(member_page.get_by_role("button", name="Send", exact=True)).to_be_visible()
    expect(member_page.get_by_role("button", name="Save", exact=True)).to_have_count(0)


def test_the_owner_rates_and_continues_only_their_own_replies(page_for, make_user, upstream):
    owner, member, owner_page, member_page = _with_both_turns(page_for, make_user, upstream)

    owner_reply = _actions(owner_page, ANSWER)
    assert (owner_reply["Good Response"], owner_reply["Regenerate"]) == (1, 1), owner_reply
    member_reply = _actions(owner_page, MEMBER_ANSWER)
    assert (member_reply["Regenerate"], member_reply["Good Response"]) == (1, 0), member_reply
    assert member_reply["Continue Response"] == 0, member_reply
    assert _actions(owner_page, MEMBER_QUESTION)["Edit"] == 1


def test_other_peoples_messages_sit_on_the_left_in_bubble_view(page_for, make_user, upstream):
    owner, member, owner_page, member_page = _with_both_turns(page_for, make_user, upstream)

    for page, own, theirs in (
        (member_page, MEMBER_QUESTION, QUESTION),
        (owner_page, QUESTION, MEMBER_QUESTION),
    ):
        middle = _middle(conversation(page))
        own_box = _message(page, own).get_by_text(own, exact=True).bounding_box()
        their_box = _message(page, theirs).get_by_text(theirs, exact=True).bounding_box()
        assert own_box["x"] > middle, f"the viewer's own {own!r} is not on the right: {own_box}"
        assert their_box["x"] + their_box["width"] < middle, (
            f"the other person's {theirs!r} is not on the left: {their_box}"
        )


def _middle(region: Locator) -> float:
    box = region.bounding_box()
    return box["x"] + box["width"] / 2


def test_stop_shows_to_someone_replying_only_while_their_own_reply_runs(
    page_for, make_user, upstream
):
    """Red on dev 226dca18c: Stop shows over the owner's running reply too, and does nothing."""
    owner, member, owner_page, member_page = _shared_chat(page_for, make_user, upstream)
    members_reply, owners_reply = threading.Event(), threading.Event()
    try:
        _ask(member_page, upstream, MEMBER_QUESTION, MEMBER_ANSWER, hold=members_reply)
        expect(stop_button(member_page)).to_be_visible()
    finally:
        members_reply.set()
    expect_reply(owner_page, MEMBER_ANSWER)
    expect(stop_button(member_page)).to_have_count(0)

    try:
        _ask(owner_page, upstream, "when is the last tour?", "At four.", hold=owners_reply)
        expect(_message(member_page, "when is the last tour?")).to_be_visible()
        expect(stop_button(owner_page)).to_be_visible()
        expect(
            stop_button(member_page), "the member is offered Stop on the owner's running reply"
        ).to_have_count(0)
    finally:
        owners_reply.set()
    expect_reply(member_page, "At four.")
