"""Journey: sharing a chat by link, opening it as someone else, and taking the link back.

The owner opens the chat's Share dialog, creates the link and grants one account read access.
That account opens `/s/{id}` in its own browser and reads the conversation, and Clone Chat copies
it into the account's own chats to carry on from; an account left off the access list is sent
back home. Once the owner deletes the link, the granted account is sent home too. Messages sent
after sharing stay out of the link until the owner presses Update and Copy Link, after which the
same link shows them. A signed-out visitor is sent to the sign-in page, which returns them to the
chat (open-webui/open-webui#31337): the page used to send them home, where they were asked to sign
in and then landed on a new chat with the link lost.

Discriminates: passes on dev ac00d40e3; in a backend copy, with `DELETE /api/v1/chats/{id}/share`
answering true without removing the share the deletion test fails (the old link still opens),
with the shared clone route storing an empty conversation the clone test fails,
and with `can_read_shared_chat` granting any signed-in account the stranger test fails. With
`2178777340` reverted (the 015dbc861 mutation build) the signed-out test fails (sent home). On
dev ebc6add67 a frontend copy whose share dialog hands back the old link without sharing again
turns the update test red.

Fork chat on a reply of the link opens a chat of the account's own that holds the link's messages
up to that reply and none the owner sent after sharing. On dev 948432b43 a backend copy whose
fork route copies the whole branch whatever reply was chosen fails the fork test.

A chat opened under `/s/`, by its link or by an admin viewing another account's chat, is fetched
once and stays where the reader scrolls it. From 0.12.3 the page fetched it again and again and
scrolled to the bottom after each fetch, so scrolling up was impossible
(open-webui/open-webui#32725, issue #32657, fix c4153a6ee). Both cases fail in the pre-fix build of
the 0.12.4 changes, which has that commit reverted.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat import ask
from harness.chat_history import seed_chat
from utils.chat_ui import chat_input, conversation, expect_reply, send
from utils.member_picker import pick_member

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

QUESTION = "which birds winter by the lake?"
ANSWER = "herons and a few grebes"
LATER_QUESTION = "and in spring?"
LATER_ANSWER = "swallows over the reeds"


def _share_dialog(page: Page) -> Locator:
    page.get_by_role("button", name="Chat actions").first.click()
    page.get_by_role("menu").get_by_role("button", name="Share").click()
    return page.get_by_role("dialog").filter(has_text="Share Chat")


@pytest.fixture
def shared_chat(page_for, make_user, upstream):
    """A chat the owner has shared by link: the owner's page, its Share dialog and the link."""
    owner = make_user()
    page = page_for(owner)
    upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
    send(page, QUESTION)
    expect_reply(page, ANSWER)

    dialog = _share_dialog(page)
    dialog.get_by_role("button", name="Copy Link").click()
    link = dialog.get_by_role("link", name="You have shared this chat before")
    expect(link).to_be_visible()
    return page, dialog, link.get_attribute("href")


def _grant(dialog: Locator, account) -> None:
    dialog.get_by_role("button", name="Add Access").click()
    picker = dialog.page.get_by_role("dialog").filter(has_text="Add Access").last
    # the picker lists one page of users; searching reaches any account
    picker.get_by_placeholder("Search").fill(account.name)
    pick_member(picker, account.name)
    picker.get_by_role("button", name="Add", exact=True).click()
    expect(dialog.page.get_by_text("Access updated")).to_be_visible()


def _sent_home(page: Page) -> None:
    expect(page).to_have_url(re.compile(r"/$"))
    expect(page.get_by_text(ANSWER)).to_have_count(0)


def test_a_granted_account_reads_the_link_until_the_owner_deletes_it(
    shared_chat, page_for, make_user
):
    owner_page, dialog, share_path = shared_chat
    viewer = make_user()
    _grant(dialog, viewer)

    viewer_page = page_for(viewer, path=share_path)
    expect(conversation(viewer_page).get_by_text(QUESTION)).to_be_visible()
    expect(conversation(viewer_page).get_by_text(ANSWER)).to_be_visible()

    dialog.get_by_role("button", name="delete this link").click()
    expect(dialog.get_by_role("button", name="Copy Link")).to_be_visible()

    viewer_page.goto(share_path)
    _sent_home(viewer_page)


def test_a_granted_account_clones_the_chat_and_carries_on_in_its_copy(
    shared_chat, page_for, make_user, upstream
):
    owner_page, dialog, share_path = shared_chat
    viewer = make_user()
    _grant(dialog, viewer)
    viewer_page = page_for(viewer, path=share_path)
    viewer_page.get_by_role("button", name="Clone Chat").click()

    expect(viewer_page).to_have_url(re.compile(r"/c/"))
    expect_reply(viewer_page, ANSWER)
    upstream.queue(reply.text("mostly coots", match=reply.answering("anything else")))
    send(viewer_page, "anything else?")
    expect_reply(viewer_page, "mostly coots")
    sent = upstream.chat_requests()[-1]["messages"]
    assert [entry["content"] for entry in sent if entry["role"] == "assistant"] == [ANSWER]

    viewer_page.reload()
    expect_reply(viewer_page, "mostly coots")
    expect(conversation(viewer_page).get_by_text(ANSWER)).to_be_visible()
    owner_page.reload()
    expect_reply(owner_page, ANSWER)
    expect(conversation(owner_page).get_by_text("mostly coots")).to_have_count(0)


def test_an_account_left_off_the_access_list_is_sent_home(shared_chat, page_for, make_user):
    _, _, share_path = shared_chat
    stranger_page = page_for(make_user(), path=share_path)
    _sent_home(stranger_page)


@pytest.mark.regression
def test_a_signed_out_visitor_signs_in_and_comes_back_to_the_link(shared_chat, make_user, page):
    _, dialog, share_path = shared_chat
    viewer = make_user()
    _grant(dialog, viewer)

    page.goto(share_path)

    expect(page, "a signed-out visitor was not sent to sign in (#31337)").to_have_url(
        re.compile(r"/auth\?redirect=" + re.escape(share_path.replace("/", "%2F")) + "$")
    )
    page.get_by_label("Email").fill(viewer.email)
    page.get_by_label("Password", exact=True).fill(viewer.password)
    page.get_by_role("button", name="Sign in", exact=True).click()
    expect(page).to_have_url(re.compile(re.escape(share_path) + "$"))
    expect(conversation(page).get_by_text(ANSWER)).to_be_visible()


def test_updating_the_link_shares_the_messages_sent_since(
    shared_chat, page_for, make_user, upstream
):
    owner_page, dialog, share_path = shared_chat
    viewer = make_user()
    _grant(dialog, viewer)
    dialog.get_by_role("button", name="Close").click()
    upstream.queue(reply.text(LATER_ANSWER, match=reply.answering(LATER_QUESTION)))
    send(owner_page, LATER_QUESTION)
    expect_reply(owner_page, LATER_ANSWER)

    viewer_page = page_for(viewer, path=share_path)
    expect(conversation(viewer_page).get_by_text(ANSWER)).to_be_visible()
    expect(conversation(viewer_page).get_by_text(LATER_ANSWER)).to_have_count(0)

    _share_dialog(owner_page).get_by_role("button", name="Update and Copy Link").click()
    expect(owner_page.get_by_text("Copied shared chat URL to clipboard!")).to_be_visible()

    viewer_page.reload()
    expect(conversation(viewer_page).get_by_text(LATER_ANSWER)).to_be_visible()
    expect(conversation(viewer_page).get_by_text(ANSWER)).to_be_visible()


def test_fork_on_a_reply_of_the_link_opens_a_copy_up_to_that_reply(page_for, make_user, upstream):
    owner, viewer = make_user(), make_user()
    first_answer = f"herons {uuid.uuid4().hex[:6]}"
    with owner.client() as client:
        chat_id, last_id = seed_chat(
            client,
            [
                {"role": "user", "content": QUESTION},
                {"role": "assistant", "content": first_answer},
                {"role": "user", "content": LATER_QUESTION},
                {"role": "assistant", "content": LATER_ANSWER},
            ],
        )
        shared = client.post(f"/api/v1/chats/{chat_id}/share")
        assert shared.status_code == 200, shared.text
        grant = {"principal_type": "user", "principal_id": viewer.id, "permission": "read"}
        granted = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update", json={"access_grants": [grant]}
        )
        assert granted.status_code == 200, granted.text
        upstream.queue(reply.text("only for the owner", match=reply.answering("after sharing")))
        ask(client, "sent after sharing", chat_id=chat_id, parent_id=last_id)

    page = page_for(viewer, path=f"/s/{shared.json()['share_id']}")
    expect(conversation(page).get_by_text(LATER_ANSWER)).to_be_visible()
    first_reply = (
        conversation(page)
        .get_by_text(first_answer)
        .locator("xpath=ancestor::*[starts-with(@id, 'message-')][1]")
    )
    first_reply.hover()
    with page.expect_response(lambda response: response.url.endswith("/fork")) as forked:
        first_reply.get_by_role("button", name="Fork chat").click()
    assert forked.value.ok, forked.value.text()
    fork_id = forked.value.json()["id"]

    expect(page).to_have_url(re.compile(rf"/c/{fork_id}$"))
    expect_reply(page, first_answer)
    expect(conversation(page).get_by_text(LATER_ANSWER)).to_have_count(0)
    expect(chat_input(page)).to_be_visible()
    with viewer.client() as client:
        stored = client.get(f"/api/v1/chats/{fork_id}")
    assert stored.status_code == 200, stored.text
    assert stored.json()["user_id"] == viewer.id
    assert "only for the owner" not in str(stored.json()["chat"])


AT_THE_BOTTOM = "box => box.scrollHeight - box.scrollTop - box.clientHeight < 5"
# drags the chat to its top each frame, as older messages load in above, until the first shows
TO_THE_FIRST_MESSAGE = """([box, first]) => {
    box.scrollTop = 0;
    return box.innerText.includes(first) && box.scrollTop === 0;
}"""


def _long_chat(owner) -> tuple[str, str, str]:
    """A chat long enough to scroll, shared by link; returns its id, share id and first line."""
    filler = "The heron waits by the reeds for a long while. " * 30
    opening = f"Opening line {uuid.uuid4().hex[:6]}"
    messages = [{"role": "user", "content": opening}]
    for turn in range(10):
        messages.append({"role": "assistant", "content": f"{filler} Entry {turn}"})
        messages.append({"role": "user", "content": f"Question {turn}"})
    with owner.client() as client:
        chat_id, _ = seed_chat(client, messages)
        shared = client.post(f"/api/v1/chats/{chat_id}/share")
    assert shared.status_code == 200, shared.text
    return chat_id, shared.json()["share_id"], opening


@pytest.mark.regression
@pytest.mark.parametrize("opened_by", ["share link", "admin viewing the chat"])
def test_a_shared_chat_loads_once_and_stays_where_it_was_scrolled(page_for, make_user, opened_by):
    # open-webui/open-webui#32725 (issue #32657), fix c4153a6ee
    owner = make_user()
    chat_id, share_id, opening = _long_chat(owner)
    if opened_by == "share link":
        viewer, page_id = owner, share_id
    else:
        viewer, page_id = make_user(role="admin"), chat_id
    page = page_for(viewer, path=f"/s/{page_id}")
    last = conversation(page).get_by_text("Question 9", exact=True)
    expect(last).to_be_in_viewport()
    messages = page.locator("#messages-container")
    page.wait_for_function(AT_THE_BOTTOM, arg=messages.element_handle())
    fetches: list[str] = []
    page.on(
        "request",
        lambda request: (
            fetches.append(request.url) if f"/api/v1/chats/share/{page_id}" in request.url else None
        ),
    )

    page.wait_for_function(TO_THE_FIRST_MESSAGE, arg=[messages.element_handle(), opening])
    page.wait_for_timeout(2000)  # a reloading page fetches the chat again within this

    assert fetches == [], f"the open chat was fetched {len(fetches)} more times (#32657)"
    # checked once, as it stands after the wait
    expect(
        conversation(page).get_by_text(opening, exact=True),
        "the page jumped back to the bottom (#32657)",
    ).to_be_in_viewport(timeout=1)
