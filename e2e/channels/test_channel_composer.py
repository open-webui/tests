"""Journey: mentioning a person or a model and attaching a file from the channel input.

Typing "@" in the channel input offers the channel's members and the models; picking a member
posts a mention that reads "@<name>", and that member, while elsewhere in the app, sees an unread
count on the channel and a toast naming the sender and the channel that opens the channel when
clicked. Picking a model has it answer in a thread under the message, which the parent shows as
a reply count that opens the thread panel on the answer; with the admin's reply mode set to the
channel the answer lands in the channel itself for every member. A file attached through "Upload
Files" posts with the message and the other member sees it on the message and can open its
content. Typing "#" offers the person's channels; a picked channel posts as a "#<name>" link
that opens that channel for the other member who clicks it.

A message sent while its file still uploads waits in the input's queue and posts by itself once
the upload is done, with a file the other member can open; it posted at once with a file nobody
could open until dev 8a4547104 (open-webui/open-webui#31587). The mention toast showed the raw
mention markup until PR #31601 (open-webui/open-webui#31586). After a pick the input takes focus
back one animation frame later, so the tests wait for that before typing on.

A message typed while an earlier one is still sending waits in the same queue, shown above the
input, and the queue posts in the order typed once the earlier one is done. Before it posts, a
queued message can be removed, or taken back into the input with Edit and sent again at the end of
the queue. "Send now" posts a queued message ahead of one that still waits for its upload. A
thread has a queue of its own: replies typed there while one still sends post into the thread in
order and nothing lands in the channel's queue.

A prompt picked from the "/" menu puts the prompt's text into the input, and that text is what
posts; it put a command chip there that posted as mention markup until
open-webui/open-webui#32713 (fix 1d0b78271).

A model or a channel typed by hand as "@<name>" or "#<name>", without a pick from the list, posts
the same mention a pick does: the model answers in a thread and the channel link opens the channel.
A name two members share stays plain text, since only a pick says which of them is meant, and a
mention copied from a posted message and pasted into the input keeps pointing at the person it named
even then.

Discriminates: passes on dev 176d31d1d; in a frontend copy, a picker that drops the mention type
turns the person and thread model tests red, a toast that goes nowhere on click turns the person
test red, sending the message without its files turns the file test red and a channel link
that opens nothing turns the channel link test red (checked on dev ebc6add67); in a backend copy,
answering in a thread whatever the reply mode turns the channel reply mode test red.
In the pre-fix build of the 0.12.4 changes (1d0b78271 reverted) the prompt test is red: the
input holds a chip in place of the prompt's text.
On dev 226dca18c (frontend built from 948432b43, the same channel input), in a frontend copy,
sending without resolving typed names turns the typed model, typed channel and shared name tests
red, a posted mention that no longer reads as one when copied turns the paste test red, and
resolving a shared name to its first match turns the shared name test red alone.
On dev 3b8205505, in a frontend copy, a queue that takes new messages in front turns all four
queue tests red; in another copy, Send now that sends the first message in place of the chosen one
turns the send now test red alone, and a Delete on a queued message that does nothing turns the
edit test red alone.
"""

from __future__ import annotations

import re
import time
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.access import grant
from harness.channel_chat import enable_channels as enable_channels_with_reply_mode
from harness.channel_quotes import enable_channels, group_channel, post_message
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CLIPBOARD = ["clipboard-read", "clipboard-write"]


@pytest.fixture
def people(admin, preserve, make_user):
    """The sender and another member of one group channel, with its id."""
    preserve("admin_config")
    enable_channels(admin)
    sender, member = make_user(), make_user()
    return sender, member, group_channel(sender, member)


def _open_channel(page_for, account, channel_id: str) -> Page:
    page = page_for(account, path=f"/channels/{channel_id}")
    expect(chat_input(page)).to_be_visible()
    return page


def _message(page: Page, text: str) -> Locator:
    posted = page.locator("[id^='message-']:not(#message-input-container)")
    return posted.filter(has_text=text).first


def _mention(page: Page, query: str, label: str) -> None:
    chat_input(page).click()
    page.keyboard.type(f"@{query}")
    suggestions = page.locator("#suggestions-container")
    suggestions.get_by_role("button", name=label).click()
    # the input takes focus back a frame after the pick; keys sent before that are lost
    expect(chat_input(page)).to_be_focused()


def _sidebar_entry(page: Page, channel_name: str) -> Locator:
    sidebar = page.get_by_role("navigation", name="Chat history")
    open_sidebar = page.get_by_role("button", name="Open Sidebar", exact=True)
    if open_sidebar.is_visible():
        open_sidebar.click()
    section = sidebar.get_by_role("button", name="Channels")
    if section.get_attribute("aria-expanded") == "false":
        section.click()
    return sidebar.get_by_role("link", name=channel_name)


def _mention_member(page: Page, member, text: str) -> None:
    _mention(page, member.name.split()[-1], member.name)
    page.keyboard.type(f" {text}")
    page.keyboard.press("Enter")


def _stored(account, channel_id: str) -> list[dict]:
    with account.client() as client:
        listed = client.get(f"/api/v1/channels/{channel_id}/messages")
    listed.raise_for_status()
    return listed.json()


def test_a_mentioned_member_elsewhere_gets_a_toast_that_opens_the_channel(people, page_for):
    sender, member, channel_id = people
    member_page = page_for(member)
    expect(chat_input(member_page)).to_be_visible()
    page = _open_channel(page_for, sender, channel_id)
    channel_name = page.title().removeprefix("#").removesuffix(" / Open WebUI")
    entry = _sidebar_entry(member_page, channel_name)
    expect(entry).to_be_visible()

    _mention_member(page, member, "can you bring the charts?")

    posted = _message(page, "can you bring the charts?")
    expect(posted.locator(".mention")).to_have_text(f"@{member.name}")
    expect(entry.get_by_title("Unread")).to_have_text("1")
    toast = member_page.get_by_text(f"{sender.name} (#{channel_name})")
    expect(toast).to_be_visible()
    # the sender's own copy shows before the message is stored
    [stored] = _stored(sender, channel_id)
    assert stored["content"].startswith(f"<@U:{member.id}|{member.name}>")
    toast.click()
    expect(member_page).to_have_url(re.compile(f"/channels/{channel_id}"))
    expect(_message(member_page, "can you bring the charts?")).to_be_visible()
    expect(entry.get_by_title("Unread")).to_have_count(0)


def test_the_mention_toast_reads_the_name_and_not_the_mention_markup(people, page_for):
    """#31586, fixed by PR #31601: the toast showed the stored `<@U:<id>|<name>>` markup as text.

    The message itself reads "@<name>"; the toast stripped HTML tags from the content, which left
    the mention markup (not a tag) in place, user id included. Red on dev 176d31d1d.
    """
    sender, member, channel_id = people
    member_page = page_for(member)
    expect(chat_input(member_page)).to_be_visible()
    page = _open_channel(page_for, sender, channel_id)

    _mention_member(page, member, "the dinghy needs air")

    toast_text = member_page.get_by_text("the dinghy needs air")
    expect(toast_text).to_be_visible()
    expect(toast_text).to_contain_text(f"@{member.name}")
    expect(toast_text).not_to_contain_text(member.id)


def _mention_model(page: Page, question: str, picked: bool = True) -> None:
    if picked:
        _mention(page, MOCK_MODEL_ID, MOCK_MODEL_ID)
        page.keyboard.type(f" {question}")
    else:
        chat_input(page).click()
        page.keyboard.type(f"@{MOCK_MODEL_ID} {question}")
    page.keyboard.press("Enter")


def _wait_for_thread_reply(account, channel_id: str, parent_id: str, text: str) -> None:
    deadline = time.monotonic() + 20
    path = f"/api/v1/channels/{channel_id}/messages/{parent_id}/thread"
    with account.client() as client:
        while True:
            thread = client.get(path)
            thread.raise_for_status()
            if any(text in message["content"] for message in thread.json()):
                return
            assert time.monotonic() < deadline, f"no reply with {text!r} was stored in the thread"
            time.sleep(0.2)


@pytest.mark.parametrize("picked", [True, False], ids=["picked", "typed"])
def test_a_mentioned_model_answers_in_a_thread_under_the_message(
    people, page_for, upstream, picked
):
    sender, _, channel_id = people
    question = f"what is the tide at noon? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("High tide, 2.4 metres", match=reply.answering(question)))
    page = _open_channel(page_for, sender, channel_id)

    _mention_model(page, question, picked)

    asked = _message(page, question)
    replies = asked.get_by_role("button", name="1 Replies")
    expect(replies).to_be_visible()
    [stored] = [m for m in _stored(sender, channel_id) if question in m["content"]]
    assert stored["content"].startswith(f"<@M:{MOCK_MODEL_ID}|"), stored["content"]
    # a thread opened while the answer is still written can miss its end and stay blank
    _wait_for_thread_reply(sender, channel_id, stored["id"], "High tide, 2.4 metres")
    replies.click()
    expect(page.get_by_label("Reply to thread...")).to_be_visible()
    expect(_message(page, "High tide, 2.4 metres")).to_contain_text(MOCK_MODEL_ID)
    expect(asked).not_to_contain_text("High tide")
    [sent] = [request for request in upstream.chat_requests() if question in str(request)]
    assert f"<@M:{MOCK_MODEL_ID}" not in str(sent["messages"][-1]["content"])


def test_in_channel_reply_mode_the_model_answers_in_the_channel_for_everyone(
    admin, people, page_for, upstream
):
    sender, member, channel_id = people
    with admin.client() as client:
        enable_channels_with_reply_mode(client, reply_mode="channel")
    question = f"how far to the island? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("About six nautical miles", match=reply.answering(question)))
    page = _open_channel(page_for, sender, channel_id)
    member_page = _open_channel(page_for, member, channel_id)

    _mention_model(page, question)

    answer = _message(member_page, "About six nautical miles")
    expect(answer).to_contain_text(MOCK_MODEL_ID)
    expect(_message(page, "About six nautical miles")).to_be_visible()
    expect(_message(page, question).get_by_role("button", name="1 Replies")).to_have_count(0)


def _attach(page: Page, name: str, text: str) -> None:
    page.get_by_role("button", name="More", exact=True).last.click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menuitem", name="Upload Files").click()
    chooser.value.set_files({"name": name, "mimeType": "text/plain", "buffer": text.encode()})
    expect(page.locator("#message-input-container").get_by_text(name)).to_be_visible()


def _send(page: Page, text: str) -> None:
    chat_input(page).click()
    page.keyboard.type(text)
    page.keyboard.press("Enter")


def _attached_content(account, channel_id: str) -> list[str]:
    """The text of every file on the channel's messages, as `account` downloads it."""
    contents = []
    with account.client() as client:
        for message in client.get(f"/api/v1/channels/{channel_id}/messages").json():
            data = client.get(f"/api/v1/channels/{channel_id}/messages/{message['id']}/data")
            for attached in (data.json() or {}).get("files") or []:
                downloaded = client.get(f"/api/v1/files/{attached['id']}/content")
                contents.append(downloaded.text if downloaded.status_code == 200 else None)
    return contents


def test_a_file_sent_with_a_message_reaches_the_other_member(people, page_for):
    sender, member, channel_id = people
    page = _open_channel(page_for, sender, channel_id)
    member_page = _open_channel(page_for, member, channel_id)

    _attach(page, "crew-list.txt", "Ana, Ben and Cleo")
    # the spinner is the only sign the upload is still running
    expect(page.locator("#message-input-container .spinner_ajPY")).to_have_count(0)
    _send(page, "here is the crew list")

    on_member = _message(member_page, "here is the crew list")
    expect(on_member.get_by_text("crew-list.txt")).to_be_visible()
    assert _attached_content(member, channel_id) == ["Ana, Ben and Cleo"]


def test_a_message_sent_while_its_file_uploads_waits_and_posts_with_the_file(people, page_for):
    """Regression for open-webui/open-webui#31587, fixed in dev 8a4547104.

    The channel input posted a message sent during its upload at once, so the stored file had no id
    and nobody could open it. Now the message waits in the input's queue, nothing is posted until
    the upload is done, and then it posts by itself and the other member downloads the file.
    """
    sender, member, channel_id = people
    page = _open_channel(page_for, sender, channel_id)
    held_uploads = []
    page.route("**/api/v1/files/?*", lambda route: held_uploads.append(route))

    _attach(page, "tide-table.txt", "low tide at six")
    _send(page, "tide table attached")

    expect(page.get_by_role("button", name="Waiting for upload")).to_be_visible()
    assert _stored(sender, channel_id) == [], "#31587: a message was posted during the upload"
    for upload in held_uploads:
        upload.continue_()

    expect(_message(page, "tide table attached")).to_be_visible()
    expect(page.get_by_role("button", name="Waiting for upload")).to_have_count(0)
    member_page = _open_channel(page_for, member, channel_id)
    on_member = _message(member_page, "tide table attached")
    expect(on_member.get_by_text("tide-table.txt")).to_be_visible()
    assert _attached_content(member, channel_id) == ["low tide at six"]


class _HeldPosts:
    """Holds every message post of a page until `release`, so the first one stays sending."""

    def __init__(self, page: Page):
        self.page = page
        self.held = []
        self.holding = True
        page.route("**/api/v1/channels/*/messages/post", self._handle)

    def _handle(self, route) -> None:
        if self.holding:
            self.held.append(route)
        else:
            route.continue_()

    def send_first(self, form: Locator, text: str) -> None:
        with self.page.expect_request(lambda request: request.url.endswith("/messages/post")):
            _type_and_send(form, text)

    def release(self) -> None:
        """Lets every post through; the page must then be waited on, which runs the route."""
        self.holding = False
        for route in self.held:
            route.continue_()


def _form(page: Page, in_thread: bool = False) -> Locator:
    """The channel's input form, or the open thread's, with its queue above the input."""
    thread_input = page.get_by_label("Reply to thread...")
    forms = page.locator("form").filter(has=page.locator("#chat-input"))
    return forms.filter(has=thread_input) if in_thread else forms.filter(has_not=thread_input)


def _type_and_send(form: Locator, text: str) -> None:
    form.locator("#chat-input").click()
    form.page.keyboard.type(text)
    form.page.keyboard.press("Enter")


def _queued(form: Locator, text: str) -> Locator:
    """The queue row showing `text`: the innermost box holding the text and the row's buttons."""
    page = form.page
    return (
        form.locator("div")
        .filter(has=page.get_by_text(text, exact=True))
        .filter(has=page.get_by_role("button", name="Delete"))
        .last
    )


def _queue_rows(form: Locator) -> Locator:
    # every queue row has one Delete button and nothing else in the form does
    return form.get_by_role("button", name="Delete")


def _stored_in_order(account, path: str, texts: list[str]) -> list[str]:
    """The contents at `path` once every text in `texts` is stored, oldest first."""
    deadline = time.monotonic() + 20
    with account.client() as client:
        while True:
            listed = client.get(path)
            listed.raise_for_status()
            messages = sorted(listed.json(), key=lambda message: message["created_at"])
            contents = [message["content"] for message in messages]
            if all(text in contents for text in texts):
                return contents
            assert time.monotonic() < deadline, f"not all of {texts} were stored: {contents}"
            time.sleep(0.2)


def test_messages_typed_while_one_is_still_sending_wait_in_the_queue_and_post_in_order(
    people, page_for
):
    sender, _, channel_id = people
    page = _open_channel(page_for, sender, channel_id)
    form = _form(page)
    posts = _HeldPosts(page)

    posts.send_first(form, "first, the anchor")
    _type_and_send(form, "second, the oars")
    _type_and_send(form, "third, the life vests")

    expect(_queue_rows(form)).to_have_count(2)
    for text in ("second, the oars", "third, the life vests"):
        expect(_queued(form, text).get_by_role("button", name="Send now")).to_be_visible()
    assert _stored(sender, channel_id) == [], "a message was posted while the first still sent"
    posts.release()

    expect(_queue_rows(form)).to_have_count(0)
    texts = ["first, the anchor", "second, the oars", "third, the life vests"]
    stored = _stored_in_order(sender, f"/api/v1/channels/{channel_id}/messages", texts)
    assert stored == texts, f"the queued messages did not post in the order typed: {stored}"
    for text in texts:
        expect(_message(page, text)).to_be_visible()


def test_a_queued_message_can_be_edited_or_removed_before_it_posts(people, page_for):
    sender, _, channel_id = people
    page = _open_channel(page_for, sender, channel_id)
    form = _form(page)
    posts = _HeldPosts(page)
    posts.send_first(form, "first, the anchor")
    for text in ("second, the oars", "third, the life vests", "fourth, the bailer"):
        _type_and_send(form, text)
    expect(_queue_rows(form)).to_have_count(3)

    _queued(form, "third, the life vests").get_by_role("button", name="Delete").click()
    expect(_queue_rows(form)).to_have_count(2)
    expect(form.get_by_text("third, the life vests", exact=True)).to_have_count(0)

    _queued(form, "second, the oars").get_by_role("button", name="Edit").click()
    expect(form.locator("#chat-input")).to_have_text("second, the oars")
    expect(_queue_rows(form)).to_have_count(1)
    page.keyboard.press("End")
    page.keyboard.type(" and the rudder")
    page.keyboard.press("Enter")
    expect(_queued(form, "second, the oars and the rudder")).to_be_visible()
    posts.release()

    expect(_queue_rows(form)).to_have_count(0)
    texts = ["first, the anchor", "fourth, the bailer", "second, the oars and the rudder"]
    stored = _stored_in_order(sender, f"/api/v1/channels/{channel_id}/messages", texts)
    assert stored == texts, f"expected the removed one gone and the edited one last: {stored}"


def test_send_now_posts_a_queued_message_ahead_of_one_waiting_for_its_upload(people, page_for):
    sender, _, channel_id = people
    page = _open_channel(page_for, sender, channel_id)
    form = _form(page)
    held_uploads = []
    page.route("**/api/v1/files/?*", lambda route: held_uploads.append(route))
    _attach(page, "tide-table.txt", "low tide at six")
    _type_and_send(form, "tide table attached")
    expect(_queued(form, "tide table attached")).to_be_visible()
    _type_and_send(form, "leaving at seven")

    waiting = _queued(form, "leaving at seven").get_by_role("button", name="Send now")
    expect(waiting).to_be_enabled()
    waiting.click()

    expect(_queue_rows(form)).to_have_count(1)
    expect(form.get_by_role("button", name="Waiting for upload")).to_be_visible()
    path = f"/api/v1/channels/{channel_id}/messages"
    assert _stored_in_order(sender, path, ["leaving at seven"]) == ["leaving at seven"]
    for upload in held_uploads:
        upload.continue_()

    expect(_queue_rows(form)).to_have_count(0)
    stored = _stored_in_order(sender, path, ["leaving at seven", "tide table attached"])
    assert stored == ["leaving at seven", "tide table attached"], stored


def test_replies_typed_in_a_thread_while_one_still_sends_queue_in_the_thread(people, page_for):
    sender, _, channel_id = people
    parent_id = post_message(sender, channel_id, "who brings what on saturday?")
    page = _open_channel(page_for, sender, channel_id)
    parent = page.locator(f"[id='message-{parent_id}']").first
    parent.hover()
    tooltip_button(parent, "Reply in Thread").click()
    thread = _form(page, in_thread=True)
    expect(thread).to_be_visible()
    posts = _HeldPosts(page)

    posts.send_first(thread, "the tent")
    _type_and_send(thread, "the stove")
    _type_and_send(thread, "the lantern")

    expect(_queue_rows(thread)).to_have_count(2)
    expect(_queued(thread, "the stove")).to_be_visible()
    expect(_queue_rows(_form(page))).to_have_count(0)
    posts.release()

    expect(_queue_rows(thread)).to_have_count(0)
    texts = ["the tent", "the stove", "the lantern"]
    path = f"/api/v1/channels/{channel_id}/messages/{parent_id}/thread"
    stored = _stored_in_order(sender, path, texts)
    assert [text for text in stored if text in texts] == texts, stored
    top_level = [message["content"] for message in _stored(sender, channel_id)]
    assert top_level == ["who brings what on saturday?"], top_level


def test_a_linked_channel_opens_for_the_member_who_clicks_it(people, page_for):
    sender, member, channel_id = people
    linked_id = group_channel(sender, member)
    with sender.client() as client:
        linked_name = client.get(f"/api/v1/channels/{linked_id}").json()["name"]
    page = _open_channel(page_for, sender, channel_id)
    member_page = _open_channel(page_for, member, channel_id)

    chat_input(page).click()
    page.keyboard.type(f"#{linked_name}")
    page.locator("#suggestions-container").get_by_role("button", name=linked_name).click()
    expect(chat_input(page)).to_be_focused()
    page.keyboard.type(" has the packing list")
    page.keyboard.press("Enter")

    link = _message(member_page, "has the packing list").locator(".mention")
    expect(link).to_have_text(f"#{linked_name}")
    # typed by hand this time, not picked from the list
    _send(page, f"#{linked_name} has the tide table")

    typed_link = _message(member_page, "has the tide table").locator(".mention")
    expect(typed_link).to_have_text(f"#{linked_name}")
    stored = [message["content"] for message in _stored(sender, channel_id)]
    assert len(stored) == 2, stored
    for content in stored:
        assert content.startswith(f"<#C:{linked_id}|{linked_name}>"), stored
    link.click()
    expect(member_page).to_have_url(re.compile(f"/channels/{linked_id}"))
    member_page.go_back()
    expect(member_page).to_have_url(re.compile(f"/channels/{channel_id}"))
    typed_link.click()
    expect(member_page).to_have_url(re.compile(f"/channels/{linked_id}"))


@pytest.mark.regression
def test_a_prompt_picked_from_the_slash_menu_posts_its_text(people, page_for, admin):
    # open-webui/open-webui#32713, fix 1d0b78271
    sender, _, channel_id = people
    suffix = uuid.uuid4().hex[:6]
    command = f"tides-{suffix}"
    text = f"When is high water at the harbour mouth? {suffix}"
    form = {
        "command": command,
        "name": f"Tides {suffix}",
        "content": text,
        "access_grants": [grant("user", sender.id, "read")],
    }
    with admin.client() as client:
        created = client.post("/api/v1/prompts/create", json=form)
    assert created.status_code == 200, created.text
    prompt_id = created.json()["id"]
    try:
        page = _open_channel(page_for, sender, channel_id)
        chat_input(page).click()
        page.keyboard.type(f"/{command}")
        page.locator("#suggestions-container").get_by_role("button", name=command).click()

        expect(
            chat_input(page), "the picked prompt did not put its text in the input"
        ).to_have_text(text)
        page.keyboard.press("Enter")
        expect(_message(page, text)).to_be_visible()
        assert [message["content"] for message in _stored(sender, channel_id)] == [text]
    finally:
        with admin.client() as client:
            client.delete(f"/api/v1/prompts/id/{prompt_id}/delete")


@pytest.fixture
def namesakes(people, make_user):
    """Two more members of the channel who share one name."""
    sender, member, channel_id = people
    name = f"Robin {uuid.uuid4().hex[:6]}"
    first, second = make_user(name=name), make_user(name=name)
    with sender.client() as client:
        added = client.post(
            f"/api/v1/channels/{channel_id}/update/members/add",
            json={"user_ids": [first.id, second.id]},
        )
    assert added.status_code == 200, added.text
    return first, second


def _wait_for_stored(account, channel_id: str, text: str) -> str:
    """The content of the one stored message holding `text`, once the server has it."""
    deadline = time.monotonic() + 20
    while True:
        found = [m["content"] for m in _stored(account, channel_id) if text in m["content"]]
        if found:
            [content] = found
            return content
        assert time.monotonic() < deadline, f"no message with {text!r} was stored"
        time.sleep(0.2)


def test_a_typed_name_two_members_share_stays_plain_text(people, namesakes, page_for):
    sender, member, channel_id = people
    first, _ = namesakes
    page = _open_channel(page_for, sender, channel_id)

    _send(page, f"@{first.name} and @{member.name} check the sails")

    stored = _wait_for_stored(sender, channel_id, "check the sails")
    expected = f"@{first.name} and <@U:{member.id}|{member.name}> check the sails"
    assert stored == expected, (
        f"the shared name should stay text, the other become a mention: {stored!r}"
    )


def test_a_mention_copied_from_a_message_keeps_its_person(people, namesakes, page_for):
    sender, member, channel_id = people
    first, _ = namesakes
    post_message(member, channel_id, f"<@U:{first.id}|{first.name}> has the boat keys")
    page = page_for(sender, path=f"/channels/{channel_id}", permissions=CLIPBOARD)
    original = _message(page, "has the boat keys")
    expect(original.locator(".mention")).to_have_text(f"@{first.name}")

    original.get_by_text("has the boat keys").click(click_count=3)
    page.keyboard.press("ControlOrMeta+C")
    chat_input(page).click()
    page.keyboard.press("ControlOrMeta+V")
    expect(chat_input(page)).to_contain_text("has the boat keys")
    page.keyboard.type(" today")
    page.keyboard.press("Enter")

    pasted = _wait_for_stored(sender, channel_id, "has the boat keys today")
    assert pasted.startswith(f"<@U:{first.id}|{first.name}>"), (
        f"the pasted mention no longer points at the person it named: {pasted!r}"
    )
