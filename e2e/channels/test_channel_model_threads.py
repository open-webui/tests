"""Journey: a model talking in a channel's threads, as the members reading along see it.

A model mentioned from inside a thread answers in that thread, also when the admin set models to
answer in the channel, and it is sent what was said in the thread before. A member with the
thread open sees the answer arrive, and the channel itself only shows the parent's reply count.
Replying to the model's answer with "Reply" asks it again without mentioning it. An answer the
provider streams slowly shows its first words to another member before it is finished.

Discriminates: passes on dev ebc6add67; in a backend copy, answering a thread mention in the
channel turns the thread mention test red, leaving the thread history out of the system message
turns its history step red, ignoring the model behind a quoted message turns the reply test red,
and storing the answer only once it is done turns the streaming test red.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.channel_chat import enable_channels as enable_channels_with_reply_mode
from harness.channel_quotes import group_channel, post_message
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def people(admin, preserve, make_user):
    """The asker and another member of one group channel, with its id."""
    preserve("admin_config")
    _reply_mode(admin, "thread")
    asker, member = make_user(), make_user()
    return asker, member, group_channel(asker, member)


def _reply_mode(admin, mode: str) -> None:
    with admin.client() as client:
        enable_channels_with_reply_mode(client, reply_mode=mode)


def _open_channel(page_for, account, channel_id: str) -> Page:
    page = page_for(account)
    page.goto(f"/channels/{channel_id}")
    expect(chat_input(page)).to_be_visible()
    return page


def _in_channel(page: Page, message_id: str) -> Locator:
    return page.locator(f"[id='message-{message_id}']").first


def _in_thread(page: Page, parent_id: str, text: str) -> Locator:
    return page.locator(f"[id^='message-{parent_id}-']").filter(has_text=text).first


def _thread_input(page: Page) -> Locator:
    return page.get_by_label("Reply to thread...")


def _open_thread(page: Page, parent_id: str) -> None:
    parent = _in_channel(page, parent_id)
    parent.hover()
    tooltip_button(parent, "Reply in Thread").click()
    expect(_thread_input(page)).to_be_visible()


def _ask_model(page: Page, box: Locator, question: str) -> None:
    box.click()
    page.keyboard.type(f"@{MOCK_MODEL_ID}")
    page.locator("#suggestions-container").get_by_role("button", name=MOCK_MODEL_ID).click()
    page.keyboard.type(f" {question}")
    page.keyboard.press("Enter")


def _request_asking(upstream, question: str) -> dict:
    [sent] = [request for request in upstream.chat_requests() if question in str(request)]
    return sent


def _message_ids_with(account, channel_id: str, text: str) -> list[str]:
    with account.client() as client:
        listed = client.get(f"/api/v1/channels/{channel_id}/messages")
    listed.raise_for_status()
    return [message["id"] for message in listed.json() if text in message["content"]]


def test_a_model_mentioned_in_a_thread_answers_there_knowing_the_thread(
    admin, people, page_for, upstream
):
    asker, member, channel_id = people
    _reply_mode(admin, "channel")
    parent_id = post_message(member, channel_id, "planning the drive to the coast")
    post_message(member, channel_id, "the coast road is closed on monday", parent_id=parent_id)
    question = f"which day should we leave? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Leave on tuesday", match=reply.answering(question)))
    asker_page = _open_channel(page_for, asker, channel_id)
    member_page = _open_channel(page_for, member, channel_id)
    _open_thread(member_page, parent_id)
    _open_thread(asker_page, parent_id)

    _ask_model(asker_page, _thread_input(asker_page), question)

    answer = _in_thread(member_page, parent_id, "Leave on tuesday")
    expect(answer).to_contain_text(MOCK_MODEL_ID)
    replies = _in_channel(asker_page, parent_id).get_by_role("button", name="3 Replies")
    expect(replies).to_be_visible()
    channel_rows = asker_page.locator(f"[id^='message-']:not([id^='message-{parent_id}-'])")
    expect(channel_rows.filter(has_text="Leave on tuesday")).to_have_count(0)
    system = _request_asking(upstream, question)["messages"][0]
    assert system["role"] == "system"
    assert "the coast road is closed on monday" in system["content"]


def test_replying_to_the_models_answer_asks_it_again(people, page_for, upstream):
    asker, _, channel_id = people
    question = f"how long is the hike? {uuid.uuid4().hex[:6]}"
    follow_up = f"and with a break for lunch? {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.text("About four hours", match=reply.answering(question)),
        reply.text("Then plan for five", match=reply.answering(follow_up)),
    )
    page = _open_channel(page_for, asker, channel_id)
    _ask_model(page, chat_input(page), question)
    replies = page.get_by_role("button", name="1 Replies")
    expect(replies).to_be_visible()
    replies.click()
    [asked] = _message_ids_with(asker, channel_id, question)
    first_answer = _in_thread(page, asked, "About four hours")
    expect(first_answer).to_be_visible()

    first_answer.hover()
    tooltip_button(first_answer, "Reply").click()
    expect(page.get_by_text(f"Replying to {MOCK_MODEL_ID}")).to_be_visible()
    _thread_input(page).click()
    page.keyboard.type(follow_up)
    page.keyboard.press("Enter")

    second_answer = _in_thread(page, asked, "Then plan for five")
    expect(second_answer).to_contain_text(MOCK_MODEL_ID)
    assert f"<@M:{MOCK_MODEL_ID}" not in str(_request_asking(upstream, follow_up)["messages"][-1])


def test_a_slowly_streamed_answer_shows_its_first_words_to_another_member(
    admin, people, page_for, upstream
):
    """Red on dev ebc6add67: a channel answer only shows once the model is done.

    The docs say a channel reply streams as the model writes it. The pipeline now sends each
    piece as a delta of an output item it never announced, and the channel's emitter, which
    starts from an empty output, drops every delta, so the message stays empty until the end.
    """
    asker, member, channel_id = people
    _reply_mode(admin, "channel")
    question = f"what is the forecast? {uuid.uuid4().hex[:6]}"
    pieces = ["The forecast says ", "sun on saturday ", "and rain on sunday"]
    upstream.queue(reply.text(pieces, chunk_delay=1.5, match=reply.answering(question)))
    asker_page = _open_channel(page_for, asker, channel_id)
    member_page = _open_channel(page_for, member, channel_id)

    _ask_model(asker_page, chat_input(asker_page), question)

    answer = member_page.locator("[id^='message-']").filter(has_text="The forecast says").first
    expect(answer).to_be_visible(timeout=30_000)
    assert "rain on sunday" not in answer.inner_text(), (
        "the answer stayed empty until the model was done: the channel never streams it"
    )
    expect(answer).to_contain_text("and rain on sunday")
