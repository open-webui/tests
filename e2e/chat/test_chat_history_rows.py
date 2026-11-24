"""Journey: a chat shows every turn, version and branch the way it was left, after every reload.

open-webui/open-webui#34688 asks to store a chat's messages one by one instead of rewriting the
whole conversation on every message (PR open-webui/open-webui#34689). What a person sees must not
change: several turns, a regenerated reply switched back to its first version, an edited question
with its own branch and a deleted version all read the same after a reload; a clone, a shared link
and a search result open on the whole conversation; and a long chat opens, takes one more turn and
reopens with it. The integration twin, `integration/chat/test_chat_history_rows.py`, reads the
same stored values over the API; these tests pin what the page shows.

Discriminates: passes on dev b6aac5dee (before the PR) and on dev 414340682 (the PR merged).
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from harness.chat import NO_BACKGROUND_TASKS, ChatTurn, ask, wait_for_reply
from harness.upstream import MOCK_MODEL_ID
from utils.chat_surface import MAIN_CHAT, ChatSurface
from utils.chat_ui import conversation, expect_reply, last_reply, send
from utils.search_dialog import open_search, open_selected, result, results, search
from utils.sidebar_chats import chat_menu, chat_row

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def _tag() -> str:
    return uuid.uuid4().hex[:8]


def _messages(chat: ChatSurface | object) -> Locator:
    return conversation(chat).locator(".chat-user, .chat-assistant")


def _expect_conversation(chat, texts: list[str]) -> None:
    """The shown questions and replies, in order, each holding its text."""
    shown = _messages(chat)
    expect(shown).to_have_count(len(texts))
    for index, text in enumerate(texts):
        expect(shown.nth(index)).to_contain_text(text)


def _regenerate(chat: ChatSurface) -> None:
    last_reply(chat).hover()
    conversation(chat).get_by_role("button", name="Regenerate").last.click()
    chat.page.get_by_text("Try Again").click()


def _question(chat: ChatSurface, number: int) -> Locator:
    """The chat's question `number`, by place: once edited its text sits in a text box."""
    question = conversation(chat).locator(".chat-user").nth(number - 1)
    question.hover()
    return question


# the side chat places of each step are pinned in test_reply_versions_and_edits.py
@pytest.mark.chat_surfaces(MAIN_CHAT)
def test_turns_versions_a_branch_and_a_deleted_version_read_the_same_after_each_reload(
    chat_for, make_user, upstream
):
    tag = _tag()
    q = [f"question {n} {tag}" for n in (1, 2, 3)]
    a = [f"answer {n} {tag}" for n in (1, 2, 3)]
    upstream.queue(
        *(reply.text(answer, match=reply.answering(question)) for question, answer in zip(q, a)),
        reply.text(f"answer 3 again {tag}", match=reply.answering(q[2])),
        reply.text(f"answer to the rewrite {tag}", match=reply.answering(f"rewritten {tag}")),
    )
    chat = chat_for(make_user())
    for question, answer in zip(q, a):
        send(chat, question)
        expect_reply(chat, answer)
    turns = [q[0], a[0], q[1], a[1], q[2], a[2]]
    _expect_conversation(chat, turns)
    chat = chat.reload()
    _expect_conversation(chat, turns)

    _regenerate(chat)
    expect_reply(chat, f"answer 3 again {tag}")
    newest = [*turns[:5], f"answer 3 again {tag}"]
    _expect_conversation(chat, newest)
    expect(conversation(chat).get_by_text("2/2")).to_be_visible()
    conversation(chat).get_by_role("button", name="Previous message").click()
    expect(conversation(chat).get_by_text("1/2")).to_be_visible()
    _expect_conversation(chat, turns)
    conversation(chat).get_by_role("button", name="Next message").click()
    expect(conversation(chat).get_by_text("2/2")).to_be_visible()
    _expect_conversation(chat, newest)
    # a reload opens on the newest version, as test_reply_versions_and_edits.py pins
    chat = chat.reload()
    _expect_conversation(chat, newest)
    expect(conversation(chat).get_by_text("2/2"), "the reload lost a version").to_be_visible()

    last_reply(chat).hover()
    with chat.page.expect_response(lambda response: "/messages/" in response.url) as deleted:
        conversation(chat).get_by_role("button", name="Delete").last.click()
        chat.page.get_by_role("button", name="Confirm").click()
    assert deleted.value.ok, f"deleting the version failed: {deleted.value.text()}"
    _expect_conversation(chat, turns)
    expect(conversation(chat).get_by_text("1/2")).to_have_count(0)
    chat = chat.reload()
    _expect_conversation(chat, turns)
    expect(conversation(chat).get_by_text("1/2"), "the deleted version came back").to_have_count(0)

    question = _question(chat, 2)
    question.get_by_role("button", name="Edit").click()
    question.locator("textarea").fill(f"rewritten {tag}")
    question.get_by_role("button", name="Send").click()
    expect_reply(chat, f"answer to the rewrite {tag}")
    branch = [q[0], a[0], f"rewritten {tag}", f"answer to the rewrite {tag}"]
    _expect_conversation(chat, branch)
    chat = chat.reload()
    _expect_conversation(chat, branch)
    expect(conversation(chat).get_by_text("2/2")).to_be_visible()

    _question(chat, 2).get_by_role("button", name="Previous message").click()
    _expect_conversation(chat, turns)
    expect(conversation(chat).get_by_text("1/2")).to_be_visible()
    expect(conversation(chat).get_by_text(f"answer 3 again {tag}")).to_have_count(0)


def _regenerated_chat(account, upstream, tag: str, title: str) -> tuple[str, list[str]]:
    """A titled chat of two turns whose last reply was regenerated; returns its id and the texts
    it shows, the newer version last."""
    turns = [(f"bake bread {tag}", f"knead it {tag}"), (f"how long? {tag}", f"an hour {tag}")]
    with account.client() as client:
        previous: ChatTurn | None = None
        for question, answer in turns:
            upstream.queue(reply.text(answer, match=reply.answering(question)))
            options = {}
            if previous:
                options = {"chat_id": previous.chat_id, "parent_id": previous.assistant_message_id}
            previous, _ = ask(client, question, **options)
        chat_id = previous.chat_id
        stored = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]
        asked = stored[previous.user_message_id]
        newer_id = str(uuid.uuid4())
        upstream.queue(reply.text(f"two hours {tag}", match=reply.answering(asked["content"])))
        regenerated = client.post(
            "/api/chat/completions",
            json={
                "model": MOCK_MODEL_ID,
                "messages": [{"role": "user", "content": asked["content"]}],
                "stream": True,
                "chat_id": chat_id,
                "parent_id": asked["parentId"],
                "id": newer_id,
                "user_message": {**asked, "childrenIds": [*asked["childrenIds"], newer_id]},
                "session_id": f"harness-{uuid.uuid4().hex[:8]}",
                "background_tasks": NO_BACKGROUND_TASKS,
            },
        )
        assert regenerated.status_code == 200, regenerated.text
        wait_for_reply(client, ChatTurn(chat_id, asked["id"], newer_id))
        titled = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": title}})
        assert titled.status_code == 200, titled.text
    return chat_id, [turns[0][0], turns[0][1], turns[1][0], f"two hours {tag}"]


def test_a_clone_a_shared_link_and_a_search_result_open_the_whole_conversation(
    page_for, make_user, upstream
):
    owner, reader = make_user(), make_user()
    tag = _tag()
    title = f"Bread {tag}"
    chat_id, shown = _regenerated_chat(owner, upstream, tag, title)
    with owner.client() as client:
        shared = client.post(f"/api/v1/chats/{chat_id}/share")
        assert shared.status_code == 200, shared.text
        grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
        granted = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update", json={"access_grants": [grant]}
        )
        assert granted.status_code == 200, granted.text

    page = page_for(owner, path=f"/c/{chat_id}")
    _expect_conversation(page, shown)
    expect(conversation(page).get_by_text("2/2")).to_be_visible()

    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    expect(chat_row(sidebar, title)).to_be_visible()
    chat_menu(sidebar, title).get_by_role("menuitem", name="Clone").click()
    expect(chat_row(sidebar, f"Clone of {title}")).to_be_visible()
    expect(page).not_to_have_url(re.compile(rf"/c/{chat_id}$"))
    _expect_conversation(page, shown)
    page.reload()
    _expect_conversation(page, shown)
    expect(
        conversation(page).get_by_text("2/2"), "the clone lost the older version"
    ).to_be_visible()
    conversation(page).get_by_role("button", name="Previous message").click()
    expect_reply(page, f"an hour {tag}")

    dialog = open_search(page)
    search(dialog, f"two hours {tag}")
    expect(results(dialog)).to_have_count(2)
    expect(result(dialog, f"Clone of {title}")).to_have_count(1)
    result(dialog, title).filter(has_not_text="Clone of").click()
    open_selected(dialog)
    expect(page).to_have_url(re.compile(rf"/c/{chat_id}(\?.*)?$"))
    _expect_conversation(page, shown)

    shared_page = page_for(reader, path=f"/s/{shared.json()['share_id']}")
    _expect_conversation(shared_page, shown)
    shared_page.reload()
    _expect_conversation(shared_page, shown)


def test_a_long_chat_opens_takes_another_turn_and_reopens_with_it(page_for, make_user, upstream):
    account = make_user()
    tag = _tag()
    count = 30
    with account.client() as client:
        previous: ChatTurn | None = None
        for n in range(1, count + 1):
            question = f"turn {n} {tag}"
            upstream.queue(reply.text(f"reply {n} {tag}", match=reply.answering(question)))
            options = {}
            if previous:
                options = {"chat_id": previous.chat_id, "parent_id": previous.assistant_message_id}
            previous, _ = ask(client, question, **options)
    chat_id = previous.chat_id

    page = page_for(account, path=f"/c/{chat_id}")
    expect_reply(page, f"reply {count} {tag}")
    question = f"one more {tag}"
    upstream.queue(reply.text(f"the last reply {tag}", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, f"the last reply {tag}")
    newest = [f"turn {count} {tag}", f"reply {count} {tag}", question, f"the last reply {tag}"]

    page.reload()
    expect_reply(page, f"the last reply {tag}")
    shown = _messages(page)
    expect(shown.last).to_contain_text(newest[-1])
    tail = [shown.nth(index) for index in range(-4, 0)]
    for message, text in zip(tail, newest):
        expect(message).to_contain_text(text)
    with account.client() as client:
        stored = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]
    assert len(stored["messages"]) == 2 * count + 2, f"the chat holds {len(stored['messages'])}"
