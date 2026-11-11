"""A reply its parent did not list lost its branch navigation in the chat view.

Fix `59d3c5b06` (#29299, open-webui 0.11.3). The chat view builds a reply's `1/2` sibling switcher
from the parent's `childrenIds`; a reply stored without that link showed as the only answer and
its sibling was unreachable. Reading the chat now relinks every orphaned message first.

Twin of unit/models/test_chat_branch_links.py.

Discriminates: passes on upstream dev `bbfa876af`; with the relink loop removed from
`_repair_chat_current_id` the reply shows no `2/2` switcher.

#34126: a window that missed a turn and then saved an edit stored its own, older position, and
the page opens a chat on that stored position, so the chat reopened on the older turn and the
next message started a new branch there. A save now keeps the newest turn of that branch open.
Discriminates: the out-of-date window test fails on upstream dev `eecba38c5` (a page opened before
its live connection is up shows only the first turn) and passes with the fix 967471e29 on branch
fix/stale-save-reopens-older-turn; the second open window test passes on both.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.inflight import wait_until_no_reply_runs
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import conversation, expect_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

# Points the page's live connection at a closed port, as for a window that has not reconnected yet.
WITHOUT_LIVE_CONNECTION = """
(() => {
  const RealWebSocket = window.WebSocket;
  function BlockedWebSocket(url, protocols) {
    const target = String(url).includes("/ws/") ? "ws://127.0.0.1:1/blocked" : url;
    return new RealWebSocket(target, protocols);
  }
  BlockedWebSocket.prototype = RealWebSocket.prototype;
  Object.assign(BlockedWebSocket, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 });
  window.WebSocket = BlockedWebSocket;
})();
"""


def _message(message_id: str, parent_id: str | None, children: list[str], content: str) -> dict:
    role = "user" if parent_id is None else "assistant"
    return {
        "id": message_id,
        "parentId": parent_id,
        "childrenIds": children,
        "role": role,
        "content": content,
        "timestamp": 1_700_000_000,
        **({"model": MOCK_MODEL_ID, "done": True} if role == "assistant" else {}),
    }


def test_an_orphaned_reply_gets_its_sibling_switcher(page_for, make_user):
    owner = make_user()
    messages = [
        _message("question", None, ["first"], "Which colour?"),
        _message("first", "question", [], "The first answer"),
        _message("second", "question", [], "The second answer"),
    ]
    chat = {
        "title": "Two answers",
        "models": [MOCK_MODEL_ID],
        "history": {"currentId": "second", "messages": {entry["id"]: entry for entry in messages}},
    }
    with owner.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": chat})
    assert created.status_code == 200, created.text

    page = page_for(owner, path=f"/c/{created.json()['id']}")
    expect_reply(page, "The second answer")
    expect(conversation(page).get_by_text("2/2")).to_be_visible()

    conversation(page).get_by_role("button", name="Previous message").click()
    expect_reply(page, "The first answer")
    expect(conversation(page).get_by_text("The second answer")).to_be_hidden()


def _history(current_id: str, *messages: dict) -> dict:
    return {"currentId": current_id, "messages": {entry["id"]: entry for entry in messages}}


def test_a_chat_saved_from_an_out_of_date_window_opens_on_its_newest_turn(page_for, make_user):
    owner = make_user()
    question = _message("question", None, ["answer"], "First question")
    answer = _message("answer", "question", [], "The first answer")
    follow_up = {**_message("follow-up", "answer", ["late"], "Second question"), "role": "user"}
    late_answer = _message("late", "follow-up", [], "The second answer")
    answer_with_follow_up = {**answer, "childrenIds": ["follow-up"]}
    edited_question = {**question, "content": "First question, typo fixed"}

    with owner.client() as client:
        chat = {
            "title": "Two windows",
            "models": [MOCK_MODEL_ID],
            "history": _history("answer", question, answer),
        }
        created = client.post("/api/v1/chats/new", json={"chat": chat})
        assert created.status_code == 200, created.text
        chat_id = created.json()["id"]
        # The second save comes from a window that never saw the second turn.
        for saved_history in (
            _history("late", question, answer_with_follow_up, follow_up, late_answer),
            _history("answer", edited_question, answer),
        ):
            saved = client.post(
                f"/api/v1/chats/{chat_id}", json={"chat": {"history": saved_history}}
            )
            assert saved.status_code == 200, saved.text

    # Once connected, the page catches up on its own; until then it opens where the chat was stored.
    page = page_for(owner, path=f"/c/{chat_id}", init_script=WITHOUT_LIVE_CONNECTION)
    expect(conversation(page).get_by_text("First question, typo fixed")).to_be_visible()
    expect(
        conversation(page).get_by_text("The second answer"),
        "the chat reopened on an older turn (#34126)",
    ).to_be_visible()


def test_an_edit_saved_from_a_second_open_window_keeps_the_newest_turn(
    page_for, make_user, upstream
):
    owner = make_user()
    question = _message("question", None, ["answer"], "First question")
    answer = _message("answer", "question", [], "The first answer")
    with owner.client() as client:
        chat = {
            "title": "Two windows",
            "models": [MOCK_MODEL_ID],
            "history": _history("answer", question, answer),
        }
        created = client.post("/api/v1/chats/new", json={"chat": chat})
        assert created.status_code == 200, created.text
        chat_id = created.json()["id"]

        first_window = page_for(owner, path=f"/c/{chat_id}")
        second_window = page_for(owner, path=f"/c/{chat_id}")
        expect_reply(second_window, "The first answer")
        upstream.queue(reply.text("The second answer", match=reply.answering("Second question")))
        send(first_window, "Second question")
        expect_reply(second_window, "The second answer")
        wait_until_no_reply_runs(client, chat_id)

    first_question = conversation(second_window).locator(".chat-user").first
    first_question.hover()
    first_question.get_by_role("button", name="Edit").click()
    first_question.locator("textarea").fill("First question, typo fixed")
    with second_window.expect_response(
        lambda response: (
            response.request.method == "POST" and response.url.endswith(f"/api/v1/chats/{chat_id}")
        )
    ) as saved:
        first_question.get_by_role("button", name="Save", exact=True).click()
    assert saved.value.ok, saved.value.status

    reopened = page_for(owner, path=f"/c/{chat_id}")
    expect(conversation(reopened).get_by_text("First question, typo fixed")).to_be_visible()
    expect_reply(reopened, "The second answer")
