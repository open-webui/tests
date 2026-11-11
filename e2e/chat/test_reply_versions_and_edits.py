"""Reply versions and message edits, driven through the browser against the scripted model.

Regenerating a reply keeps the old one as a version behind a `1/2` switcher, editing a sent
question starts a branch with its own reply, and an edited reply or a deleted version is what a
reload shows. Each test signs in as a fresh account and scripts every reply by its prompt.

Two tests pin a fixed bug: saving an edited reply, or saving it as a copy, emptied the reply's
plain text, so chat search no longer found the chat by the new words (#31471, fixed by PR #31844).

Discriminates: passes on upstream dev `176d31d1d`; in a frontend build with the reply switcher
removed, the edit-and-send branch replacing the question, an edited reply not saved, continue
sent without its marker or a deleted version not stored, the matching tests turn red; in the
build with de9234a2b reverted (saving leaves the reply's plain text empty) the two search tests
fail on the search finding nothing.

One test pins a fixed bug: a model added to the chat before pressing Regenerate answered side by
side, but a reload showed one column with its reply as another version, since the server kept
the question's old model list (PR open-webui/open-webui#32326, fix 99a9f60b9). It fails on a
backend copy of dev 5032e834e with 99a9f60b9 reverted (the reload shows one reply and `3/3`).

In a chat comparing two models, Regenerate on a column asks the model now picked for that column
(dev 3aaecdd0e). Discriminates: passes on dev c5981aaea, fails in a build that regenerates with
the model that wrote the reply (the request names the replaced model).

Keyboard focus stays on the version switch after switching versions and comes back to Edit
after cancelling a reply edit (ba9274530, PR open-webui/open-webui#33678). Both pass on dev
e8c0ad85e and fail on that build with ba9274530 reverted. Continue Response on a Responses API
connection grows the same paragraph of the reply (Tim's 8366e2872); it passes on dev e8c0ad85e
and on a backend copy with 8366e2872 reverted, and fails on a copy that keeps the continued text
as an output item of its own.

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`), and the tests of
Regenerate, the version switch, Continue Response, Edit, Delete and several models answering in
every way a side chat starts and every state it can be in. A side chat reply had none of these
buttons (open-webui/open-webui#34014), a side chat showed one answer of several models
(open-webui/open-webui#34030) and went blank on switching between regenerated answers
(open-webui/open-webui#34036); fix PR open-webui/open-webui#34064. On dev c15f28649 those side
chat variants are red. A temporary side chat is left out of the tests that reload, as the chat is
gone after the reload.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, expect

from harness import responses_provider as responses_api
from harness import upstream as reply
from harness.python_tools import EVERYONE_READS
from harness.second_provider import OPENAI_CONFIG
from harness.upstream import MOCK_MODEL_ID
from utils.chat_surface import (
    DRAGGED_TAB,
    EVERYWHERE,
    MAIN_CHAT,
    SIDE_CHAT,
    TEMPORARY_SIDE,
    ChatSurface,
)
from utils.chat_ui import conversation, expect_reply, last_reply, replies, send
from utils.model_selector import model_search, select_model
from utils.search_dialog import open_search, results, search, search_dialog

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


# the model count is crossed in the tests that pick several models
# a temporary chat is gone after a reload
EVERY_LASTING_PLACE = tuple(place for place in EVERYWHERE if place != TEMPORARY_SIDE)


@pytest.fixture
def chat(chat_for, make_user) -> ChatSurface:
    return chat_for(make_user())


def _reply_buttons(chat: ChatSurface) -> Locator:
    last_reply(chat).hover()
    return conversation(chat)


def _regenerate(chat: ChatSurface) -> None:
    _reply_buttons(chat).get_by_role("button", name="Regenerate").last.click()
    chat.page.get_by_text("Try Again").click()


def _last_question(chat: ChatSurface) -> Locator:
    question = conversation(chat).locator(".chat-user").last
    question.hover()
    return question


def _replies_to_the_last_question(chat: ChatSurface) -> Locator:
    """The replies after the chat's last question; a side chat can hold earlier turns."""
    question = conversation(chat).locator(".chat-user").last
    return question.locator("xpath=following::*[contains(@class, 'chat-assistant')]")


def _seed_two_versions(account, question: str, first: str, second: str) -> str:
    """A saved chat whose one question has two answers, the second shown; returns its id."""
    question_id, first_id, second_id = (str(uuid.uuid4()) for _ in range(3))
    stored = {"model": MOCK_MODEL_ID, "done": True, "timestamp": 1_700_000_000}
    messages = {
        question_id: {
            **stored,
            "id": question_id,
            "parentId": None,
            "childrenIds": [first_id, second_id],
            "role": "user",
            "content": question,
        },
        first_id: {
            **stored,
            "id": first_id,
            "parentId": question_id,
            "childrenIds": [],
            "role": "assistant",
            "content": first,
        },
        second_id: {
            **stored,
            "id": second_id,
            "parentId": question_id,
            "childrenIds": [],
            "role": "assistant",
            "content": second,
        },
    }
    chat = {
        "title": f"Two answers {uuid.uuid4().hex[:8]}",
        "models": [MOCK_MODEL_ID],
        "history": {"currentId": second_id, "messages": messages},
    }
    with account.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": chat})
    assert created.status_code == 200, created.text
    return created.json()["id"]


@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_regenerating_a_reply_keeps_the_first_as_a_version_behind_a_switcher(chat, upstream):
    """In a side chat: open-webui/open-webui#34014 and #34036."""
    upstream.queue(
        reply.text("first version", match=reply.answering("name a colour")),
        reply.text("second version", match=reply.answering("name a colour")),
    )
    send(chat, "name a colour")
    expect_reply(chat, "first version")
    expect(conversation(chat).get_by_text("1/2")).to_be_hidden()

    _regenerate(chat)
    expect_reply(chat, "second version")
    expect(conversation(chat).get_by_text("2/2")).to_be_visible()

    conversation(chat).get_by_role("button", name="Previous message").click()
    expect_reply(chat, "first version")
    expect(conversation(chat).get_by_text("1/2")).to_be_visible()
    expect(conversation(chat).get_by_text("second version")).to_be_hidden()

    conversation(chat).get_by_role("button", name="Next message").click()
    expect_reply(chat, "second version")


# the version switch in every place is the test above and the saved chat test below
def test_both_versions_of_a_regenerated_reply_are_there_after_a_reload(chat, upstream):
    """In a side chat: open-webui/open-webui#34014 and #34036."""
    upstream.queue(
        reply.text("kept version", match=reply.answering("name a fruit")),
        reply.text("newer version", match=reply.answering("name a fruit")),
    )
    send(chat, "name a fruit")
    expect_reply(chat, "kept version")
    _regenerate(chat)
    expect_reply(chat, "newer version")

    chat = chat.reload()
    expect_reply(chat, "newer version")
    expect(conversation(chat).get_by_text("2/2")).to_be_visible()
    conversation(chat).get_by_role("button", name="Previous message").click()
    expect_reply(chat, "kept version")


# a saved chat opens in the main view or is dragged into a tab, the only places it can be opened
@pytest.mark.chat_surfaces(MAIN_CHAT, SIDE_CHAT, DRAGGED_TAB)
def test_a_saved_chat_with_two_answers_switches_between_them(chat_for, make_user):
    """In a side chat: open-webui/open-webui#34036."""
    account = make_user()
    chat_id = _seed_two_versions(account, "name a lake", "the older lake", "the newer lake")
    chat = chat_for(account, chat_id=chat_id)
    expect_reply(chat, "the newer lake")
    expect(conversation(chat).get_by_text("2/2")).to_be_visible()

    conversation(chat).get_by_role("button", name="Previous message").click()
    expect(last_reply(chat), "the chat went blank after switching versions").to_contain_text(
        "the older lake"
    )
    expect(conversation(chat).get_by_text("1/2")).to_be_visible()

    conversation(chat).get_by_role("button", name="Next message").click()
    expect_reply(chat, "the newer lake")
    expect(conversation(chat).get_by_text("2/2")).to_be_visible()


@pytest.mark.chat_surfaces(*EVERY_LASTING_PLACE)
def test_continue_response_appends_to_the_same_reply(chat, upstream):
    """In a side chat: open-webui/open-webui#34014."""
    upstream.queue(
        reply.text("The story begins", match=reply.answering("tell a story")),
        reply.text(" and then it ends.", match=reply.answering("tell a story")),
    )
    send(chat, "tell a story")
    expect_reply(chat, "The story begins")

    _reply_buttons(chat).get_by_role("button", name="Continue Response").click()
    expect_reply(chat, "and then it ends.")
    expect(last_reply(chat)).to_contain_text("The story begins")
    expect(_replies_to_the_last_question(chat)).to_have_count(1)
    expect(conversation(chat).get_by_text("2/2")).to_be_hidden()
    resumed_from = upstream.chat_requests()[-1]["messages"][-1]
    assert (resumed_from["role"], resumed_from["content"]) == ("assistant", "The story begins")

    chat = chat.reload()
    expect_reply(chat, "and then it ends.")
    expect(last_reply(chat)).to_contain_text("The story begins")


@pytest.fixture
def responses_model(admin, preserve, listener):
    preserve(OPENAI_CONFIG)
    with admin.client() as client:
        yield responses_api.connect_responses(client, listener)


def test_continue_response_on_a_responses_connection_appends_to_the_same_reply(
    chat_for, make_user, responses_model
):
    """In a side chat: open-webui/open-webui#34014."""
    continued = responses_api.completed()
    # the real API repeats the round's whole output when it completes
    continued["response"]["output"] = [
        {
            "type": "message",
            "id": "msg_2",
            "role": "assistant",
            "content": [{"type": "output_text", "text": " and then it ends."}],
        }
    ]
    responses_model.answer(
        responses_api.events_stream(
            *responses_api.message("The story begins"), responses_api.completed()
        ),
        responses_api.events_stream(
            *responses_api.message(" and then ", "it ends.", item_id="msg_2"), continued
        ),
    )
    # an admin account, so the Responses connection's model needs no access grant
    chat = chat_for(make_user(role="admin"))
    select_model(chat, responses_api.RESPONSES_MODEL)
    send(chat, "tell a story")
    expect_reply(chat, "The story begins")

    _reply_buttons(chat).get_by_role("button", name="Continue Response").click()

    # the continued text runs on in the same paragraph, never as a block of its own
    joined_text = "The story begins and then it ends."
    expect(last_reply(chat).locator("p", has_text=joined_text)).to_have_count(1)
    expect(replies(chat)).to_have_count(1)
    chat = chat.reload()
    expect(last_reply(chat).locator("p", has_text=joined_text)).to_have_count(1)


def test_switching_versions_from_the_keyboard_keeps_focus_on_the_switch(chat, upstream):
    """In a side chat: open-webui/open-webui#34014."""
    # the switch was drawn anew for the other version and focus fell to the page (#33678)
    upstream.queue(
        reply.text("first take", match=reply.answering("name a river")),
        reply.text("second take", match=reply.answering("name a river")),
    )
    send(chat, "name a river")
    expect_reply(chat, "first take")
    _regenerate(chat)
    expect_reply(chat, "second take")

    previous = conversation(chat).get_by_role("button", name="Previous message")
    previous.focus()
    chat.page.keyboard.press("Enter")
    expect_reply(chat, "first take")
    expect(previous, "focus left the version switch after going back").to_be_focused()

    following = conversation(chat).get_by_role("button", name="Next message")
    following.focus()
    chat.page.keyboard.press("Enter")
    expect_reply(chat, "second take")
    expect(following, "focus left the version switch after going forward").to_be_focused()


def test_cancelling_a_reply_edit_returns_focus_to_its_edit_button(chat, upstream):
    """In a side chat: open-webui/open-webui#34014."""
    upstream.queue(reply.text("a reply to keep", match=reply.answering("say a thing")))
    send(chat, "say a thing")
    expect_reply(chat, "a reply to keep")
    edit_button = _reply_buttons(chat).get_by_role("button", name="Edit").last
    edit_button.focus()
    chat.page.keyboard.press("Enter")
    expect(last_reply(chat).locator("textarea")).to_be_visible()

    last_reply(chat).get_by_role("button", name="Cancel", exact=True).focus()
    chat.page.keyboard.press("Enter")

    expect(last_reply(chat).locator("textarea")).to_have_count(0)
    expect(edit_button, "focus did not come back to Edit after cancelling (#33678)").to_be_focused()


@pytest.mark.chat_surfaces(*EVERY_LASTING_PLACE)
def test_sending_an_edited_question_starts_a_branch_with_its_own_reply(chat, upstream):
    """In a side chat: open-webui/open-webui#34014."""
    upstream.queue(
        reply.text("answer to the original", match=reply.answering("original question")),
        reply.text("answer to the rewrite", match=reply.answering("rewritten question")),
    )
    send(chat, "original question")
    expect_reply(chat, "answer to the original")

    question = _last_question(chat)
    question.get_by_role("button", name="Edit").click()
    question.locator("textarea").fill("rewritten question")
    question.get_by_role("button", name="Send").click()

    expect_reply(chat, "answer to the rewrite")
    expect(conversation(chat).get_by_text("rewritten question")).to_be_visible()
    expect(conversation(chat).get_by_text("2/2")).to_be_visible()
    expect(conversation(chat).get_by_text("original question")).to_be_hidden()

    chat = chat.reload()
    expect_reply(chat, "answer to the rewrite")
    expect(conversation(chat).get_by_text("2/2")).to_be_visible()

    _last_question(chat).get_by_role("button", name="Previous message").click()
    expect(conversation(chat).get_by_text("original question")).to_be_visible()
    expect_reply(chat, "answer to the original")
    expect(conversation(chat).get_by_text("answer to the rewrite")).to_be_hidden()
    expect(conversation(chat).get_by_text("1/2")).to_be_visible()


@pytest.mark.chat_surfaces(*EVERY_LASTING_PLACE)
def test_saving_an_edited_reply_keeps_the_edit_after_a_reload(chat, upstream):
    """In a side chat: open-webui/open-webui#34014."""
    upstream.queue(reply.text("a reply with a typo", match=reply.answering("say something")))
    send(chat, "say something")
    expect_reply(chat, "a reply with a typo")

    _reply_buttons(chat).get_by_role("button", name="Edit").last.click()
    editor = last_reply(chat).locator("textarea")
    editor.fill("a reply without the typo")
    last_reply(chat).get_by_role("button", name="Save", exact=True).click()
    expect_reply(chat, "a reply without the typo")
    expect(conversation(chat).get_by_text("with a typo")).to_be_hidden()
    expect(conversation(chat).get_by_text("1/2")).to_be_hidden()

    chat = chat.reload()
    expect_reply(chat, "a reply without the typo")
    expect(conversation(chat).get_by_text("with a typo")).to_be_hidden()


@pytest.mark.chat_surfaces(*EVERY_LASTING_PLACE)
def test_deleting_a_reply_version_leaves_the_other_one_after_a_reload(chat, upstream):
    """In a side chat: open-webui/open-webui#34014."""
    upstream.queue(
        reply.text("version to keep", match=reply.answering("pick a number")),
        reply.text("version to delete", match=reply.answering("pick a number")),
    )
    send(chat, "pick a number")
    expect_reply(chat, "version to keep")
    _regenerate(chat)
    expect_reply(chat, "version to delete")

    _reply_buttons(chat).get_by_role("button", name="Delete").last.click()
    chat.page.get_by_role("button", name="Confirm").click()
    expect_reply(chat, "version to keep")
    expect(conversation(chat).get_by_text("version to delete")).to_be_hidden()
    expect(conversation(chat).get_by_text("1/2")).to_be_hidden()

    chat = chat.reload()
    expect_reply(chat, "version to keep")
    expect(conversation(chat).get_by_text("version to delete")).to_be_hidden()


def _shared_preset(admin, prefix: str):
    """A preset on the scripted model that every account may add to a chat."""
    model_id = f"{prefix}-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": model_id,
        "meta": {},
        "params": {},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        saved = client.post("/api/v1/models/create", json=form)
        assert saved.status_code == 200, saved.text
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def second_model(admin):
    yield from _shared_preset(admin, "second")


@pytest.fixture
def third_model(admin):
    yield from _shared_preset(admin, "third")


def _add_model(chat: ChatSurface, model_id: str) -> None:
    page = chat.page
    chat.button(f"Selected model: {MOCK_MODEL_ID}").click()
    page.get_by_role("button", name="Compare").click()
    model_search(page).fill(model_id)
    page.get_by_role("option", name=f"Select {model_id} model").click()
    page.keyboard.press("Escape")
    expect(chat.button(f"Selected model: {MOCK_MODEL_ID} +1")).to_be_visible()


@pytest.mark.regression
@pytest.mark.chat_surfaces(*EVERY_LASTING_PLACE)
def test_a_model_added_before_regenerate_stays_side_by_side_after_a_reload(
    chat, upstream, second_model
):
    """In a side chat: open-webui/open-webui#34014 and #34030."""
    question = f"name a river {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.text("the only answer", match=reply.answering(question)),
        reply.text("the left answer", match=reply.answering(question)),
        reply.text("the right answer", match=reply.answering(question)),
    )
    send(chat, question)
    expect_reply(chat, "the only answer")

    _add_model(chat, second_model)
    _regenerate(chat)
    for answer in ("the left answer", "the right answer"):
        expect(conversation(chat).get_by_text(answer)).to_be_visible()

    chat = chat.reload()
    for answer in ("the left answer", "the right answer"):
        expect(conversation(chat).get_by_text(answer)).to_be_visible()
    expect(_replies_to_the_last_question(chat)).to_have_count(2)


def _column_of(chat: ChatSurface, model_id: str) -> Locator:
    """The reply written under `model_id`'s name, with its buttons."""
    name = chat.page.get_by_text(model_id, exact=True)
    regenerate = chat.page.get_by_role("button", name="Regenerate")
    return conversation(chat).locator("div").filter(has=name).filter(has=regenerate).last


@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_regenerate_in_a_compared_chat_asks_the_model_now_picked_for_that_column(
    chat, upstream, second_model, third_model
):
    """In a side chat: open-webui/open-webui#34014 and #34030."""
    page = chat.page
    question = f"name a mountain {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.text("a first peak", match=reply.answering(question)),
        reply.text("a second peak", match=reply.answering(question)),
    )
    _add_model(chat, second_model)
    send(chat, question)
    expect(_column_of(chat, second_model)).to_contain_text("peak")

    chat.button(f"Selected model: {MOCK_MODEL_ID} +1").click()
    model_search(page).fill(second_model)
    page.get_by_role("option", name=f"Select {second_model} model").click()
    model_search(page).fill(third_model)
    page.get_by_role("option", name=f"Select {third_model} model").click()
    page.keyboard.press("Escape")
    expect(chat.button(f"Selected model: {MOCK_MODEL_ID} +1")).to_be_visible()

    upstream.queue(reply.text("a third peak", match=reply.answering(question)))
    column = _column_of(chat, second_model)
    column.hover()
    with page.expect_request(
        lambda request: request.method == "POST" and request.url.endswith("/api/chat/completions")
    ) as sent:
        column.get_by_role("button", name="Regenerate").click()
        page.get_by_text("Try Again").click()
    asked = sent.value.post_data_json["model"]
    assert asked == third_model, f"Regenerate asked {asked}, not the column's new {third_model}"
    expect(conversation(chat).get_by_text("a third peak")).to_be_visible()


def _edit_reply(chat: ChatSurface, text: str) -> Locator:
    _reply_buttons(chat).get_by_role("button", name="Edit").last.click()
    editor = last_reply(chat).locator("textarea")
    editor.fill(text)
    return last_reply(chat)


def _search_api(account, word: str) -> list[dict]:
    with account.client() as client:
        found = client.get("/api/v1/chats/search", params={"text": word})
    assert found.status_code == 200, found.text
    return found.json()


def _search_dialog_hits(chat: ChatSurface, word: str) -> Locator:
    dialog = open_search(chat)
    search(dialog, word)
    return results(dialog)


def _close_search(chat: ChatSurface) -> None:
    """Close the search dialog, which would cover the side chat's tab the reload selects."""
    chat.page.keyboard.press("Escape")
    expect(search_dialog(chat.page)).to_be_hidden()


@pytest.mark.regression
def test_chat_search_finds_a_chat_by_the_text_a_reply_was_edited_to(chat_for, make_user, upstream):
    """In a side chat: open-webui/open-webui#34014."""
    account = make_user()
    chat = chat_for(account)
    word = f"quokka{uuid.uuid4().hex[:8]}"
    upstream.queue(reply.text("a plain answer", match=reply.answering("say something")))
    send(chat, "say something")
    expect_reply(chat, "a plain answer")

    editing = _edit_reply(chat, f"an answer about the {word}")
    editing.get_by_role("button", name="Save", exact=True).click()
    expect_reply(chat, word)

    expect(_search_dialog_hits(chat, word)).to_have_count(1)
    assert len(_search_api(account, word)) == 1
    _close_search(chat)
    chat = chat.reload()
    expect_reply(chat, f"an answer about the {word}")


@pytest.mark.regression
def test_chat_search_finds_a_chat_by_the_text_of_a_reply_saved_as_a_copy(
    chat_for, make_user, upstream
):
    """In a side chat: open-webui/open-webui#34014."""
    account = make_user()
    chat = chat_for(account)
    word = f"quokka{uuid.uuid4().hex[:8]}"
    upstream.queue(reply.text("the original answer", match=reply.answering("say something")))
    send(chat, "say something")
    expect_reply(chat, "the original answer")

    editing = _edit_reply(chat, f"a copy about the {word}")
    editing.get_by_role("button", name="Save As Copy").click()
    expect_reply(chat, word)
    expect(conversation(chat).get_by_text("2/2")).to_be_visible()

    expect(_search_dialog_hits(chat, word)).to_have_count(1)
    assert len(_search_api(account, word)) == 1
    _close_search(chat)
    chat = chat.reload()
    expect_reply(chat, f"a copy about the {word}")
