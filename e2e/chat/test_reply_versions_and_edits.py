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
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import conversation, expect_reply, last_reply, replies, send
from utils.model_selector import model_search
from utils.search_dialog import open_search, results, search

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def chat_page(page_for, make_user):
    page = page_for(make_user())
    return page


def _reply_buttons(page: Page) -> Locator:
    last_reply(page).hover()
    return conversation(page)


def _regenerate(page: Page) -> None:
    _reply_buttons(page).get_by_role("button", name="Regenerate").last.click()
    page.get_by_text("Try Again").click()


def _last_question(page: Page) -> Locator:
    question = conversation(page).locator(".chat-user").last
    question.hover()
    return question


def test_regenerating_a_reply_keeps_the_first_as_a_version_behind_a_switcher(chat_page, upstream):
    upstream.queue(
        reply.text("first version", match=reply.answering("name a colour")),
        reply.text("second version", match=reply.answering("name a colour")),
    )
    send(chat_page, "name a colour")
    expect_reply(chat_page, "first version")
    expect(conversation(chat_page).get_by_text("1/2")).to_be_hidden()

    _regenerate(chat_page)
    expect_reply(chat_page, "second version")
    expect(conversation(chat_page).get_by_text("2/2")).to_be_visible()

    conversation(chat_page).get_by_role("button", name="Previous message").click()
    expect_reply(chat_page, "first version")
    expect(conversation(chat_page).get_by_text("1/2")).to_be_visible()
    expect(conversation(chat_page).get_by_text("second version")).to_be_hidden()

    conversation(chat_page).get_by_role("button", name="Next message").click()
    expect_reply(chat_page, "second version")


def test_both_versions_of_a_regenerated_reply_are_there_after_a_reload(chat_page, upstream):
    upstream.queue(
        reply.text("kept version", match=reply.answering("name a fruit")),
        reply.text("newer version", match=reply.answering("name a fruit")),
    )
    send(chat_page, "name a fruit")
    expect_reply(chat_page, "kept version")
    _regenerate(chat_page)
    expect_reply(chat_page, "newer version")

    chat_page.reload()
    expect_reply(chat_page, "newer version")
    expect(conversation(chat_page).get_by_text("2/2")).to_be_visible()
    conversation(chat_page).get_by_role("button", name="Previous message").click()
    expect_reply(chat_page, "kept version")


def test_continue_response_appends_to_the_same_reply(chat_page, upstream):
    upstream.queue(
        reply.text("The story begins", match=reply.answering("tell a story")),
        reply.text(" and then it ends.", match=reply.answering("tell a story")),
    )
    send(chat_page, "tell a story")
    expect_reply(chat_page, "The story begins")

    _reply_buttons(chat_page).get_by_role("button", name="Continue Response").click()
    expect_reply(chat_page, "and then it ends.")
    expect(last_reply(chat_page)).to_contain_text("The story begins")
    expect(conversation(chat_page).locator(".chat-assistant")).to_have_count(1)
    expect(conversation(chat_page).get_by_text("2/2")).to_be_hidden()
    resumed_from = upstream.chat_requests()[-1]["messages"][-1]
    assert (resumed_from["role"], resumed_from["content"]) == ("assistant", "The story begins")

    chat_page.reload()
    expect_reply(chat_page, "and then it ends.")
    expect(last_reply(chat_page)).to_contain_text("The story begins")


def test_sending_an_edited_question_starts_a_branch_with_its_own_reply(chat_page, upstream):
    upstream.queue(
        reply.text("answer to the original", match=reply.answering("original question")),
        reply.text("answer to the rewrite", match=reply.answering("rewritten question")),
    )
    send(chat_page, "original question")
    expect_reply(chat_page, "answer to the original")

    question = _last_question(chat_page)
    question.get_by_role("button", name="Edit").click()
    question.locator("textarea").fill("rewritten question")
    question.get_by_role("button", name="Send").click()

    expect_reply(chat_page, "answer to the rewrite")
    expect(conversation(chat_page).get_by_text("rewritten question")).to_be_visible()
    expect(conversation(chat_page).get_by_text("2/2")).to_be_visible()
    expect(conversation(chat_page).get_by_text("original question")).to_be_hidden()

    chat_page.reload()
    expect_reply(chat_page, "answer to the rewrite")
    expect(conversation(chat_page).get_by_text("2/2")).to_be_visible()

    _last_question(chat_page).get_by_role("button", name="Previous message").click()
    expect(conversation(chat_page).get_by_text("original question")).to_be_visible()
    expect_reply(chat_page, "answer to the original")
    expect(conversation(chat_page).get_by_text("answer to the rewrite")).to_be_hidden()
    expect(conversation(chat_page).get_by_text("1/2")).to_be_visible()


def test_saving_an_edited_reply_keeps_the_edit_after_a_reload(chat_page, upstream):
    upstream.queue(reply.text("a reply with a typo", match=reply.answering("say something")))
    send(chat_page, "say something")
    expect_reply(chat_page, "a reply with a typo")

    _reply_buttons(chat_page).get_by_role("button", name="Edit").last.click()
    editor = last_reply(chat_page).locator("textarea")
    editor.fill("a reply without the typo")
    last_reply(chat_page).get_by_role("button", name="Save", exact=True).click()
    expect_reply(chat_page, "a reply without the typo")
    expect(conversation(chat_page).get_by_text("with a typo")).to_be_hidden()
    expect(conversation(chat_page).get_by_text("1/2")).to_be_hidden()

    chat_page.reload()
    expect_reply(chat_page, "a reply without the typo")
    expect(conversation(chat_page).get_by_text("with a typo")).to_be_hidden()


def test_deleting_a_reply_version_leaves_the_other_one_after_a_reload(chat_page, upstream):
    upstream.queue(
        reply.text("version to keep", match=reply.answering("pick a number")),
        reply.text("version to delete", match=reply.answering("pick a number")),
    )
    send(chat_page, "pick a number")
    expect_reply(chat_page, "version to keep")
    _regenerate(chat_page)
    expect_reply(chat_page, "version to delete")

    _reply_buttons(chat_page).get_by_role("button", name="Delete").last.click()
    chat_page.get_by_role("button", name="Confirm").click()
    expect_reply(chat_page, "version to keep")
    expect(conversation(chat_page).get_by_text("version to delete")).to_be_hidden()
    expect(conversation(chat_page).get_by_text("1/2")).to_be_hidden()

    chat_page.reload()
    expect_reply(chat_page, "version to keep")
    expect(conversation(chat_page).get_by_text("version to delete")).to_be_hidden()


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


def _add_model(page: Page, model_id: str) -> None:
    page.get_by_role("button", name=f"Selected model: {MOCK_MODEL_ID}").click()
    page.get_by_role("button", name="Compare").click()
    model_search(page).fill(model_id)
    page.get_by_role("option", name=f"Select {model_id} model").click()
    page.keyboard.press("Escape")
    expect(page.get_by_role("button", name=f"Selected model: {MOCK_MODEL_ID} +1")).to_be_visible()


@pytest.mark.regression
def test_a_model_added_before_regenerate_stays_side_by_side_after_a_reload(
    chat_page, upstream, second_model
):
    question = f"name a river {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.text("the only answer", match=reply.answering(question)),
        reply.text("the left answer", match=reply.answering(question)),
        reply.text("the right answer", match=reply.answering(question)),
    )
    send(chat_page, question)
    expect_reply(chat_page, "the only answer")

    _add_model(chat_page, second_model)
    _regenerate(chat_page)
    for answer in ("the left answer", "the right answer"):
        expect(conversation(chat_page).get_by_text(answer)).to_be_visible()

    chat_page.reload()
    for answer in ("the left answer", "the right answer"):
        expect(conversation(chat_page).get_by_text(answer)).to_be_visible()
    expect(replies(chat_page)).to_have_count(2)


def _column_of(page: Page, model_id: str) -> Locator:
    """The reply written under `model_id`'s name, with its buttons."""
    name = page.get_by_text(model_id, exact=True)
    regenerate = page.get_by_role("button", name="Regenerate")
    return conversation(page).locator("div").filter(has=name).filter(has=regenerate).last


def test_regenerate_in_a_compared_chat_asks_the_model_now_picked_for_that_column(
    chat_page, upstream, second_model, third_model
):
    question = f"name a mountain {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.text("a first peak", match=reply.answering(question)),
        reply.text("a second peak", match=reply.answering(question)),
    )
    _add_model(chat_page, second_model)
    send(chat_page, question)
    expect(_column_of(chat_page, second_model)).to_contain_text("peak")

    chat_page.get_by_role("button", name=f"Selected model: {MOCK_MODEL_ID} +1").click()
    model_search(chat_page).fill(second_model)
    chat_page.get_by_role("option", name=f"Select {second_model} model").click()
    model_search(chat_page).fill(third_model)
    chat_page.get_by_role("option", name=f"Select {third_model} model").click()
    chat_page.keyboard.press("Escape")
    expect(
        chat_page.get_by_role("button", name=f"Selected model: {MOCK_MODEL_ID} +1")
    ).to_be_visible()

    upstream.queue(reply.text("a third peak", match=reply.answering(question)))
    column = _column_of(chat_page, second_model)
    column.hover()
    with chat_page.expect_request(
        lambda request: request.method == "POST" and request.url.endswith("/api/chat/completions")
    ) as sent:
        column.get_by_role("button", name="Regenerate").click()
        chat_page.get_by_text("Try Again").click()
    asked = sent.value.post_data_json["model"]
    assert asked == third_model, f"Regenerate asked {asked}, not the column's new {third_model}"
    expect(conversation(chat_page).get_by_text("a third peak")).to_be_visible()


def _edit_reply(page: Page, text: str) -> Locator:
    _reply_buttons(page).get_by_role("button", name="Edit").last.click()
    editor = last_reply(page).locator("textarea")
    editor.fill(text)
    return last_reply(page)


def _search_api(account, word: str) -> list[dict]:
    with account.client() as client:
        found = client.get("/api/v1/chats/search", params={"text": word})
    assert found.status_code == 200, found.text
    return found.json()


def _search_dialog_hits(page: Page, word: str) -> Locator:
    dialog = open_search(page)
    search(dialog, word)
    return results(dialog)


@pytest.mark.regression
def test_chat_search_finds_a_chat_by_the_text_a_reply_was_edited_to(page_for, make_user, upstream):
    account = make_user()
    page = page_for(account)
    word = f"quokka{uuid.uuid4().hex[:8]}"
    upstream.queue(reply.text("a plain answer", match=reply.answering("say something")))
    send(page, "say something")
    expect_reply(page, "a plain answer")

    editing = _edit_reply(page, f"an answer about the {word}")
    editing.get_by_role("button", name="Save", exact=True).click()
    expect_reply(page, word)

    expect(_search_dialog_hits(page, word)).to_have_count(1)
    assert len(_search_api(account, word)) == 1
    page.reload()
    expect_reply(page, f"an answer about the {word}")


@pytest.mark.regression
def test_chat_search_finds_a_chat_by_the_text_of_a_reply_saved_as_a_copy(
    page_for, make_user, upstream
):
    account = make_user()
    page = page_for(account)
    word = f"quokka{uuid.uuid4().hex[:8]}"
    upstream.queue(reply.text("the original answer", match=reply.answering("say something")))
    send(page, "say something")
    expect_reply(page, "the original answer")

    editing = _edit_reply(page, f"a copy about the {word}")
    editing.get_by_role("button", name="Save As Copy").click()
    expect_reply(page, word)
    expect(conversation(page).get_by_text("2/2")).to_be_visible()

    expect(_search_dialog_hits(page, word)).to_have_count(1)
    assert len(_search_api(account, word)) == 1
    page.reload()
    expect_reply(page, f"a copy about the {word}")
