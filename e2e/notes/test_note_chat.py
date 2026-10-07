"""Journey: asking the chat beside a note about the note, and using it as a plain chat.

The Chat button in the note header opens a chat panel. Its suggested prompts summarize the note
or pull action items out of it: the answer shows in the panel and the note stays as written. With
text selected in the editor, "Rewrite the selected text." sends that text along, and the model's
range edit replaces only that part of the note, in the open editor and in the stored note. The
panel is also a plain chat: a model picked by searching its name answers a typed question, the
note stays as written, and the answer's "Insert into note" button puts the answer into the note,
which stores it.

Discriminates: passes on dev 176d31d1d; in a frontend copy, each test fails when its behaviour
is cut: the suggested prompts for a summary and for action items renamed, the panel not sending
the selected text, the model chosen in the selector not reaching the request and "Insert into
note" inserting nothing.
"""

from __future__ import annotations

import json
import time
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import expect_reply, send
from utils.model_selector import select_model

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SUMMARIZE_PROMPT = "Summarize this note."
ACTION_ITEMS_PROMPT = "Extract action items from this note."
REWRITE_PROMPT = "Rewrite the selected text."
SELECTION_MARKER = "Selected note text for replace_note_content operations:"
WORKSPACE_MODEL_PROMPT = "You are the note chat model."


def _unique(prefix: str) -> str:
    return f"{prefix} {uuid.uuid4().hex[:6]}"


def _create_note(owner: Actor, title: str, markdown: str) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": title, "data": {"content": {"md": markdown}}, "access_grants": []},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _stored_markdown(owner: Actor, note_id: str) -> str:
    with owner.client() as client:
        note = client.get(f"/api/v1/notes/{note_id}").json()
    return str(((note.get("data") or {}).get("content") or {}).get("md"))


def _wait_until_stored(owner: Actor, note_id: str, text: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    stored = _stored_markdown(owner, note_id)
    while text not in stored and time.monotonic() < deadline:
        time.sleep(0.2)
        stored = _stored_markdown(owner, note_id)
    assert text in stored, f"the note never stored {text!r}; it holds {stored!r}"


def _note_editor(page: Page) -> Locator:
    return page.get_by_role("main").get_by_label("Write something...")


def _open_note(page: Page, note_id: str, text: str) -> Locator:
    page.goto(f"/notes/{note_id}")
    editor = _note_editor(page)
    expect(editor).to_contain_text(text)
    return editor


def _open_note_chat(page: Page) -> None:
    page.get_by_role("main").get_by_role("button", name="Chat", exact=True).click()


def _requests_answering(upstream, prompt: str) -> list[dict]:
    return [body for body in upstream.chat_requests() if reply.answering(prompt)(body)]


# ---------------------------------------------------------------- suggested prompts


def test_summarizing_the_note_shows_the_answer_and_leaves_the_note_alone(
    page_for, make_user, upstream
):
    author = make_user()
    text = "Launch is on Friday and QA signs off Thursday."
    note_id = _create_note(author, _unique("Plan"), text)
    upstream.queue(
        reply.text(
            "The launch is Friday after QA on Thursday.", match=reply.answering(SUMMARIZE_PROMPT)
        )
    )
    page = page_for(author)
    editor = _open_note(page, note_id, text)

    _open_note_chat(page)
    page.get_by_role("button", name=SUMMARIZE_PROMPT).click()

    expect_reply(page, "The launch is Friday after QA on Thursday.")
    expect(editor).to_have_text(text)
    assert _stored_markdown(author, note_id) == text


def test_extracting_action_items_shows_the_answer_and_leaves_the_note_alone(
    page_for, make_user, upstream
):
    author = make_user()
    text = "Ana books the room. Ben sends the agenda."
    note_id = _create_note(author, _unique("Meeting"), text)
    upstream.queue(
        reply.text(
            "Ana: book the room. Ben: send the agenda.", match=reply.answering(ACTION_ITEMS_PROMPT)
        )
    )
    page = page_for(author)
    editor = _open_note(page, note_id, text)

    _open_note_chat(page)
    page.get_by_role("button", name=ACTION_ITEMS_PROMPT).click()

    expect_reply(page, "Ana: book the room. Ben: send the agenda.")
    expect(editor).to_have_text(text)
    assert _stored_markdown(author, note_id) == text


def test_rewriting_the_selected_text_changes_only_that_part_of_the_note(
    page_for, make_user, upstream
):
    author = make_user()
    kept, selected = "This line stays.", "This line gets rewritten."
    note_id = _create_note(author, _unique("Draft"), f"{kept}\n\n{selected}")
    rewritten = "This line was rewritten."
    start = len(kept) + 1  # the editor stores paragraphs one newline apart
    upstream.queue(
        reply.tool_call(
            "replace_note_content",
            {
                "note_id": note_id,
                "operations": [
                    {
                        "action": "replace_range",
                        "start": start,
                        "end": start + len(selected),
                        "content": rewritten,
                        "expected": selected,
                    }
                ],
            },
            match=reply.answering(REWRITE_PROMPT),
        ),
        reply.text("Rewrote the selected line.", match=reply.answering(REWRITE_PROMPT)),
    )
    page = page_for(author)
    editor = _open_note(page, note_id, selected)
    _wait_until_stored(author, note_id, f"{kept}\n{selected}")
    editor.get_by_text(selected).click(click_count=3)

    _open_note_chat(page)
    page.get_by_role("button", name=REWRITE_PROMPT).click()

    expect(page.get_by_text("Rewrote the selected line.")).to_be_visible()
    expect(editor).to_contain_text(rewritten)
    expect(editor).not_to_contain_text(selected)
    expect(editor).to_contain_text(kept)
    _wait_until_stored(author, note_id, rewritten)
    # the open editor may already have re-saved the edit with a hard line break
    stored = _stored_markdown(author, note_id).strip()
    assert [line.rstrip() for line in stored.splitlines()] == [kept, rewritten]
    first_request = _requests_answering(upstream, REWRITE_PROMPT)[0]
    assert f"{SELECTION_MARKER}\n{selected}" in str(first_request["messages"][-1]["content"])


# ---------------------------------------------------------------- a plain chat


@pytest.fixture
def note_chat_model(make_user) -> tuple[Actor, str]:
    """An admin with a workspace model on the scripted model, whose system prompt is its mark."""
    admin = make_user(role="admin")
    name = _unique("note-chat-model").replace(" ", "-")
    with admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": name,
                "name": name,
                "base_model_id": MOCK_MODEL_ID,
                "meta": {},
                "params": {"system": WORKSPACE_MODEL_PROMPT},
            },
        )
        assert created.status_code == 200, created.text
    return admin, name


def test_the_note_chat_answers_with_the_model_picked_by_name(page_for, note_chat_model, upstream):
    admin, model_name = note_chat_model
    text = "Groceries: milk and eggs."
    note_id = _create_note(admin, _unique("List"), text)
    question = "what is a good breakfast?"
    upstream.queue(reply.text("Eggs on toast.", match=reply.answering(question)))
    page = page_for(admin)
    editor = _open_note(page, note_id, text)

    _open_note_chat(page)
    select_model(page, model_name)
    send(page, question)

    expect_reply(page, "Eggs on toast.")
    expect(editor).to_have_text(text)
    assert _stored_markdown(admin, note_id) == text
    sent = _requests_answering(upstream, question)[0]
    assert WORKSPACE_MODEL_PROMPT in json.dumps(sent["messages"])


def test_insert_into_note_puts_the_answer_into_the_note(page_for, make_user, upstream):
    author = make_user()
    text = "Ideas for the offsite."
    note_id = _create_note(author, _unique("Ideas"), text)
    question = "suggest one activity"
    answer = "Go for a group hike."
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    page = page_for(author)
    editor = _open_note(page, note_id, text)

    _open_note_chat(page)
    send(page, question)
    expect_reply(page, answer)
    page.get_by_role("button", name="Insert into note").click()

    expect(editor).to_contain_text(answer)
    expect(editor).to_contain_text(text)
    _wait_until_stored(author, note_id, answer)
    assert text in _stored_markdown(author, note_id)
