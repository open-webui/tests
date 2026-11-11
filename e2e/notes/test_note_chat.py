"""Journey: asking the chat beside a note about the note, and using it as a plain chat.

The Chat button in the note header opens a chat panel. Its suggested prompts summarize the note
or pull action items out of it: the answer shows in the panel and the note stays as written. With
text selected in the editor, "Rewrite the selected text." sends that text along, and the model's
range edit replaces only that part of the note, in the open editor and in the stored note.
Since dev b34885efc the note tool edits only a note the model read in the same turn, so the
scripted model reads it with `view_note` first. The panel is also a plain chat: a model picked
by searching its name answers a typed question, the note stays as written, and the answer's
"Insert into note" button puts the answer into the note, which stores it.

Discriminates: passes on dev 176d31d1d; in a frontend copy, each test fails when its behaviour
is cut: the suggested prompts for a summary and for action items renamed, the panel not sending
the selected text, the model chosen in the selector not reaching the request and "Insert into
note" inserting nothing.

Since dev da850248d a note opens as a tab beside the chat tab, whose message box stays in the
page unseen, so the typed questions go into the note chat's own box, found by its name. Notes
open directly in their page, so a test never waits for the app to load a second time, and
every answer is read in the chat, since dev c15f28649 repeats a finished note chat answer in a
notification. The module passes five runs in a row on dev c15f28649. In a copy of that build
each test fails when its behaviour is cut: the two suggested prompts renamed, the selected text
left out of the message and "Insert into note" inserting nothing. In a backend copy, the picked
model's system prompt left out of the request turns the model test red.

The page's own chat keeps working next to a note chat. Text typed into a second, new chat next to
the note leaves a new chat's unsent text alone, and the regenerate shortcut in the chat's message
box regenerates the chat's reply. Both are red on dev c15f28649 (the note chat's text shows in
the new chat, the shortcut leaves the chat's reply as it was) and pass on the build of
3cab22263, fix PR open-webui/open-webui#34064, and on dev e791cbdde, which has it. A question
the model asks with Ask User is answered in its own chat with the other one on the page, in both
directions; it passes on both.
The input request that fix also covers (an MCP elicitation) has no test: the MCP server of the
harness refuses to ask on the connection Open WebUI opens.
"""

from __future__ import annotations

import json
import re
import time
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor
from harness.chat_history import seed_chat
from harness.upstream import MOCK_MODEL_ID
from utils.chat_surface import ChatSurface, main_chat
from utils.chat_ui import REPLY_TIMEOUT_MS, expect_reply
from utils.model_selector import select_model
from utils.note_chat import open_note_chat, send_in_note_chat

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


def _open_note(page_for, actor: Actor, note_id: str, text: str) -> tuple[Page, Locator]:
    """The note in a page of the actor's own, opened directly so it is handed over loaded."""
    page = page_for(actor, path=f"/notes/{note_id}")
    editor = _note_editor(page)
    expect(editor).to_contain_text(text)
    return page, editor


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
    page, editor = _open_note(page_for, author, note_id, text)

    open_note_chat(page)
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
    page, editor = _open_note(page_for, author, note_id, text)

    open_note_chat(page)
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
            "view_note",
            {"note_id": note_id},
            call_id="call_view",
            match=reply.answering(REWRITE_PROMPT),
        ),
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
            call_id="call_edit",
            match=reply.answering(REWRITE_PROMPT),
        ),
        reply.text("Rewrote the selected line.", match=reply.answering(REWRITE_PROMPT)),
    )
    page, editor = _open_note(page_for, author, note_id, selected)
    _wait_until_stored(author, note_id, f"{kept}\n{selected}")
    editor.get_by_text(selected).click(click_count=3)

    open_note_chat(page)
    page.get_by_role("button", name=REWRITE_PROMPT).click()

    expect_reply(page, "Rewrote the selected line.")
    expect(editor).to_contain_text(rewritten)
    expect(editor).not_to_contain_text(selected)
    expect(editor).to_contain_text(kept)
    _wait_until_stored(author, note_id, rewritten)
    # the open editor may already have re-saved the edit with a hard line break
    stored = _stored_markdown(author, note_id).strip()
    assert [line.rstrip() for line in stored.splitlines()] == [kept, rewritten]
    first_request = _requests_answering(upstream, REWRITE_PROMPT)[0]
    assert f"{SELECTION_MARKER}\n{selected}" in str(
        reply.conversation(first_request)[-1]["content"]
    )


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
    page, editor = _open_note(page_for, admin, note_id, text)

    open_note_chat(page)
    select_model(page, model_name)
    send_in_note_chat(page, question)

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
    page, editor = _open_note(page_for, author, note_id, text)

    open_note_chat(page)
    send_in_note_chat(page, question)
    expect_reply(page, answer)
    page.get_by_role("button", name="Insert into note").click()

    expect(editor).to_contain_text(answer)
    expect(editor).to_contain_text(text)
    _wait_until_stored(author, note_id, answer)
    assert text in _stored_markdown(author, note_id)


# ---------------------------------------------------------------- next to the page's own chat


def _note_tab_chat(page: Page) -> ChatSurface:
    """The chat beside the open note, held as a chat of its own."""
    # the note chat's pane has no role or name, only an id of its own
    panel = page.locator("[id^='note-chat-container-']")
    expect(panel).to_have_count(1)
    return ChatSurface(page, panel, "note chat")


def _note_opened_from_a_chat(page_for, author: Actor, note_id: str, text: str) -> Page:
    """The note opened in a page that showed a saved chat of `author` before."""
    with author.client() as client:
        chat_id, _ = seed_chat(
            client, [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        )
    page = page_for(author, path=f"/c/{chat_id}")
    expect_reply(page, "hi")
    page.goto(f"/notes/{note_id}")
    expect(_note_editor(page)).to_contain_text(text, timeout=REPLY_TIMEOUT_MS)
    return page


def _wait_until_draft_stored(page: Page, draft: str) -> None:
    """Unsent text is stored in the tab's session a moment after the last key."""
    page.wait_for_function(
        "draft => Object.values(sessionStorage).some((value) => value.includes(draft))",
        arg=draft,
    )


def test_text_typed_into_a_new_note_chat_leaves_a_new_chats_unsent_text_alone(
    page_for, make_user, upstream
):
    """Red on dev c15f28649; fix PR open-webui/open-webui#34064 (3cab22263)."""
    author = make_user()
    first_question = _unique("what else should I pack?")
    upstream.queue(reply.text("Gloves and a hat.", match=reply.answering(first_question)))
    text = "Packing list for the hut."
    note_id = _create_note(author, _unique("Hut"), text)
    main_draft = _unique("Ask about the hut opening hours")
    note_draft = _unique("Add a head torch")
    page = page_for(author)
    main_chat(page).input.click()
    page.keyboard.type(main_draft)
    _wait_until_draft_stored(page, main_draft)

    page.goto(f"/notes/{note_id}")
    expect(_note_editor(page)).to_contain_text(text, timeout=REPLY_TIMEOUT_MS)
    open_note_chat(page)
    send_in_note_chat(page, first_question)
    note_chat = _note_tab_chat(page)
    note_chat.expect_reply("Gloves and a hat.")
    # a second, new chat next to the note; the answer's notification can cover the button
    note_chat.button("Chat history").press("Enter")
    page.get_by_role("menu").get_by_text("New chat", exact=True).click()
    expect(note_chat.replies).to_have_count(0)
    note_chat.input.click()
    page.keyboard.type(note_draft)
    _wait_until_draft_stored(page, note_draft)
    page.goto("/")

    expect(main_chat(page).input, "the note chat's text replaced the new chat's").to_have_text(
        main_draft, timeout=REPLY_TIMEOUT_MS
    )


PLAN_QUESTION = {
    "id": "plan",
    "header": "Plan",
    "question": "Which plan fits you best?",
    "options": [
        {"label": "Basic", "description": "The cheapest one"},
        {"label": "Pro", "description": "The fastest one"},
    ],
}


def _queue_plan_question(upstream, prompt: str) -> None:
    upstream.queue(
        reply.tool_call("ask_user", {"questions": [PLAN_QUESTION]}, match=reply.answering(prompt)),
        reply.text("Pro it is.", match=reply.answering(prompt)),
    )


def _answer_pro(chat: ChatSurface) -> None:
    """Pick Pro on the model's question once it shows in `chat`."""
    question = chat.panel.get_by_text("Which plan fits you best?")
    expect(question, f"the {chat.place} never showed the model's question").to_be_visible(
        timeout=REPLY_TIMEOUT_MS
    )
    card = chat.panel.locator("section", has=chat.page.get_by_text("Which plan fits you best?"))
    card.get_by_role("button", name=re.compile(r"^Pro\b")).click()


def _expect_pro_reached_the_model(upstream, prompt: str) -> None:
    final = _requests_answering(upstream, prompt)[-1]
    conversation = reply.conversation(final)
    results = [str(entry["content"]) for entry in conversation if entry["role"] == "tool"]
    assert len(results) == 1, f"expected one answer to the question, the model got {results}"
    answered = json.loads(results[0])
    assert answered.get("answers", {}).get("plan", {}).get("label") == "Pro", (
        f"the model got another answer than the picked one: {answered}"
    )


def test_a_question_the_model_asks_in_the_chat_is_answered_next_to_an_open_note_chat(
    page_for, make_user, upstream
):
    author = make_user()
    text = "Guest list for Saturday."
    note_id = _create_note(author, _unique("Guests"), text)
    prompt = _unique("help me choose a plan")
    _queue_plan_question(upstream, prompt)
    page, _ = _open_note(page_for, author, note_id, text)
    open_note_chat(page)
    main = main_chat(page)
    main.show()

    main.send(prompt)

    _answer_pro(main)
    main.expect_reply("Pro it is.")
    _expect_pro_reached_the_model(upstream, prompt)


def test_a_question_the_model_asks_in_a_note_chat_is_answered_next_to_the_chat(
    page_for, make_user, upstream
):
    author = make_user()
    text = "Guest list for Sunday."
    note_id = _create_note(author, _unique("Guests"), text)
    prompt = _unique("help me choose a plan")
    _queue_plan_question(upstream, prompt)
    page = _note_opened_from_a_chat(page_for, author, note_id, text)
    open_note_chat(page)

    send_in_note_chat(page, prompt)

    note_chat = _note_tab_chat(page)
    _answer_pro(note_chat)
    note_chat.expect_reply("Pro it is.")
    _expect_pro_reached_the_model(upstream, prompt)


def test_the_regenerate_shortcut_in_the_chat_regenerates_the_chats_reply_not_the_note_chats(
    page_for, make_user, upstream
):
    """Red on dev c15f28649; fix PR open-webui/open-webui#34064 (3cab22263)."""
    author = make_user()
    text = "Route ideas for the weekend."
    note_id = _create_note(author, _unique("Routes"), text)
    main_question, note_question = _unique("Name a lake"), _unique("Name a pass")
    upstream.queue(
        reply.text("Lake Constance.", match=reply.answering(main_question)),
        reply.text("Lake Geneva.", match=reply.answering(main_question)),
        reply.text("The Brenner Pass.", match=reply.answering(note_question)),
    )
    page, _ = _open_note(page_for, author, note_id, text)
    open_note_chat(page)
    send_in_note_chat(page, note_question)
    note_chat = _note_tab_chat(page)
    note_chat.expect_reply("The Brenner Pass.")
    main = main_chat(page)
    main.show()
    main.send(main_question)
    main.expect_reply("Lake Constance.")

    main.input.click()
    page.keyboard.press("Control+R")

    main.expect_reply("Lake Geneva.")
    expect(main.conversation.get_by_text("2/2")).to_be_visible()
    asked_again = [
        body for body in upstream.chat_requests() if reply.answering(note_question)(body)
    ]
    assert len(asked_again) == 1, "the shortcut regenerated the note chat's reply"
