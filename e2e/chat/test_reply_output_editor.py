"""Journey: editing a reply's steps, a tool call and its result included, in the browser.

A reply made of steps (its text, its thoughts, the tools it called and what they returned) opens
in a step editor when its Edit button is pressed: one row per text, thought and tool call, each
with a Delete button, and a JSON editor holding the whole list behind a toggle. What is saved
there is what a reload shows and what the model is sent with the next question, so a person can
correct a wrong tool result, reword a thought or drop a tool call from the conversation. The JSON
editor refuses to switch back while its text is not a list of steps, and Cancel leaves the reply
as it was.

Each test seeds a chat whose reply looked up a tide table, then asks a follow-up and reads the
conversation the provider got.

Discriminates: passes on the dev 30f3f6a8f build; in a frontend build whose JSON editor no longer
takes its edits, whose Delete button removes nothing, whose toggle switches back over a JSON
error and whose Cancel saves the edit, the other tests fail; in one whose thought rows drop their
edits, the thought test fails.
"""

from __future__ import annotations

import json
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from utils.chat_ui import conversation, expect_reply, last_reply, send
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TOOL = "look_up_tide"
LOOKED_UP = '{"high_tide": "06:40"}'
CORRECTED = '{"high_tide": "07:15"}'
BEFORE_THE_TOOL = "Let me check the tide table."
AFTER_THE_TOOL = "High tide is in the early morning."


def _text_step(text: str) -> dict:
    return {
        "type": "message",
        "id": f"msg_{uuid.uuid4().hex[:8]}",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _steps(tool_result: str) -> list[dict]:
    return [
        _text_step(BEFORE_THE_TOOL),
        {
            "type": "function_call",
            "id": "fc_tide",
            "call_id": "call_tide",
            "name": TOOL,
            "arguments": '{"harbour": "Whitby"}',
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "id": "fc_tide_out",
            "call_id": "call_tide",
            "output": [{"type": "input_text", "text": tool_result}],
            "status": "completed",
        },
        _text_step(AFTER_THE_TOOL),
    ]


def _open_seeded_reply(page_for, make_user, steps: list[dict]) -> Page:
    owner = make_user()
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "When is high tide in Whitby?"},
                {"role": "assistant", "content": AFTER_THE_TOOL, "output": steps},
            ],
        )
    page = page_for(owner)
    page.goto(f"/c/{chat_id}")
    expect_reply(page, AFTER_THE_TOOL)
    return page


@pytest.fixture
def tide_chat(page_for, make_user) -> Page:
    return _open_seeded_reply(page_for, make_user, _steps(LOOKED_UP))


def _edit_reply(page: Page) -> Locator:
    last_reply(page).hover()
    conversation(page).get_by_role("button", name="Edit").last.click()
    save = last_reply(page).get_by_role("button", name="Save", exact=True)
    expect(save).to_be_visible()
    # the editor box: the nearest block holding both Save and the Visual/JSON toggle
    return save.locator(
        "xpath=ancestor::div[.//button[normalize-space()='Visual' or normalize-space()='JSON']][1]"
    )


def _open_json_editor(editing: Locator) -> Locator:
    tooltip_button(editing, "Switch to JSON editor").click()
    code = editing.get_by_role("textbox")
    expect(code).to_contain_text(TOOL)
    return code


def _replace_json(page: Page, code: Locator, text: str) -> None:
    code.click()
    page.keyboard.press("ControlOrMeta+A")
    page.keyboard.insert_text(text)


def _ask_follow_up(page: Page, upstream) -> dict:
    question = f"and tomorrow? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("A little later.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "A little later.")
    return next(
        body
        for body in reversed(upstream.chat_requests())
        if question in json.dumps(body["messages"])
    )


def _tool_results(request: dict) -> list[str]:
    return [str(entry["content"]) for entry in request["messages"] if entry["role"] == "tool"]


def test_a_tool_result_corrected_in_the_json_editor_is_what_the_model_gets_next(
    tide_chat, upstream
):
    editing = _edit_reply(tide_chat)
    code = _open_json_editor(editing)
    _replace_json(tide_chat, code, json.dumps(_steps(CORRECTED), indent=2))
    editing.get_by_role("button", name="Save", exact=True).click()
    expect(editing.get_by_role("button", name="Save", exact=True)).to_be_hidden()

    tide_chat.reload()
    expect_reply(tide_chat, AFTER_THE_TOOL)
    request = _ask_follow_up(tide_chat, upstream)

    results = _tool_results(request)
    assert any("07:15" in result for result in results), results
    assert not any("06:40" in result for result in results), results


def test_the_visual_editor_shows_the_corrected_result_after_the_json_editor(tide_chat):
    editing = _edit_reply(tide_chat)
    code = _open_json_editor(editing)
    _replace_json(tide_chat, code, json.dumps(_steps(CORRECTED), indent=2))

    tooltip_button(editing, "Switch to visual editor").click()

    expect(editing.get_by_text("07:15")).to_be_visible()
    expect(editing.get_by_text("06:40")).to_have_count(0)
    expect(editing.get_by_text(TOOL, exact=True)).to_be_visible()


def test_a_tool_call_deleted_in_the_editor_leaves_the_conversation(tide_chat, upstream):
    editing = _edit_reply(tide_chat)
    tool_row = editing.get_by_text(TOOL, exact=True).locator(
        "xpath=ancestor::div[.//button[@aria-label='Delete']][1]"
    )
    tool_row.hover()
    tool_row.get_by_role("button", name="Delete").click()
    expect(editing.get_by_text(TOOL, exact=True)).to_have_count(0)
    editing.get_by_role("button", name="Save", exact=True).click()

    tide_chat.reload()
    expect_reply(tide_chat, AFTER_THE_TOOL)
    expect(last_reply(tide_chat)).to_contain_text(BEFORE_THE_TOOL)
    expect(last_reply(tide_chat).get_by_text(f"View Result from {TOOL}")).to_have_count(0)
    request = _ask_follow_up(tide_chat, upstream)

    assert _tool_results(request) == []
    assert TOOL not in json.dumps(request["messages"])


def test_the_json_editor_names_a_broken_list_and_keeps_it_open(tide_chat):
    editing = _edit_reply(tide_chat)
    code = _open_json_editor(editing)

    _replace_json(tide_chat, code, '[{"type": "message"')
    expect(editing.get_by_text("Invalid JSON", exact=True)).to_be_visible()
    tooltip_button(editing, "Switch to visual editor").click()
    expect(code).to_be_visible()

    _replace_json(tide_chat, code, '{"type": "message"}')
    expect(editing.get_by_text("Must be a JSON array", exact=True)).to_be_visible()
    tooltip_button(editing, "Switch to visual editor").click()
    expect(code).to_be_visible()


def test_cancel_leaves_the_reply_as_it_was(tide_chat):
    editing = _edit_reply(tide_chat)
    editing.get_by_placeholder("Message text...").last.fill("High tide is at midnight.")
    editing.get_by_role("button", name="Cancel").click()
    expect(editing.get_by_role("button", name="Save", exact=True)).to_be_hidden()

    tide_chat.reload()
    expect_reply(tide_chat, AFTER_THE_TOOL)
    expect(last_reply(tide_chat).get_by_text("at midnight")).to_have_count(0)


def test_an_edited_thought_is_what_the_reply_shows_after_a_reload(page_for, make_user):
    thought = {
        "type": "reasoning",
        "id": "rs_tide",
        "status": "completed",
        "duration": 3,
        "summary": [{"type": "summary_text", "text": "The almanac says six forty."}],
    }
    page = _open_seeded_reply(page_for, make_user, [thought, _text_step(AFTER_THE_TOOL)])

    editing = _edit_reply(page)
    editing.get_by_placeholder("Reasoning text...").fill("The harbour master says seven fifteen.")
    editing.get_by_role("button", name="Save", exact=True).click()

    page.reload()
    expect_reply(page, AFTER_THE_TOOL)
    last_reply(page).get_by_text("Thought for 3 seconds").click()
    expect(last_reply(page)).to_contain_text("The harbour master says seven fifteen.")
    expect(last_reply(page).get_by_text("The almanac says six forty.")).to_have_count(0)
