"""Journey: attaching to a message by dropping, pasting and removing files in the composer.

A text file dropped on the message box is attached and the model can read its text; one removed
from the composer before sending never reaches it. A chat dragged from the sidebar onto the
composer is attached as a reference the model can open. A picture pasted into the message box is
sent to the model as an image, and with Paste Large Text as File on, a long paste becomes an
attached file instead of message text. Each is read in what the model was sent; since bd6fc15be
File Context is off with native function calling, so the model reads a file or a chat through its
tools and the scripted model calls them.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose composer ignores dropped files,
dropped sidebar items and pasted files, and whose remove button leaves the file attached, every
test but the long paste one fails, and with Paste Large Text as File ignored that one fails.

Retargeted for 3b1a95383: only the composer takes dropped files and chats; the rest of the chat
area ignores dropped files and can open a dropped chat beside it.
"""

from __future__ import annotations

import base64
import json
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from harness.upstream import Reply
from utils.cached_chat import this_turns_tool_results
from utils.chat_ui import chat_input, expect_reply, send
from utils.clipboard import on_clipboard, write_text

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

# a 2x2 red PNG
RED_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP4z8AARAwQCgAf7gP9i18U1AAAAABJRU5E"
    "rkJggg=="
)


def _question() -> str:
    return f"what does the attachment say? {uuid.uuid4().hex[:6]}"


def _request_answering(upstream, question: str) -> dict:
    [request] = [body for body in upstream.chat_requests() if reply.answering(question)(body)]
    return request


def _read_back(upstream, question: str) -> str:
    """What the model's own tool call returned while answering `question`."""
    *_, answered = [body for body in upstream.chat_requests() if reply.answering(question)(body)]
    return json.dumps(this_turns_tool_results(answered))


def _ask(page: Page, upstream, question: str, *reading: Reply) -> None:
    answering = reply.answering(question)
    for call in reading:
        call.match = answering
    upstream.queue(*reading, reply.text("Noted.", match=answering))
    send(page, question)
    expect_reply(page, "Noted.")


def _searching_files(pattern: str) -> Reply:
    return reply.tool_call("grep_chat_files", {"pattern": pattern})


def _attach_by_upload(page: Page, name: str, text: str) -> None:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="More", exact=True).last.click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("menuitem", name="Upload Files").click()
    chooser.value.set_files({"name": name, "mimeType": "text/plain", "buffer": text.encode()})


def _drop(page: Page, script: str, *args) -> None:
    """Drop what `script` puts into a DataTransfer onto the message box, as a drag does."""
    box = chat_input(page)
    expect(box).to_be_visible()
    data = page.evaluate_handle(script, list(args))
    box.dispatch_event("dragover", {"dataTransfer": data})
    box.dispatch_event("drop", {"dataTransfer": data})


def test_a_file_removed_before_sending_never_reaches_the_model(page_for, make_user, upstream):
    page = page_for(make_user())
    _attach_by_upload(page, "kept.txt", "The kept file mentions lighthouses.")
    _attach_by_upload(page, "removed.txt", "The removed file mentions submarines.")
    removed = page.get_by_role("button").filter(has_text="removed.txt")
    expect(removed).to_be_visible()
    removed.hover()
    removed.locator("xpath=..").get_by_role("button", name="Remove File").click()
    expect(page.get_by_text("removed.txt")).to_have_count(0)

    question = _question()
    _ask(page, upstream, question, _searching_files("file mentions"))
    found = _read_back(upstream, question)
    assert "lighthouses" in found, found
    assert "submarines" not in found, found


def test_a_file_dropped_on_the_chat_is_attached_and_reaches_the_model(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    _drop(
        page,
        """([name, text]) => {
            const data = new DataTransfer();
            data.items.add(new File([text], name, { type: 'text/plain' }));
            return data;
        }""",
        "dropped.txt",
        "The dropped file mentions kingfishers.",
    )
    expect(page.get_by_role("button").filter(has_text="dropped.txt")).to_be_visible()

    question = _question()
    _ask(page, upstream, question, _searching_files("dropped file"))
    assert "kingfishers" in _read_back(upstream, question)


def test_a_sidebar_chat_dropped_on_the_composer_is_referenced(page_for, make_user, upstream):
    account = make_user()
    with account.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "Where did we park?"},
                {"role": "assistant", "content": "Level four, bay twelve."},
            ],
        )
        renamed = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": "Parking"}})
        assert renamed.status_code == 200, renamed.text
    page = page_for(account)
    _drop(
        page,
        """([chatId]) => {
            const data = new DataTransfer();
            data.setData('application/x-open-webui-drag', '');
            data.setData('application/x-open-webui-chat', chatId);
            data.setData('text/plain', JSON.stringify({ type: 'chat', id: chatId }));
            return data;
        }""",
        chat_id,
    )
    expect(page.get_by_role("button").filter(has_text="Parking")).to_be_visible()

    question = _question()
    _ask(page, upstream, question, reply.tool_call("view_chat", {"chat_id": chat_id}))
    asked = next(filter(reply.answering(question), upstream.chat_requests()))
    assert f'type="chat" id="{chat_id}"' in str(reply.conversation(asked)[-1]["content"])
    assert "bay twelve" in _read_back(upstream, question)


def test_a_pasted_picture_is_sent_to_the_model_as_an_image(page_for, make_user, upstream):
    page = page_for(make_user(), permissions=["clipboard-read", "clipboard-write"])
    expect(chat_input(page)).to_be_visible()
    on_clipboard(
        page,
        """async (encoded) => {
            const bytes = Uint8Array.from(atob(encoded), (c) => c.charCodeAt(0));
            const picture = new Blob([bytes], { type: 'image/png' });
            await navigator.clipboard.write([new ClipboardItem({ 'image/png': picture })]);
        }""",
        base64.b64encode(RED_PNG).decode(),
    )
    chat_input(page).click()
    page.keyboard.press("Control+V")
    expect(page.get_by_role("button", name="Show image preview")).to_be_visible()

    question = _question()
    _ask(page, upstream, question)
    content = reply.conversation(_request_answering(upstream, question))[-1]["content"]
    images = [part for part in content if part.get("type") == "image_url"]
    assert len(images) == 1, content
    # the clipboard re-encodes the picture, so its header is compared and not its bytes
    sent_picture = base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1])
    assert sent_picture[:24] == RED_PNG[:24]


def test_a_long_paste_becomes_an_attached_file_when_the_account_asks(page_for, make_user, upstream):
    account = make_user()
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"largeTextAsFile": True}}
        )
        assert saved.status_code == 200, saved.text
    page = page_for(account, permissions=["clipboard-read", "clipboard-write"])
    long_text = "The tide table for the harbour. " * 100
    expect(chat_input(page)).to_be_visible()
    write_text(page, long_text)
    chat_input(page).click()
    page.keyboard.press("Control+V")

    expect(page.get_by_role("button").filter(has_text="Pasted_Text_")).to_be_visible()
    expect(chat_input(page)).not_to_contain_text("tide table")
    question = _question()
    _ask(page, upstream, question, _searching_files("for the harbour"))
    assert "tide table" in _read_back(upstream, question)
