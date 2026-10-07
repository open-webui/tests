"""Journey: what a note offers beyond writing in it, and what a shared note looks like to others.

Asking the chat beside a note to enhance it runs the model with the note's id in its context;
the model rewrites the note through its note tool and the new text shows in the open editor and
is stored. A note shared read-only opens for the reader marked Read-Only Access with an editor
that takes no typing, a note shared for writing through a group takes the writer's typing, and
an account it was never shared with is sent away from it. A note downloads as plain text and as
Markdown with its title as the file name, the search on the Notes page keeps only the notes
whose title matches, and a note attached to a chat from the composer reaches the model. The
composer's note picker keeps a note shared read-only in its list when a name is typed into its
search (open-webui/open-webui#30968, issue #30967).

The writer test is red with Redis (`OWUI_TEST_REDIS=1`) on dev 0f5a58f5f: the server handles the
editor's live updates concurrently, so an update that arrives early can cancel the save of a later
one and store its older text last, and the note keeps missing the last characters typed
(open-webui/open-webui#31585, fix PR #31596 open).

Discriminates: passes on dev 176d31d1d; in a frontend copy, each test fails when its behaviour
is cut: the note editor ignoring the model's edit event, the editor staying editable for a reader,
the editor not leaving a note it cannot load, the downloads writing the HTML, the Notes page
search sending no query and the composer's note picker attaching nothing. In the a5bc78300
build with #30968 reverted, the shared note is gone from the picker once a name is typed. In a
backend copy, `GET /api/v1/notes/{id}` ignoring write grants turns the writer test red.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.access import grant, make_group
from harness.actors import Actor
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ENHANCE_PROMPT = "Enhance this note and update it."


def _unique(prefix: str) -> str:
    return f"{prefix} {uuid.uuid4().hex[:6]}"


def _create_note(owner: Actor, title: str, markdown: str, grants: list[dict] | None = None) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={
                "title": title,
                "data": {"content": {"md": markdown}},
                "access_grants": grants or [],
            },
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


def _note_card(page: Page, title: str) -> Locator:
    return page.get_by_role("main").get_by_role("button", name="Open note").filter(has_text=title)


# ---------------------------------------------------------------- the note's chat


def test_asking_the_note_chat_to_enhance_rewrites_the_open_note(page_for, make_user, upstream):
    author = make_user()
    note_id = _create_note(author, _unique("Draft"), "meeting tuesday bring slides")
    enhanced = "Meeting on Tuesday. Bring the slides."
    upstream.queue(
        reply.tool_call(
            "replace_note_content",
            {"note_id": note_id, "content": enhanced},
            match=reply.answering(ENHANCE_PROMPT),
        ),
        reply.text("The note is tidied up.", match=reply.answering(ENHANCE_PROMPT)),
    )
    page = page_for(author)
    editor = _open_note(page, note_id, "meeting tuesday bring slides")

    page.get_by_role("main").get_by_role("button", name="Chat", exact=True).click()
    page.get_by_role("button", name=ENHANCE_PROMPT).click()

    expect(page.get_by_text("The note is tidied up.")).to_be_visible()
    expect(editor).to_have_text(enhanced)
    assert _stored_markdown(author, note_id) == enhanced
    first_request = upstream.chat_requests()[0]
    system = " ".join(
        str(message.get("content"))
        for message in first_request["messages"]
        if message["role"] == "system"
    )
    assert f"Current note id: {note_id}" in system


# ---------------------------------------------------------------- shared notes


def test_a_reader_sees_the_note_read_only(page_for, make_user):
    owner, reader = make_user(), make_user()
    text = "the reader may only look"
    note_id = _create_note(owner, _unique("Shared"), text, [grant("user", reader.id, "read")])
    page = page_for(reader)

    editor = _open_note(page, note_id, text)

    expect(page.get_by_role("main").get_by_text("Read-Only Access")).to_be_visible()
    expect(editor).to_have_attribute("contenteditable", "false")
    editor.click()
    page.keyboard.type(" and nothing more")
    expect(editor).to_have_text(text)


def test_a_writer_in_a_shared_group_edits_the_note(page_for, make_user, admin):
    owner, writer = make_user(), make_user()
    group_id = make_group(admin, [writer])
    grants = [grant("group", group_id, "read"), grant("group", group_id, "write")]
    text = "packing list"
    note_id = _create_note(owner, _unique("Shared"), text, grants)
    page = page_for(writer)

    editor = _open_note(page, note_id, text)
    expect(page.get_by_role("main").get_by_text("Read-Only Access")).to_have_count(0)
    editor.click()
    page.keyboard.press("End")
    page.keyboard.type(" with sunscreen")

    _wait_until_stored(owner, note_id, "packing list with sunscreen")


def test_an_account_the_note_was_not_shared_with_is_sent_away(page_for, make_user):
    owner, stranger = make_user(), make_user()
    note_id = _create_note(owner, _unique("Private"), "for my eyes only")
    page = page_for(stranger)

    page.goto(f"/notes/{note_id}")

    expect(page).to_have_url(f"{stranger.base_url}/")
    expect(chat_input(page)).to_be_visible()
    expect(page.get_by_text("for my eyes only")).to_have_count(0)


# ---------------------------------------------------------------- download


@pytest.mark.parametrize(
    "format_name, extension",
    [("Plain text (.txt)", "txt"), ("Plain text (.md)", "md")],
    ids=["txt", "md"],
)
def test_a_note_downloads_as_its_markdown(page_for, make_user, format_name, extension):
    author = make_user()
    title = _unique("Recipe")
    markdown = "# Soup\n\n- leeks\n- **potatoes**"
    _create_note(author, title, markdown)
    page = page_for(author)
    page.goto("/notes")

    _note_card(page, title).get_by_role("button", name="Note Menu").first.click()
    page.get_by_role("menu").get_by_role("button", name="Download").hover()
    with page.expect_download() as download_info:
        page.get_by_role("button", name=format_name).click()

    download = download_info.value
    assert download.suggested_filename == f"{title}.{extension}"
    assert Path(download.path()).read_text() == markdown


# ---------------------------------------------------------------- search


def test_the_notes_search_keeps_only_matching_notes(page_for, make_user):
    author = make_user()
    heron, otter = _unique("Heron sightings"), _unique("Otter tracks")
    _create_note(author, heron, "by the reed bed")
    _create_note(author, otter, "on the muddy bank")
    page = page_for(author)
    page.goto("/notes")
    expect(_note_card(page, heron)).to_have_count(1)
    expect(_note_card(page, otter)).to_have_count(1)

    page.get_by_placeholder("Search Notes").fill("heron")

    expect(_note_card(page, otter)).to_have_count(0)
    expect(_note_card(page, heron)).to_have_count(1)


# ---------------------------------------------------------------- a note as chat context


def _open_note_picker(page: Page) -> Locator:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="More", exact=True).last.click()
    menu = page.get_by_role("menu")
    menu.get_by_role("button", name="Attach Notes").click()
    return menu


def test_a_note_attached_in_the_composer_reaches_the_model(page_for, make_user, upstream):
    author = make_user()
    title = _unique("Allergies")
    note_text = "Sam is allergic to hazelnuts"
    _create_note(author, title, note_text)
    question = f"what should I avoid baking? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Skip the hazelnuts.", match=reply.answering(question)))
    page = page_for(author)

    picker = _open_note_picker(page)
    picker.get_by_role("button", name=title).click()
    send(page, question)

    expect_reply(page, "Skip the hazelnuts.")
    [request] = [body for body in upstream.chat_requests() if reply.answering(question)(body)]
    sent = " ".join(str(message.get("content")) for message in request["messages"])
    assert note_text in sent, f"the note never reached the model: {sent}"


@pytest.mark.regression
def test_the_composer_note_picker_finds_a_note_shared_read_only_by_name(page_for, make_user):
    author, reader = make_user(), make_user()
    shared = _unique("Shared tide table")
    own = _unique("Own tide table")
    other = _unique("Unrelated recipe")
    _create_note(author, shared, "high at noon", [grant("user", reader.id, "read")])
    _create_note(reader, own, "low at dusk")
    _create_note(reader, other, "soup")
    picker = _open_note_picker(page_for(reader))
    expect(picker.get_by_role("button", name=shared)).to_be_visible()

    picker.get_by_placeholder("Search Notes").fill("tide table")

    expect(picker.get_by_role("button", name=other)).to_have_count(0)
    expect(picker.get_by_role("button", name=own)).to_be_visible()
    expect(
        picker.get_by_role("button", name=shared),
        "the read-only shared note vanished from the picker once a search was typed"
        " (open-webui/open-webui#30967)",
    ).to_be_visible()
