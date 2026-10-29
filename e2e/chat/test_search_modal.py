"""The search dialog finds what an account may open, previews it and opens the one picked.

The dialog opens from the sidebar's Search entry or the search shortcut. Typing sends the query to
`GET /api/v1/search` (since b4db56ae9), which matches chat titles and message text, notes and
channel messages; a match in the text shows as a highlighted snippet. The highlighted result shows
in a preview pane beside the list, hovering or arrowing onto another result moves the preview
there, and Ctrl+Enter hides and shows the pane; Enter or the Open button opens it. "Start a new
conversation" in the Actions menu opens a new chat with the typed text in its message box, where
the person sends it (since 0ffd86967 the chat address only fills the box), and "Copy link" copies
the result's full address. Archived chats and another account's matching chat never appear, a
chat in a folder shared with the account, its note and a message in its channel do. The content
type menu keeps one type, the View filter the account's own or others' results and the Location
filter one folder, and more matches than one page of 24 load as the result list is scrolled to its
end. A channel message opens its thread scrolled to the matching reply.

Two tests pin fixed bugs: Enter on a highlighted chat closed the dialog without opening the chat
(#31003, fixed by PR #31004), and a new conversation started from the dialog dropped the text after
an `&` because the query went into the page address unencoded (#31469, fixed by PR #31592).

Twins: integration/chat/test_workspace_search.py and
integration/security/test_workspace_search_scope.py.

Discriminates: passes on dev f5a26f53a. In a build of it whose Enter opens nothing, whose ArrowUp
stays put, whose rows ignore the pointer, whose list never loads a second page, whose Location
filter is never sent, whose new conversation drops the query and whose snippet never highlights,
the Enter, arrow, hover, paging, Location, both new-conversation and highlight tests go red, and
the title, no-match and other-account tests stay green. In a build that puts the query into the
new chat's address unencoded the ampersand test goes red, and in a backend copy whose search
reaches every account's chats or includes archived ones the other-account test goes red.
On dev 948432b43, a backend copy that ignores `kind` turns the content type tests red, one that
ignores `sharing` the View tests, and one whose links carry no message turns the Copy link and
channel thread tests red; a build whose Ctrl+Enter does nothing turns the preview shortcut test
red and one whose thread ignores the linked message turns the channel thread test red.
"""

from __future__ import annotations

import re
import uuid

import httpx
import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.access import grant
from harness.channel_quotes import enable_channels, group_channel, post_message
from utils.chat_ui import chat_input, expect_reply
from utils.search_dialog import (
    answers_query,
    choose_content_type,
    choose_view,
    no_matches,
    open_actions,
    open_search,
    open_search_from_sidebar,
    open_selected,
    preview,
    result,
    results,
    search,
    search_box,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

LONG_AGO = 1_700_000_000


def unique_word() -> str:
    return f"quokka{uuid.uuid4().hex[:8]}"


def conversation(question: str, answer: str) -> dict:
    asked = {"id": "q", "parentId": None, "childrenIds": ["a"], "role": "user"}
    answered = {"id": "a", "parentId": "q", "childrenIds": [], "role": "assistant"}
    messages = {"q": {**asked, "content": question}, "a": {**answered, "content": answer}}
    return {"currentId": "a", "messages": messages}


def import_chat(
    client: httpx.Client,
    title: str,
    *,
    question: str = "hello",
    answer: str = "hi there",
    updated_at: int = LONG_AGO,
    **import_fields,
) -> str:
    """A stored chat with one question and answer, last changed at `updated_at`."""
    chat = {"title": title, "models": [], "history": conversation(question, answer)}
    entry = {"chat": chat, "created_at": updated_at, "updated_at": updated_at, **import_fields}
    imported = client.post("/api/v1/chats/import", json={"chats": [entry]})
    assert imported.status_code == 200, imported.text
    return imported.json()[0]["id"]


def post(client: httpx.Client, path: str, **body) -> dict:
    response = client.post(path, json=body or None)
    assert response.status_code == 200, response.text
    return response.json()


def type_query(dialog: Locator, text: str) -> None:
    """Types the way a person does, so the input's key handling runs as well."""
    search_box(dialog).click()
    with dialog.page.expect_response(answers_query(text)):
        search_box(dialog).press_sequentially(text)


def test_the_sidebar_entry_finds_a_chat_by_its_title_and_opens_it(page_for, make_user):
    account = make_user()
    word = unique_word()
    with account.client() as client:
        chat_id = import_chat(client, f"Trip {word}", answer="Pack a raincoat.")
        import_chat(client, "Weekly report")

    page = page_for(account)
    expect(chat_input(page)).to_be_visible()
    dialog = open_search_from_sidebar(page)
    search(dialog, word)

    expect(results(dialog)).to_have_count(1)
    expect(result(dialog, f"Trip {word}")).to_be_visible()
    result(dialog, f"Trip {word}").click()
    expect(preview(dialog)).to_contain_text("Pack a raincoat.")
    open_selected(dialog)

    expect(page).to_have_url(re.compile(f"/c/{chat_id}$"))
    expect(dialog).to_be_hidden()
    expect(page.get_by_text("Pack a raincoat.")).to_be_visible()


def test_the_shortcut_finds_a_chat_by_its_message_text_and_highlights_the_match(
    page_for, make_user
):
    account = make_user()
    word = unique_word()
    with account.client() as client:
        import_chat(client, "Untitled trip", question=f"Where does the {word} live?")

    page = page_for(account)
    dialog = open_search(page)
    search(dialog, word)

    found = result(dialog, "Untitled trip")
    expect(results(dialog)).to_have_count(1)
    expect(found).to_be_visible()
    expect(found).to_contain_text(f"Where does the {word} live?")
    expect(found.get_by_role("mark")).to_have_text(word)


@pytest.fixture
def two_matching_chats(make_user) -> tuple:
    """An account with two chats matching one word: the newer lists first."""
    account = make_user()
    word = unique_word()
    with account.client() as client:
        older_id = import_chat(client, f"Older {word}", answer="The older answer.")
        newer_id = import_chat(
            client, f"Newer {word}", answer="The newer answer.", updated_at=LONG_AGO + 100
        )
    return account, word, older_id, newer_id


def test_hovering_a_result_previews_its_conversation(page_for, two_matching_chats):
    account, word, _older_id, _newer_id = two_matching_chats
    page = page_for(account)
    dialog = open_search(page)
    search(dialog, word)
    expect(results(dialog)).to_have_count(2)
    expect(preview(dialog)).to_contain_text("The newer answer.")

    result(dialog, f"Older {word}").hover()

    expect(preview(dialog)).to_contain_text("The older answer.")
    expect(preview(dialog)).not_to_contain_text("The newer answer.")


def test_arrow_keys_move_the_preview_from_result_to_result(page_for, two_matching_chats):
    account, word, _older_id, _newer_id = two_matching_chats
    page = page_for(account)
    dialog = open_search(page)
    type_query(dialog, word)
    expect(results(dialog)).to_have_count(2)

    # the selection stops at the last result, however far past it the arrow goes
    for _ in range(6):
        page.keyboard.press("ArrowDown")
    expect(preview(dialog)).to_contain_text("The older answer.")

    page.keyboard.press("ArrowUp")
    expect(preview(dialog)).to_contain_text("The newer answer.")
    expect(preview(dialog)).not_to_contain_text("The older answer.")


@pytest.mark.regression
def test_enter_opens_the_highlighted_chat(page_for, two_matching_chats):
    account, word, _older_id, newer_id = two_matching_chats
    page = page_for(account)
    dialog = open_search(page)
    type_query(dialog, word)
    expect(results(dialog)).to_have_count(2)
    for _ in range(6):
        page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowUp")
    expect(preview(dialog)).to_contain_text("The newer answer.")

    page.keyboard.press("Enter")

    expect(page, "Enter closed the dialog without opening the chat (#31003)").to_have_url(
        re.compile(f"/c/{newer_id}$")
    )


def test_a_query_nothing_matches_says_no_results(page_for, make_user):
    account = make_user()
    with account.client() as client:
        import_chat(client, "Weekly report")

    page = page_for(account)
    dialog = open_search(page)
    expect(result(dialog, "Weekly report")).to_be_visible()
    search(dialog, unique_word())

    expect(no_matches(dialog)).to_be_visible()
    expect(results(dialog)).to_have_count(0)


def test_archived_and_other_accounts_matching_chats_never_show(page_for, make_user):
    account = make_user()
    word = unique_word()
    with account.client() as client:
        import_chat(client, f"Mine {word}")
        import_chat(client, f"Archived {word}", archived=True)
    with make_user().client() as other_client:
        import_chat(other_client, f"Theirs {word}", question=f"about {word}")

    page = page_for(account)
    dialog = open_search(page)
    search(dialog, word)

    expect(results(dialog)).to_have_count(1)
    expect(result(dialog, f"Mine {word}")).to_be_visible()


def test_the_location_filter_narrows_the_results_to_that_folder(page_for, make_user):
    account = make_user()
    word = unique_word()
    folder = f"Travel {word}"
    with account.client() as client:
        folder_id = post(client, "/api/v1/folders/", name=folder)["id"]
        import_chat(client, f"Filed {word}", folder_id=folder_id)
        import_chat(client, f"Loose {word}")

    page = page_for(account)
    dialog = open_search(page)
    search(dialog, word)
    expect(results(dialog)).to_have_count(2)
    dialog.get_by_role("button", name="Filters").click()
    page.get_by_role("button", name="Location").click()
    page.get_by_role("menuitem", name=folder).click()

    expect(dialog.get_by_role("button", name="Remove location filter")).to_have_text(folder)
    expect(results(dialog)).to_have_count(1)
    expect(result(dialog, f"Filed {word}")).to_be_visible()


def start_a_new_conversation(page: Page, text: str) -> None:
    dialog = open_search(page)
    search(dialog, text)
    open_actions(dialog).get_by_role("button", name="Start a new conversation").click()


def test_start_a_new_conversation_sends_the_typed_text(page_for, make_user, upstream):
    question = f"How far can a {unique_word()} hop?"
    upstream.queue(reply.text("About two metres.", match=reply.answering(question)))
    page = page_for(make_user())

    start_a_new_conversation(page, question)
    expect(chat_input(page)).to_have_text(question)
    chat_input(page).click()
    page.keyboard.press("Enter")

    expect_reply(page, "About two metres.")
    expect(page.get_by_label("Chat Conversation").get_by_text(question)).to_be_visible()


@pytest.mark.regression
def test_start_a_new_conversation_keeps_an_ampersand(page_for, make_user):
    question = f"Q&A about the {unique_word()}"
    page = page_for(make_user())

    start_a_new_conversation(page, question)

    expect(
        chat_input(page),
        "the text after the & was dropped from the new chat's message box (#31469)",
    ).to_have_text(question)


# --------------------------------------------------------------------------- paging

PAGE_SIZE = 24


def test_more_matches_than_a_page_load_as_the_results_are_scrolled(page_for, make_user):
    account = make_user()
    word = unique_word()
    with account.client() as client:
        entries = [
            {
                "chat": {"title": f"Log {index:03d} {word}", "models": []},
                "created_at": LONG_AGO + index,
                "updated_at": LONG_AGO + index,
            }
            for index in range(PAGE_SIZE + 5)
        ]
        imported = client.post("/api/v1/chats/import", json={"chats": entries})
        assert imported.status_code == 200, imported.text

    page = page_for(account)
    dialog = open_search(page)
    search(dialog, word)
    expect(results(dialog)).to_have_count(PAGE_SIZE)
    expect(result(dialog, f"Log 000 {word}")).to_have_count(0)

    results(dialog).last.scroll_into_view_if_needed()

    expect(results(dialog)).to_have_count(PAGE_SIZE + 5)
    expect(result(dialog, f"Log 000 {word}")).to_be_visible()


# --------------------------------------------------------------------------- the whole workspace


@pytest.fixture
def workspace(admin, preserve, make_user) -> tuple:
    """An account whose search word matches a chat in a folder shared with it, its own note and
    a message in its channel."""
    preserve("admin_config")
    enable_channels(admin)
    account, other = make_user(), make_user()
    word = unique_word()
    with other.client() as client:
        folder_id = post(client, "/api/v1/folders/", name=f"Allotment {word}")["id"]
        post(
            client,
            f"/api/v1/folders/{folder_id}/access/update",
            access_grants=[grant("user", account.id, "read")],
        )
        import_chat(client, f"Tools {word}", answer="Bring the spade.", folder_id=folder_id)
    with account.client() as client:
        post(
            client,
            "/api/v1/notes/create",
            title=f"Plot {word}",
            data={"content": {"md": "Beans by the fence."}},
        )
    channel_id = group_channel(account, other)
    channel_name = _channel_name(account, channel_id)
    post_message(other, channel_id, f"the {word} is ready to pick")
    return account, other, word, folder_id, channel_name


def _channel_name(account, channel_id: str) -> str:
    with account.client() as client:
        channel = client.get(f"/api/v1/channels/{channel_id}")
    assert channel.status_code == 200, channel.text
    return channel.json()["name"]


def test_a_shared_folder_chat_a_note_and_a_channel_message_all_show(page_for, workspace):
    account, other, word, _folder_id, channel_name = workspace
    page = page_for(account)
    dialog = open_search(page)
    search(dialog, word)

    expect(results(dialog)).to_have_count(3)
    expect(result(dialog, f"Tools {word}")).to_contain_text(f"Allotment {word}")
    expect(result(dialog, f"Plot {word}")).to_contain_text("Personal note")
    expect(result(dialog, other.name)).to_contain_text(f"#{channel_name}")

    result(dialog, f"Plot {word}").hover()
    expect(preview(dialog)).to_contain_text("Beans by the fence.")
    result(dialog, other.name).hover()
    expect(preview(dialog)).to_contain_text("ready to pick")
    result(dialog, f"Tools {word}").hover()
    expect(preview(dialog)).to_contain_text("Bring the spade.")


@pytest.mark.parametrize(
    "label, kind, shown",
    [("Chats", "chat", "Tools"), ("Notes", "note", "Plot"), ("Messages", "message", "")],
)
def test_the_content_type_keeps_only_that_type(page_for, workspace, label, kind, shown):
    account, other, word, _folder_id, _channel_name = workspace
    page = page_for(account)
    dialog = open_search(page)
    search(dialog, word)
    expect(results(dialog)).to_have_count(3)

    choose_content_type(dialog, label, kind)

    expect(results(dialog)).to_have_count(1)
    expect(results(dialog)).to_contain_text(f"{shown} {word}" if shown else other.name)
    expect(dialog.get_by_role("button", name=f"Content type: {label}")).to_be_visible()


@pytest.mark.parametrize(
    "label, sharing, shown",
    [("Created by you", "mine", ("Plot",)), ("Shared with you", "shared", ("Tools", ""))],
)
def test_the_view_filter_splits_your_own_from_what_was_shared(
    page_for, workspace, label, sharing, shown
):
    account, other, word, _folder_id, _channel_name = workspace
    page = page_for(account)
    dialog = open_search(page)
    search(dialog, word)
    expect(results(dialog)).to_have_count(3)

    choose_view(dialog, label, sharing)

    expect(dialog.get_by_role("button", name="Remove sharing filter")).to_have_text(label)
    expect(results(dialog)).to_have_count(len(shown))
    for title in shown:
        expect(result(dialog, f"{title} {word}" if title else other.name)).to_be_visible()


def test_ctrl_enter_hides_and_shows_the_preview(page_for, make_user):
    account = make_user()
    word = unique_word()
    with account.client() as client:
        import_chat(client, f"Trip {word}", answer="Pack a raincoat.")

    page = page_for(account)
    dialog = open_search(page)
    type_query(dialog, word)
    expect(results(dialog)).to_have_count(1)
    expect(preview(dialog)).to_contain_text("Pack a raincoat.")
    preview_button = dialog.get_by_role("button", name=re.compile(r"^Preview"))
    expect(preview_button).to_have_attribute("aria-pressed", "true")

    page.keyboard.press("Control+Enter")
    expect(preview(dialog)).to_be_hidden()
    expect(preview_button).to_have_attribute("aria-pressed", "false")
    expect(dialog).to_be_visible()

    page.keyboard.press("Control+Enter")
    expect(preview(dialog)).to_contain_text("Pack a raincoat.")


CLIPBOARD = ["clipboard-read", "clipboard-write"]


def test_copy_link_copies_the_results_address(page_for, make_user):
    account = make_user()
    word = unique_word()
    with account.client() as client:
        chat_id = import_chat(client, "Weather", answer=f"rain over the {word}")

    page = page_for(account, permissions=CLIPBOARD)
    dialog = open_search(page)
    search(dialog, word)
    expect(results(dialog)).to_have_count(1)

    open_actions(dialog).get_by_role("button", name="Copy link").click()

    expect(page.get_by_text("Link copied to clipboard")).to_be_visible()
    origin = page.evaluate("location.origin")
    assert page.evaluate("navigator.clipboard.readText()") == f"{origin}/c/{chat_id}?message=a"


THREAD_REPLIES = 60


def test_a_channel_result_opens_its_thread_at_the_matching_reply(
    page_for, admin, preserve, make_user
):
    preserve("admin_config")
    enable_channels(admin)
    author, member = make_user(), make_user()
    word = unique_word()
    channel_id = group_channel(author, member)
    parent_id = post_message(author, channel_id, "Who is bringing what on Saturday?")
    post_message(author, channel_id, "first", parent_id=parent_id)
    match = f"I will bring the {word}"
    reply_id = post_message(member, channel_id, match, parent_id=parent_id)
    # enough later replies that the thread opened at its newest end would not hold the match
    for index in range(THREAD_REPLIES):
        post_message(author, channel_id, f"later reply {index}", parent_id=parent_id)

    page = page_for(member)
    dialog = open_search(page)
    type_query(dialog, word)
    expect(results(dialog)).to_have_count(1)
    page.keyboard.press("Enter")

    expect(page).to_have_url(
        re.compile(rf"/channels/{channel_id}\?thread={parent_id}&message={reply_id}$")
    )
    expect(page.get_by_text(match, exact=True)).to_be_in_viewport()
