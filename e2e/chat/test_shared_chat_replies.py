"""Journey: a chat shared with "Allow replies", where the people it is shared with keep chatting.

The owner adds an account in the chat's Share dialog and sets Sharing mode to Allow replies. The
account opens the owner's live chat with a message box, its message and the reply show in the
owner's open tab without a reload, and each user message carries its author's name. A group
member loses the open chat as soon as an admin takes them out of the group. Setting the mode back
to Clone only closes the live chat for them, and the link then shows a frozen copy that holds
their message and leaves later ones out. A signed-in visitor who reaches the link only through
Open visibility gets that copy with no message box. The owner's chat settings stay the owner's:
the Controls of someone replying show none of the owner's system prompt, the model is not sent it
for their turn, and a clone made from the link starts without it. A folder shared with Allow
replies lets its readers reply in the owner's chats in it, taking them off the folder's Access
List closes the open chat, and only the folder's owner finds Access in its folder editor. Since
3a54306c5 someone replying in a shared chat gets no Voice mode button (the owner keeps theirs), and
a call they left open in a chat of their own closes when they go back to the shared chat.

Since v0.12.3 a folder reader who may reply regenerates the owner's reply and edits the owner's
question into a new branch (Send, no Save), while the owner's open tab keeps the question it
shows with the new branch one switch away. The people with the chat open see each other's
picture in the top bar (never their own), a picture opens that person's profile card, past three
people the rest sit in a Collaborators list whose entries open the card too, and the picture goes
once the other person leaves the chat.

Two tests are red on dev b5a20423e and name the bug. Opening an Allow replies link sends the
account to the home page, or leaves an empty new chat on the chat's address: the shared page
moves to the chat and then reports the share as not found, which sends it home. The other member
tests therefore open the address the link leads to. A member taken out of the group keeps the
chat and its message box on screen and only finds out on sending ("We could not find what you're
looking for"): the group change drops their socket before the chat access check runs, and the
chat page ignores the refused rejoin.

Discriminates: the other eight pass on dev b5a20423e. In a frontend build whose chat page ignores
the shared chat's socket events the live view, folder reply, Clone only and folder removal tests
fail; a build that labels every user message with the chat owner's name fails the live view and
folder reply tests; one with Access always in the folder editor fails the folder editor test. In
backend copies, the access check ignoring Allow replies shares fails the five tests that open a
shared live chat, no fresh copy on switching back fails the Clone only test, a live chat for any
signed-in visitor of the link fails the Open visibility test, the owner's settings left in for
others fails the system prompt and clone tests, and folder readers never allowed to reply fails
the folder reply and folder removal tests. A frontend build with both bugs fixed (the shared page
treating its move to the chat as done, the chat page reloading the chat when a rejoin is
refused) passes all ten. The live view test passes on dev c31423955 and turns red with 3a54306c5
reverted (Voice mode shows for the member) and in a build of it that only leaves out closing the
call (the call stays open over the shared chat).

The edit and presence tests pass on dev 948432b43. A frontend build whose chat page moves other
viewers onto an edited branch and keeps one's own picture in the presence list fails all three;
one showing Regenerate only on one's own replies fails the edit test; a backend copy whose
`publish_chat_presence` sends nothing fails both presence tests. The Clone test, run as a
control, passes on every mutant build.
"""

from __future__ import annotations

import contextlib
import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.access import grant, make_group
from harness.chat_history import seed_chat
from harness.socket_client import connected
from utils.chat_ui import chat_input, conversation, expect_reply, last_reply, send
from utils.member_picker import pick_member
from utils.sidebar_folders import click_folder_menu_button, folder_row
from utils.voice_call import start_call

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

QUESTION = "when does the ferry leave?"
ANSWER = "At half past nine."
OWNER_PROMPT = "You answer as the harbour master and never mention the toll."


def _accounts(make_user):
    suffix = uuid.uuid4().hex[:6]
    return make_user(name=f"Ada Owner {suffix}"), make_user(name=f"Ben Crew {suffix}")


def _start_chat(page: Page, upstream) -> str:
    upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
    send(page, QUESTION)
    expect_reply(page, ANSWER)
    expect(page).to_have_url(re.compile(r"/c/"))
    return page.url.rsplit("/", 1)[-1]


def _ask(page: Page, upstream, question: str, answer: str) -> dict:
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    send(page, question)
    expect_reply(page, answer)
    return next(filter(reply.answering(question), upstream.chat_requests()))


def _system_prompts(request: dict) -> list[str]:
    return [str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"]


def _user_message(page: Page, text: str) -> Locator:
    return conversation(page).locator(".user-message").filter(has_text=text)


def _share_dialog(page: Page) -> Locator:
    page.get_by_role("button", name="Chat actions").first.click()
    page.get_by_role("menu").get_by_role("menuitem", name="Share").click()
    return page.get_by_role("dialog").filter(has_text="Share Chat")


def _add_to_access_list(dialog: Locator, name: str) -> None:
    dialog.get_by_role("button", name="Add Access").click()
    picker = dialog.page.get_by_role("dialog").filter(has_text="Add Access").last
    picker.get_by_placeholder("Search").fill(name)
    pick_member(picker, name)
    with dialog.page.expect_response(re.compile(r"/access/update")):
        picker.get_by_role("button", name="Add", exact=True).click()


def _set_sharing_mode(dialog: Locator, label: str) -> None:
    mode = dialog.get_by_role("combobox", name=re.compile("Sharing mode"))
    with dialog.page.expect_response(re.compile(r"/access/update")):
        mode.select_option(label=label)


def _share_link(dialog: Locator) -> str:
    link = dialog.get_by_role("link", name="You have shared this chat before")
    expect(link).to_be_visible()
    return link.get_attribute("href")


def _share_over_api(owner, chat_id: str, grants: list[dict], mode: str | None) -> str:
    """Share the chat the way the dialog does: create the link, then save who may open it."""
    with owner.client() as client:
        shared = client.post(f"/api/v1/chats/{chat_id}/share", json={"share_mode": None})
        assert shared.status_code == 200, shared.text
        saved = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update",
            json={"access_grants": grants, "share_mode": mode},
        )
    assert saved.status_code == 200, saved.text
    return f"/s/{shared.json()['share_id']}"


def _set_chat_prompt(page: Page) -> None:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("navigation").get_by_role("button", name="Controls").click()
    page.get_by_role("textbox", name="Enter system prompt").fill(OWNER_PROMPT)


def _sent_home(page: Page, message: str | None = None) -> None:
    expect(page, message).to_have_url(re.compile(r"/$"))
    expect(conversation(page).get_by_text(ANSWER)).to_have_count(0)


def _open_live_chat(page: Page, chat_id: str) -> None:
    # the address the link leads to; the link itself sends people home (the next test)
    page.goto(f"/c/{chat_id}")
    expect(conversation(page).get_by_text(ANSWER)).to_be_visible()
    expect(chat_input(page)).to_be_visible()


def test_the_link_opens_the_owners_live_chat_straight_away(page_for, make_user, upstream):
    """Red on dev b5a20423e: the link sends the account home (open-webui/open-webui#32062, fixed by
    917f2fd1d)."""
    owner, member = _accounts(make_user)
    chat_id = _start_chat(page_for(owner), upstream)
    share_path = _share_over_api(owner, chat_id, [grant("user", member.id, "read")], "continue")

    member_page = page_for(member, path=share_path)

    expect(
        conversation(member_page).get_by_text(ANSWER),
        "the link sent the account home (or to an empty page) in place of the owner's chat",
    ).to_be_visible(timeout=10_000)
    expect(member_page).to_have_url(re.compile(f"/c/{chat_id}$"))
    expect(chat_input(member_page)).to_be_visible()


def test_a_user_allowed_to_reply_chats_on_and_the_owner_sees_it_live(
    page_for, voice_page_for, make_user, upstream
):
    owner, member = _accounts(make_user)
    owner_page = page_for(owner)
    chat_id = _start_chat(owner_page, upstream)
    dialog = _share_dialog(owner_page)
    _add_to_access_list(dialog, member.name)
    _set_sharing_mode(dialog, "Allow replies")
    _share_link(dialog)
    dialog.get_by_role("button", name="Close").click()

    # a microphone, for the call the member leaves open below
    member_page = voice_page_for(member)
    _open_live_chat(member_page, chat_id)
    expect(_user_message(member_page, QUESTION)).to_contain_text(owner.name)
    expect(member_page.get_by_role("button", name="Voice mode")).to_have_count(0)
    expect(owner_page.get_by_role("button", name="Voice mode")).to_be_visible()
    sent = _ask(member_page, upstream, "is there a later one?", "One more at noon.")

    assert [entry["content"] for entry in sent["messages"] if entry["role"] == "assistant"] == [
        ANSWER
    ]
    expect(_user_message(member_page, "is there a later one?")).not_to_contain_text(member.name)
    # the owner's tab was left open on the chat: no reload
    expect(_user_message(owner_page, "is there a later one?")).to_contain_text(member.name)
    expect(conversation(owner_page).get_by_text("One more at noon.")).to_be_visible()
    expect(_user_message(owner_page, QUESTION)).not_to_contain_text(owner.name)
    owner_page.reload()
    expect(_user_message(owner_page, "is there a later one?")).to_contain_text(member.name)

    member_page.get_by_role("link", name="New Chat").first.click()
    start_call(member_page)
    member_page.go_back()
    expect(conversation(member_page).get_by_text("One more at noon.")).to_be_visible()
    expect(member_page.get_by_role("button", name="End call")).to_have_count(0)


def test_a_group_member_loses_the_open_chat_when_taken_out_of_the_group(
    page_for, make_user, admin, upstream
):
    """Red on dev b5a20423e: the chat stays open on screen (open-webui/open-webui#32063, fixed by
    917f2fd1d). Red again on dev f5a26f53a, where 4dedaaf6e reloads the chat only when the rejoin
    is accepted; a build of it that reloads on every rejoin answer passes."""
    owner, member = _accounts(make_user)
    group_id = make_group(admin, [member])
    owner_page = page_for(owner)
    chat_id = _start_chat(owner_page, upstream)
    _share_over_api(owner, chat_id, [grant("group", group_id, "read")], "continue")

    member_page = page_for(member)
    _open_live_chat(member_page, chat_id)
    with admin.client() as client:
        removed = client.post(
            f"/api/v1/groups/id/{group_id}/users/remove", json={"user_ids": [member.id]}
        )
    assert removed.status_code == 200, removed.text

    _sent_home(member_page, "the removed member's page kept the chat and its message box open")


def test_setting_the_mode_back_to_clone_only_leaves_a_frozen_copy(page_for, make_user, upstream):
    owner, member = _accounts(make_user)
    owner_page = page_for(owner)
    chat_id = _start_chat(owner_page, upstream)
    share_path = _share_over_api(owner, chat_id, [grant("user", member.id, "read")], "continue")
    member_page = page_for(member)
    _open_live_chat(member_page, chat_id)
    _ask(member_page, upstream, "can bikes go on board?", "Bikes ride free.")

    owner_page.reload()
    dialog = _share_dialog(owner_page)
    _set_sharing_mode(dialog, "Clone only")
    dialog.get_by_role("button", name="Close").click()
    _sent_home(member_page)
    _ask(owner_page, upstream, "and dogs?", "Dogs on a lead.")

    member_page.goto(share_path)
    expect(member_page).to_have_url(re.compile(re.escape(share_path) + "$"))
    expect(conversation(member_page).get_by_text("Bikes ride free.")).to_be_visible()
    expect(member_page.get_by_role("button", name="Clone Chat")).to_be_visible()
    expect(chat_input(member_page)).to_have_count(0)
    expect(conversation(member_page).get_by_text("Dogs on a lead.")).to_have_count(0)


def test_a_visitor_through_open_visibility_reads_a_copy_without_a_message_box(
    page_for, make_user, admin, upstream
):
    owner, member = _accounts(make_user)
    make_group(admin, [owner], {"sharing": {"open_chats": True}})
    owner_page = page_for(owner)
    chat_id = _start_chat(owner_page, upstream)
    grants = [grant("user", member.id, "read"), grant("anyone", "*", "read")]
    share_path = _share_over_api(owner, chat_id, grants, "continue")
    _ask(owner_page, upstream, "is the cafe open?", "Until six.")

    visitor_page = page_for(make_user(), path=share_path)
    expect(conversation(visitor_page).get_by_text(ANSWER)).to_be_visible()
    expect(_user_message(visitor_page, QUESTION)).to_contain_text(owner.name)
    expect(visitor_page).to_have_url(re.compile(re.escape(share_path) + "$"))
    expect(visitor_page.get_by_role("button", name="Clone Chat")).to_be_visible()
    expect(chat_input(visitor_page)).to_have_count(0)
    expect(conversation(visitor_page).get_by_text("Until six.")).to_have_count(0)

    member_page = page_for(member)
    _open_live_chat(member_page, chat_id)
    expect(conversation(member_page).get_by_text("Until six.")).to_be_visible()


def test_someone_replying_does_not_get_the_owners_system_prompt(page_for, make_user, upstream):
    owner, member = _accounts(make_user)
    owner_page = page_for(owner)
    _set_chat_prompt(owner_page)
    chat_id = _start_chat(owner_page, upstream)
    assert _system_prompts(upstream.chat_requests()[-1]) == [OWNER_PROMPT]
    _share_over_api(owner, chat_id, [grant("user", member.id, "read")], "continue")

    member_page = page_for(member)
    _open_live_chat(member_page, chat_id)
    member_page.get_by_role("navigation").get_by_role("button", name="Controls").click()
    expect(member_page.get_by_role("textbox", name="Enter system prompt")).to_have_value("")
    member_turn = _ask(member_page, upstream, "what is the toll?", "Ask at the office.")
    assert OWNER_PROMPT not in "\n".join(_system_prompts(member_turn))

    owner_page.reload()
    owner_turn = _ask(owner_page, upstream, "any delays today?", "None so far.")
    assert _system_prompts(owner_turn) == [OWNER_PROMPT]


def test_a_clone_from_the_link_starts_without_the_owners_system_prompt(
    page_for, make_user, upstream
):
    owner, member = _accounts(make_user)
    owner_page = page_for(owner)
    _set_chat_prompt(owner_page)
    chat_id = _start_chat(owner_page, upstream)
    share_path = _share_over_api(owner, chat_id, [grant("user", member.id, "read")], None)

    member_page = page_for(member, path=share_path)
    member_page.get_by_role("button", name="Clone Chat").click()
    expect(member_page).to_have_url(re.compile(r"/c/"))
    expect(member_page).not_to_have_url(re.compile(chat_id))
    expect(conversation(member_page).get_by_text(ANSWER)).to_be_visible()
    clone_turn = _ask(member_page, upstream, "what is the toll?", "Two euros.")

    assert OWNER_PROMPT not in "\n".join(_system_prompts(clone_turn))
    assert [
        entry["content"] for entry in clone_turn["messages"] if entry["role"] == "assistant"
    ] == [ANSWER]


def _chat_header(page: Page) -> Locator:
    return page.get_by_role("navigation").filter(
        has=page.get_by_role("button", name="Controls", exact=True)
    )


def _present(page: Page, person) -> Locator:
    """The picture of someone with the chat open, in the chat's top bar."""
    return _chat_header(page).get_by_role("button", name=person.name, exact=True)


def _profile_card(page: Page) -> Locator:
    return page.locator("[data-link-preview-content]")


def test_people_with_the_chat_open_see_each_others_picture_until_one_leaves(
    page_for, make_user, upstream
):
    owner, member = _accounts(make_user)
    owner_page = page_for(owner)
    chat_id = _start_chat(owner_page, upstream)
    _share_over_api(owner, chat_id, [grant("user", member.id, "read")], "continue")
    member_page = page_for(member)
    _open_live_chat(member_page, chat_id)

    expect(_present(owner_page, member)).to_be_visible()
    expect(_present(member_page, owner)).to_be_visible()
    expect(_present(owner_page, owner)).to_have_count(0)
    expect(_present(member_page, member)).to_have_count(0)

    _present(owner_page, member).click()
    expect(_profile_card(owner_page).get_by_title(member.name)).to_be_visible()

    member_page.get_by_role("link", name="New Chat").first.click()
    expect(chat_input(member_page)).to_be_visible()
    expect(_present(owner_page, member)).to_have_count(0)
    expect(_present(member_page, owner)).to_have_count(0)


def test_the_collaborators_list_names_everyone_past_three_and_opens_their_card(
    page_for, make_user, upstream
):
    owner, _ = _accounts(make_user)
    crew = [make_user(name=f"Deckhand {n} {uuid.uuid4().hex[:6]}") for n in range(4)]
    owner_page = page_for(owner)
    chat_id = _start_chat(owner_page, upstream)
    _share_over_api(owner, chat_id, [grant("user", c.id, "read") for c in crew], "continue")

    with contextlib.ExitStack() as tabs:
        for person in crew:
            tab = tabs.enter_context(connected(person))
            joined = tab.client.call(
                "events:chat", {"chat_id": chat_id, "data": {"type": "join"}}, timeout=30
            )
            assert joined is True, f"{person.name} could not open the chat"

        listed = _chat_header(owner_page).get_by_role("button", name="Collaborators")
        expect(listed).to_have_text("+1")
        listed.click()
        collaborators = owner_page.get_by_role("dialog", name="Collaborators")
        for person in crew:
            expect(
                collaborators.get_by_role("button", name=person.name, exact=True)
            ).to_be_visible()
        expect(collaborators.get_by_role("button", name=owner.name, exact=True)).to_have_count(0)

        collaborators.get_by_role("button", name=crew[-1].name, exact=True).click()
        expect(_profile_card(owner_page).get_by_title(crew[-1].name)).to_be_visible()


# ------------------------------------------------------------------ folders shared with replies


FOLDER_SHARING = {"sharing": {"folders": True}}


@pytest.fixture
def crew(make_user, admin):
    """An owner and a member in one group allowed to share folders, and the group's name."""
    owner, member = _accounts(make_user)
    group_id = make_group(admin, [owner, member], FOLDER_SHARING)
    with admin.client() as client:
        group_name = client.get(f"/api/v1/groups/id/{group_id}").json()["name"]
    return owner, member, group_id, group_name


def _folder_with_chat(owner) -> tuple[str, str, str]:
    """A folder of the owner's, expanded in the sidebar, holding one answered chat."""
    name = f"Harbour {uuid.uuid4().hex[:6]}"
    with owner.client() as client:
        created = client.post("/api/v1/folders/", json={"name": name})
        assert created.status_code == 200, created.text
        folder_id = created.json()["id"]
        client.post(f"/api/v1/folders/{folder_id}/update/expanded", json={"is_expanded": True})
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": QUESTION}, {"role": "assistant", "content": ANSWER}],
        )
        moved = client.post(f"/api/v1/chats/{chat_id}/folder", json={"folder_id": folder_id})
        assert moved.status_code == 200, moved.text
    return folder_id, name, chat_id


def _share_folder_over_api(owner, folder_id: str, grants: list[dict], mode: str | None) -> None:
    with owner.client() as client:
        saved = client.post(
            f"/api/v1/folders/{folder_id}/access/update",
            json={"access_grants": grants, "share_mode": mode},
        )
    assert saved.status_code == 200, saved.text


def _open_sidebar(page: Page) -> Locator:
    expect(chat_input(page)).to_be_visible()
    opener = page.get_by_role("button", name="Open Sidebar", exact=True)
    if opener.is_visible():
        opener.click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    folders = sidebar.get_by_role("button", name="Folders", exact=True)
    expect(folders).to_be_visible()
    if folders.get_attribute("aria-expanded") != "true":
        folders.click()
    return sidebar


def _folder_menu(page: Page, name: str) -> Locator:
    row = folder_row(_open_sidebar(page), name)
    click_folder_menu_button(row)
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("menuitem", name="Export")).to_be_visible()
    return menu


def _folder_page_menu(page: Page, folder_id: str) -> Locator:
    page.goto(f"/folders/{folder_id}")
    page.get_by_label("Folder options").click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("menuitem", name="Export")).to_be_visible()
    return menu


def _folder_editor(page: Page, menu: Locator, name: str) -> Locator:
    menu.get_by_role("menuitem", name="Edit").click()
    editor = page.get_by_role("dialog").filter(has_text="Edit Folder")
    # the folder loads after the dialog opens, and with it the Access button
    expect(editor.get_by_placeholder("Enter folder name")).to_have_value(name)
    return editor


def _folder_share_dialog(page: Page, name: str) -> Locator:
    editor = _folder_editor(page, _folder_menu(page, name), name)
    editor.get_by_role("button", name="Access", exact=True).click()
    dialog = page.get_by_role("dialog").filter(has_text=f"Access: {name}")
    expect(dialog.get_by_role("button", name="Add Access")).to_be_visible()
    return dialog


def test_a_reader_of_a_folder_shared_with_replies_answers_in_the_owners_chat(
    page_for, crew, upstream
):
    owner, member, group_id, group_name = crew
    folder_id, name, chat_id = _folder_with_chat(owner)
    owner_page = page_for(owner)
    dialog = _folder_share_dialog(owner_page, name)
    _add_to_access_list(dialog, group_name)
    dialog.get_by_role("combobox", name="Access level").select_option("read")
    _set_sharing_mode(dialog, "Allow replies")
    owner_page.goto(f"/c/{chat_id}")
    expect_reply(owner_page, ANSWER)

    member_page = page_for(member, path=f"/c/{chat_id}")
    expect(conversation(member_page).get_by_text(ANSWER)).to_be_visible()
    expect(chat_input(member_page)).to_be_visible()
    _ask(member_page, upstream, "where do I buy a ticket?", "At the kiosk.")

    expect(_user_message(owner_page, "where do I buy a ticket?")).to_contain_text(member.name)
    expect(conversation(owner_page).get_by_text("At the kiosk.")).to_be_visible()


def test_a_reader_taken_off_the_folders_access_list_loses_the_open_chat(page_for, crew):
    owner, member, group_id, group_name = crew
    folder_id, name, chat_id = _folder_with_chat(owner)
    _share_folder_over_api(owner, folder_id, [grant("group", group_id, "read")], "continue")
    member_page = page_for(member, path=f"/c/{chat_id}")
    expect(conversation(member_page).get_by_text(ANSWER)).to_be_visible()
    expect(chat_input(member_page)).to_be_visible()

    owner_page = page_for(owner)
    dialog = _folder_share_dialog(owner_page, name)
    group_entry = dialog.get_by_text(group_name).locator("xpath=ancestor::div[.//select][1]")
    with owner_page.expect_response(re.compile(r"/access/update")):
        group_entry.get_by_role("button").last.click()

    _sent_home(member_page)


def _first_question_id(owner, chat_id: str) -> str:
    with owner.client() as client:
        stored = client.get(f"/api/v1/chats/{chat_id}")
    assert stored.status_code == 200, stored.text
    messages = stored.json()["chat"]["history"]["messages"].values()
    return next(m["id"] for m in messages if m["role"] == "user" and m["parentId"] is None)


def test_a_folder_reader_regenerates_and_edits_the_owners_turn_while_the_owner_keeps_theirs(
    page_for, crew, upstream
):
    owner, member, group_id, _ = crew
    folder_id, _, chat_id = _folder_with_chat(owner)
    _share_folder_over_api(owner, folder_id, [grant("group", group_id, "read")], "continue")
    owner_page = page_for(owner, path=f"/c/{chat_id}")
    expect_reply(owner_page, ANSWER)
    member_page = page_for(member, path=f"/c/{chat_id}")
    expect_reply(member_page, ANSWER)

    upstream.queue(reply.text("At ten past nine.", match=reply.answering(QUESTION)))
    last_reply(member_page).hover()
    conversation(member_page).get_by_role("button", name="Regenerate").last.click()
    member_page.get_by_text("Try Again").click()
    expect_reply(member_page, "At ten past nine.")
    expect(conversation(member_page).get_by_text("2/2")).to_be_visible()

    edited = "when does the last ferry leave?"
    upstream.queue(reply.text("At midnight.", match=reply.answering(edited)))
    question_id = _first_question_id(owner, chat_id)
    question = member_page.locator(f"#message-{question_id}")
    question.hover()
    question.get_by_role("button", name="Edit").click()
    # the question is the owner's: the member branches off it and cannot save over it
    expect(question.get_by_role("button", name="Save", exact=True)).to_have_count(0)
    question.locator("textarea").fill(edited)
    question.get_by_role("button", name="Send").click()
    expect_reply(member_page, "At midnight.")

    # the owner's open tab keeps the question it shows, with the new branch beside it
    owner_question = owner_page.locator(f"#message-{question_id}")
    expect(owner_question.get_by_text("1/2")).to_be_visible()
    expect(owner_question).to_contain_text(QUESTION)
    expect(conversation(owner_page).get_by_text(edited)).to_have_count(0)
    owner_question.hover()
    owner_question.get_by_role("button", name="Next message").click()
    expect(_user_message(owner_page, edited)).to_contain_text(member.name)
    expect_reply(owner_page, "At midnight.")


def test_only_the_folders_owner_finds_access_editing_it(page_for, crew):
    owner, member, group_id, _ = crew
    folder_id, name, _ = _folder_with_chat(owner)
    _share_folder_over_api(
        owner,
        folder_id,
        [grant("group", group_id, "read"), grant("group", group_id, "write")],
        "continue",
    )

    member_page = page_for(member)
    member_editor = _folder_editor(member_page, _folder_page_menu(member_page, folder_id), name)
    expect(member_editor.get_by_role("button", name="Access", exact=True)).to_have_count(0)
    owner_page = page_for(owner)
    owner_editor = _folder_editor(owner_page, _folder_page_menu(owner_page, folder_id), name)
    expect(owner_editor.get_by_role("button", name="Access", exact=True)).to_be_visible()
