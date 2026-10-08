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
List closes the open chat, and only the folder's owner finds Share in its menu.

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
folder reply tests; one with Share always in a folder page's menu fails the folder menu test. In
backend copies, the access check ignoring Allow replies shares fails the five tests that open a
shared live chat, no fresh copy on switching back fails the Clone only test, a live chat for any
signed-in visitor of the link fails the Open visibility test, the owner's settings left in for
others fails the system prompt and clone tests, and folder readers never allowed to reply fails
the folder reply and folder removal tests. A frontend build with both bugs fixed (the shared page
treating its move to the chat as done, the chat page reloading the chat when a rejoin is
refused) passes all ten.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.access import grant, make_group
from harness.chat_history import seed_chat
from utils.chat_ui import chat_input, conversation, expect_reply, send

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
    page.get_by_role("menu").get_by_role("button", name="Share").click()
    return page.get_by_role("dialog").filter(has_text="Share Chat")


def _add_to_access_list(dialog: Locator, name: str) -> None:
    dialog.get_by_role("button", name="Add Access").click()
    picker = dialog.page.get_by_role("dialog").filter(has_text="Add Access").last
    picker.get_by_placeholder("Search").fill(name)
    picker.get_by_role("button", name=name).last.click()
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
    """Red on dev b5a20423e: the link sends the account home (open-webui/open-webui#32062)."""
    owner, member = _accounts(make_user)
    chat_id = _start_chat(page_for(owner), upstream)
    share_path = _share_over_api(owner, chat_id, [grant("user", member.id, "read")], "continue")

    member_page = page_for(member)
    member_page.goto(share_path)

    expect(
        conversation(member_page).get_by_text(ANSWER),
        "the link sent the account home (or to an empty page) in place of the owner's chat",
    ).to_be_visible(timeout=10_000)
    expect(member_page).to_have_url(re.compile(f"/c/{chat_id}$"))
    expect(chat_input(member_page)).to_be_visible()


def test_a_user_allowed_to_reply_chats_on_and_the_owner_sees_it_live(page_for, make_user, upstream):
    owner, member = _accounts(make_user)
    owner_page = page_for(owner)
    chat_id = _start_chat(owner_page, upstream)
    dialog = _share_dialog(owner_page)
    _add_to_access_list(dialog, member.name)
    _set_sharing_mode(dialog, "Allow replies")
    _share_link(dialog)
    dialog.get_by_role("button", name="Close").click()

    member_page = page_for(member)
    _open_live_chat(member_page, chat_id)
    expect(_user_message(member_page, QUESTION)).to_contain_text(owner.name)
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


def test_a_group_member_loses_the_open_chat_when_taken_out_of_the_group(
    page_for, make_user, admin, upstream
):
    """Red on dev b5a20423e: the chat stays open on screen (open-webui/open-webui#32063)."""
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

    visitor_page = page_for(make_user())
    visitor_page.goto(share_path)
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

    member_page = page_for(member)
    member_page.goto(share_path)
    member_page.get_by_role("button", name="Clone Chat").click()
    expect(member_page).to_have_url(re.compile(r"/c/"))
    expect(member_page).not_to_have_url(re.compile(chat_id))
    expect(conversation(member_page).get_by_text(ANSWER)).to_be_visible()
    clone_turn = _ask(member_page, upstream, "what is the toll?", "Two euros.")

    assert OWNER_PROMPT not in "\n".join(_system_prompts(clone_turn))
    assert [
        entry["content"] for entry in clone_turn["messages"] if entry["role"] == "assistant"
    ] == [ANSWER]


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


def _folder_row(sidebar: Locator, name: str) -> Locator:
    return sidebar.get_by_role("button", name=re.compile(f"^{re.escape(name)}"))


def _folder_menu(page: Page, name: str) -> Locator:
    row = _folder_row(_open_sidebar(page), name)
    row.hover()
    row.get_by_role("button").last.click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("button", name="Export")).to_be_visible()
    return menu


def _folder_page_menu(page: Page, folder_id: str) -> Locator:
    page.goto(f"/folders/{folder_id}")
    page.get_by_label("Folder options").click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("button", name="Export")).to_be_visible()
    return menu


def _folder_share_dialog(page: Page, name: str) -> Locator:
    _folder_menu(page, name).get_by_role("button", name="Share").click()
    dialog = page.get_by_role("dialog").filter(has_text=f"Share: {name}")
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

    member_page = page_for(member)
    member_page.goto(f"/c/{chat_id}")
    expect(conversation(member_page).get_by_text(ANSWER)).to_be_visible()
    expect(chat_input(member_page)).to_be_visible()
    _ask(member_page, upstream, "where do I buy a ticket?", "At the kiosk.")

    expect(_user_message(owner_page, "where do I buy a ticket?")).to_contain_text(member.name)
    expect(conversation(owner_page).get_by_text("At the kiosk.")).to_be_visible()


def test_a_reader_taken_off_the_folders_access_list_loses_the_open_chat(page_for, crew):
    owner, member, group_id, group_name = crew
    folder_id, name, chat_id = _folder_with_chat(owner)
    _share_folder_over_api(owner, folder_id, [grant("group", group_id, "read")], "continue")
    member_page = page_for(member)
    member_page.goto(f"/c/{chat_id}")
    expect(conversation(member_page).get_by_text(ANSWER)).to_be_visible()
    expect(chat_input(member_page)).to_be_visible()

    owner_page = page_for(owner)
    dialog = _folder_share_dialog(owner_page, name)
    group_entry = dialog.get_by_text(group_name).locator("xpath=ancestor::div[.//select][1]")
    with owner_page.expect_response(re.compile(r"/access/update")):
        group_entry.get_by_role("button").last.click()

    _sent_home(member_page)


def test_only_the_folders_owner_finds_share_in_its_menu(page_for, crew):
    owner, member, group_id, _ = crew
    folder_id, _, _ = _folder_with_chat(owner)
    _share_folder_over_api(
        owner,
        folder_id,
        [grant("group", group_id, "read"), grant("group", group_id, "write")],
        "continue",
    )

    member_menu = _folder_page_menu(page_for(member), folder_id)
    expect(member_menu.get_by_role("button", name="Share")).to_have_count(0)
    owner_menu = _folder_page_menu(page_for(owner), folder_id)
    expect(owner_menu.get_by_role("button", name="Share")).to_be_visible()
