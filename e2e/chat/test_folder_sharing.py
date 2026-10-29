"""Journey: a folder shared with a group from its sidebar menu, and what a member then meets.

The owner, in a group allowed to share folders, opens Edit in the folder's "More" menu, then Access
in the folder editor, and adds the group in the Access List; the folder is shared for writing
unless the owner turns the group's Access level down to Read. A member of the group finds the
folder in their own sidebar, lists the owner's chats in it and opens one read-only: no message
input, a Clone Chat button in its place. A member allowed to write starts a chat on the folder's
page: the chat is filed in the folder, the folder's system prompt reaches the model for it as for
the owner's own chats, and the owner finds the member's chat in the folder. A member who may only
read gets no message input on the folder's page. Dragged onto the shared folder in the sidebar, a
writing member's own chat is filed there, while a reading member's chat is not taken at all.
Removing the group from the Access List takes the folder out of the member's sidebar again. Only
the owner or an admin changes a folder's sharing (8145774e3): a writing member who opens the folder
editor from its sidebar menu or from the menu on its page finds no Access button there, while the
owner and an admin still find it. Since 0.12.4 (226207314) the owner renames a member's chat in
the shared folder from Rename in its sidebar menu, and drags it into a folder inside the shared
one, while a drag out to a private folder of the owner's is refused and the chat stays.

Discriminates: passes on dev ebc6add67, and on dev b5a20423e once the read share test looks for the
Clone Chat button that replaced the Read only label (c6dd9a451), and on dev f5a26f53a once sharing
is reached through the folder editor's Access button, where f4557ca06 moved the menu's Share entry.
In a frontend copy whose Access dialog never saves a change the three tests that share or unshare
in the dialog go red. In backend copies: listing no shared folders turns the read share, unshare
and both drag tests red at the member's sidebar; the middleware's folder lookup returning no folder
turns the writing member's test red (no folder prompt); reporting every shared folder as writable
turns the read-only input and drop tests red. In a frontend copy whose folder editor shows Access
to everyone the two tests for a writing member go red. The owner's rename and drag of a member's
chat pass on dev 948432b43; both go red in a frontend copy whose folder rows never let the owner
rename or move another account's chat (no menu, not draggable), the rename test in a backend copy
refusing the owner in the chat update handler and the drag test in one whose chat move check
always answers False.
"""

from __future__ import annotations

import re
import time
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.access import make_group
from harness.chat_history import seed_chat
from utils.chat_ui import chat_input, conversation, expect_reply, send
from utils.sidebar_chats import chat_menu, chat_row
from utils.sidebar_folders import folder_link, folder_row

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PROMPT = "Answer as the harbour master of Port Ellen."
FOLDER_SHARING = {"sharing": {"folders": True}}


def folder_name() -> str:
    return f"Harbour {uuid.uuid4().hex[:6]}"


@pytest.fixture
def crew(make_user, admin):
    """An owner and a member in one group allowed to share folders; yields them and its name."""
    owner, member = make_user(), make_user()
    group_id = make_group(admin, [owner, member], FOLDER_SHARING)
    with admin.client() as client:
        group_name = client.get(f"/api/v1/groups/id/{group_id}").json()["name"]
    return owner, member, group_id, group_name


def create_folder(owner, name: str) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/folders/", json={"name": name, "data": {"system_prompt": PROMPT}}
        )
        assert created.status_code == 200, created.text
        folder_id = created.json()["id"]
        expanded = client.post(
            f"/api/v1/folders/{folder_id}/update/expanded", json={"is_expanded": True}
        )
        assert expanded.status_code == 200, expanded.text
    return folder_id


def filed_chat(owner, folder_id: str, title: str) -> str:
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "when is high tide?"},
                {"role": "assistant", "content": "High tide at noon."},
            ],
        )
        titled = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": title}})
        assert titled.status_code == 200, titled.text
        moved = client.post(f"/api/v1/chats/{chat_id}/folder", json={"folder_id": folder_id})
        assert moved.status_code == 200, moved.text
    return chat_id


def share_over_api(owner, folder_id: str, group_id: str, permission: str) -> None:
    grants = [{"principal_type": "group", "principal_id": group_id, "permission": "read"}]
    if permission == "write":
        grants.append({"principal_type": "group", "principal_id": group_id, "permission": "write"})
    with owner.client() as client:
        shared = client.post(
            f"/api/v1/folders/{folder_id}/access/update", json={"access_grants": grants}
        )
    assert shared.status_code == 200, shared.text


def stored_grants(owner, folder_id: str) -> set[tuple[str, str, str]]:
    with owner.client() as client:
        folder = client.get(f"/api/v1/folders/{folder_id}").json()
    grants = folder.get("access_grants") or []
    return {
        (grant["principal_type"], grant["principal_id"], grant["permission"]) for grant in grants
    }


def eventually_grants(owner, folder_id: str, expected: set, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    current = stored_grants(owner, folder_id)
    while current != expected and time.monotonic() < deadline:
        time.sleep(0.2)
        current = stored_grants(owner, folder_id)
    assert current == expected


def open_sidebar(page: Page) -> Locator:
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


def open_editor(page: Page, menu: Locator, name: str) -> Locator:
    menu.get_by_role("menuitem", name="Edit").click()
    editor = page.get_by_role("dialog").filter(has_text="Edit Folder")
    # the folder loads after the dialog opens, and with it the Access button
    expect(editor.get_by_placeholder("Enter folder name")).to_have_value(name)
    return editor


def open_share_dialog(page: Page, name: str) -> Locator:
    editor = open_editor(page, open_row_menu(page, name), name)
    editor.get_by_role("button", name="Access", exact=True).click()
    dialog = page.get_by_role("dialog").filter(has_text=f"Access: {name}")
    expect(dialog.get_by_role("button", name="Add Access")).to_be_visible()
    return dialog


def add_group(page: Page, dialog: Locator, group_name: str) -> None:
    dialog.get_by_role("button", name="Add Access").click()
    picker = page.get_by_role("dialog").filter(has=page.get_by_role("button", name="Add"))
    picker.get_by_placeholder("Search").fill(group_name)
    picker.get_by_role("button", name=group_name).last.click()
    picker.get_by_role("button", name="Add", exact=True).click()
    expect(dialog.get_by_text(group_name)).to_be_visible()


def test_a_group_shared_to_read_lists_the_folder_and_opens_its_chats_read_only(page_for, crew):
    owner, member, group_id, group_name = crew
    name = folder_name()
    folder_id = create_folder(owner, name)
    title = f"Tide log {uuid.uuid4().hex[:6]}"
    chat_id = filed_chat(owner, folder_id, title)
    owner_page = page_for(owner)

    dialog = open_share_dialog(owner_page, name)
    add_group(owner_page, dialog, group_name)
    dialog.get_by_role("combobox", name="Access level").select_option("read")
    eventually_grants(owner, folder_id, {("group", group_id, "read")})

    member_page = page_for(member)
    shared = folder_row(open_sidebar(member_page), name)
    expect(shared).to_be_visible()
    shared.click()
    expect(member_page).to_have_url(re.compile(f"/folders/{folder_id}"))
    member_page.get_by_role("navigation", name="Chat history").get_by_role(
        "button", name=title
    ).click()
    expect(member_page).to_have_url(re.compile(f"/c/{chat_id}"))
    expect(conversation(member_page).get_by_text("High tide at noon.")).to_be_visible()
    expect(member_page.get_by_role("button", name="Clone Chat")).to_be_visible()
    expect(chat_input(member_page)).to_have_count(0)


def test_a_writing_member_starts_a_chat_in_the_folder_under_its_prompt(page_for, crew, upstream):
    owner, member, group_id, group_name = crew
    name = folder_name()
    folder_id = create_folder(owner, name)
    owner_page = page_for(owner)
    dialog = open_share_dialog(owner_page, name)
    add_group(owner_page, dialog, group_name)
    expect(dialog.get_by_role("combobox", name="Access level")).to_have_value("write")
    eventually_grants(owner, folder_id, {("group", group_id, "read"), ("group", group_id, "write")})

    member_page = page_for(member, path=f"/folders/{folder_id}")
    upstream.queue(reply.text("Berth 4 is free.", match=reply.answering("where may I moor?")))
    send(member_page, "where may I moor?")
    expect_reply(member_page, "Berth 4 is free.")
    expect(member_page).to_have_url(re.compile("/c/"))
    member_chat_id = member_page.url.rsplit("/", 1)[-1]

    [request] = [r for r in upstream.chat_requests() if reply.answering("where may I moor?")(r)]
    system = "\n".join(str(m["content"]) for m in request["messages"] if m["role"] == "system")
    assert system.startswith(PROMPT), "the member's chat in the folder was sent without its prompt"
    with member.client() as client:
        assert client.get(f"/api/v1/chats/{member_chat_id}").json()["folder_id"] == folder_id

    owner_page.reload()
    sidebar = open_sidebar(owner_page)
    owners_view = sidebar.locator(f"#folder-{folder_id}-button").locator(
        "xpath=ancestor::div[@draggable][1]"
    )
    expect(owners_view.get_by_role("button", name=re.compile("where may I moor"))).to_be_visible()


def test_removing_the_group_takes_the_folder_out_of_the_members_sidebar(page_for, crew):
    owner, member, group_id, group_name = crew
    name = folder_name()
    folder_id = create_folder(owner, name)
    share_over_api(owner, folder_id, group_id, "read")
    own_name = folder_name()
    create_folder(member, own_name)
    member_page = page_for(member)
    expect(folder_row(open_sidebar(member_page), name)).to_be_visible()

    owner_page = page_for(owner)
    dialog = open_share_dialog(owner_page, name)
    group_entry = dialog.get_by_text(group_name).locator("xpath=ancestor::div[.//select][1]")
    group_entry.get_by_role("button").last.click()
    expect(dialog.get_by_text(group_name)).to_have_count(0)
    eventually_grants(owner, folder_id, set())

    member_page.reload()
    sidebar = open_sidebar(member_page)
    # the member's own folder shows once the folders have loaded
    expect(folder_row(sidebar, own_name)).to_be_visible()
    expect(folder_row(sidebar, name)).to_have_count(0)


def test_a_reading_member_gets_no_message_input_on_the_folders_page(page_for, crew):
    owner, member, group_id, _ = crew
    folder_id = create_folder(owner, folder_name())
    title = f"Tide log {uuid.uuid4().hex[:6]}"
    filed_chat(owner, folder_id, title)
    share_over_api(owner, folder_id, group_id, "read")
    writer_page = page_for(owner, path=f"/folders/{folder_id}")
    expect(chat_input(writer_page)).to_be_visible()

    member_page = page_for(member, path=f"/folders/{folder_id}")
    expect(member_page.get_by_role("link", name=re.compile(re.escape(title)))).to_be_visible()
    expect(chat_input(member_page)).to_have_count(0)


def drag(page: Page, source: Locator, target: Locator) -> None:
    # a synthetic drag: a mouse drag across the sidebar rows sometimes never fires the drop
    transfer = page.evaluate_handle("() => new DataTransfer()")
    source.dispatch_event("dragstart", {"dataTransfer": transfer})
    target.dispatch_event("dragover", {"dataTransfer": transfer})
    target.dispatch_event("drop", {"dataTransfer": transfer})
    source.dispatch_event("dragend", {"dataTransfer": transfer})


def own_chat(member, title: str) -> str:
    with member.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": "ahoy"}, {"role": "assistant", "content": "ahoy there"}],
        )
        titled = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": title}})
    assert titled.status_code == 200, titled.text
    return chat_id


def stored_folder_id(account, chat_id: str) -> str | None:
    with account.client() as client:
        return client.get(f"/api/v1/chats/{chat_id}").json()["folder_id"]


def test_a_writing_member_drags_their_own_chat_into_the_shared_folder(page_for, crew):
    owner, member, group_id, _ = crew
    name = folder_name()
    folder_id = create_folder(owner, name)
    share_over_api(owner, folder_id, group_id, "write")
    title = f"My mooring {uuid.uuid4().hex[:6]}"
    chat_id = own_chat(member, title)
    member_page = page_for(member)
    sidebar = open_sidebar(member_page)
    expect(sidebar.get_by_role("button", name=title)).to_be_visible()

    with member_page.expect_response(re.compile(f"/api/v1/chats/{chat_id}/folder")):
        drag(member_page, sidebar.get_by_role("button", name=title), folder_row(sidebar, name))

    assert stored_folder_id(member, chat_id) == folder_id
    owner_page = page_for(owner, path=f"/folders/{folder_id}")
    expect(owner_page.get_by_role("link", name=re.compile(re.escape(title)))).to_be_visible()


def test_a_reading_member_cannot_drop_a_chat_into_the_shared_folder(page_for, crew):
    owner, member, group_id, _ = crew
    name = folder_name()
    folder_id = create_folder(owner, name)
    share_over_api(owner, folder_id, group_id, "read")
    title = f"My mooring {uuid.uuid4().hex[:6]}"
    chat_id = own_chat(member, title)
    member_page = page_for(member)
    sidebar = open_sidebar(member_page)
    expect(sidebar.get_by_role("button", name=title)).to_be_visible()
    moves = []
    member_page.on(
        "request", lambda request: moves.append(request) if "/folder" in request.url else None
    )

    drag(member_page, sidebar.get_by_role("button", name=title), folder_row(sidebar, name))
    member_page.wait_for_timeout(1000)  # bounded: a refused drop sends nothing

    assert moves == [], "a read-only shared folder took a dropped chat"
    assert stored_folder_id(member, chat_id) is None


def open_row_menu(page: Page, name: str) -> Locator:
    sidebar = open_sidebar(page)
    row = folder_row(sidebar, name)
    # focus on the name shows the row's buttons like a hover, and keeps them shown while it slides
    folder_link(sidebar, name).focus()
    row.get_by_role("button").last.click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("menuitem", name="Export")).to_be_visible()
    return menu


def open_page_menu(page: Page, folder_id: str) -> Locator:
    page.goto(f"/folders/{folder_id}")
    # the sidebar rows carry the same label, so look in the page itself
    page.get_by_role("button", name="Folder options").last.click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("menuitem", name="Export")).to_be_visible()
    return menu


def test_a_writing_member_finds_no_access_button_editing_from_the_sidebar(page_for, crew):
    owner, member, group_id, _ = crew
    name = folder_name()
    folder_id = create_folder(owner, name)
    share_over_api(owner, folder_id, group_id, "write")
    member_page = page_for(member)

    members_editor = open_editor(member_page, open_row_menu(member_page, name), name)

    expect(members_editor.get_by_role("button", name="Access", exact=True)).to_have_count(0)
    owner_page = page_for(owner)
    owners_editor = open_editor(owner_page, open_row_menu(owner_page, name), name)
    expect(owners_editor.get_by_role("button", name="Access", exact=True)).to_be_visible()


def test_a_writing_member_finds_no_access_button_editing_from_the_folders_page(page_for, crew):
    owner, member, group_id, _ = crew
    name = folder_name()
    folder_id = create_folder(owner, name)
    share_over_api(owner, folder_id, group_id, "write")
    member_page = page_for(member)

    members_editor = open_editor(member_page, open_page_menu(member_page, folder_id), name)

    expect(members_editor.get_by_role("button", name="Access", exact=True)).to_have_count(0)
    owner_page = page_for(owner)
    owners_editor = open_editor(owner_page, open_page_menu(owner_page, folder_id), name)
    expect(owners_editor.get_by_role("button", name="Access", exact=True)).to_be_visible()


def test_an_admin_still_finds_access_editing_a_folder_shared_with_them(page_for, crew, make_user):
    owner, _, _, _ = crew
    admin = make_user(role="admin")
    name = folder_name()
    folder_id = create_folder(owner, name)
    with owner.client() as client:
        grants = [
            {"principal_type": "user", "principal_id": admin.id, "permission": permission}
            for permission in ("read", "write")
        ]
        shared = client.post(
            f"/api/v1/folders/{folder_id}/access/update", json={"access_grants": grants}
        )
    assert shared.status_code == 200, shared.text

    admin_page = page_for(admin)
    admins_editor = open_editor(admin_page, open_row_menu(admin_page, name), name)

    expect(admins_editor.get_by_role("button", name="Access", exact=True)).to_be_visible()


def members_chat_in(member, folder_id: str, title: str) -> str:
    chat_id = own_chat(member, title)
    with member.client() as client:
        filed = client.post(f"/api/v1/chats/{chat_id}/folder", json={"folder_id": folder_id})
    assert filed.status_code == 200, filed.text
    return chat_id


def test_the_owner_renames_a_members_chat_from_its_menu(page_for, crew):
    owner, member, group_id, _ = crew
    folder_id = create_folder(owner, folder_name())
    share_over_api(owner, folder_id, group_id, "write")
    title = f"My mooring {uuid.uuid4().hex[:6]}"
    chat_id = members_chat_in(member, folder_id, title)
    renamed = f"Berth 4 {uuid.uuid4().hex[:6]}"
    owner_page = page_for(owner)
    sidebar = open_sidebar(owner_page)
    expect(chat_row(sidebar, title)).to_be_visible()

    chat_menu(sidebar, title).get_by_role("menuitem", name="Rename").click()
    expect(owner_page.locator(f"#chat-title-input-{chat_id}")).to_be_focused()
    owner_page.keyboard.press("Control+A")
    owner_page.keyboard.type(renamed)
    with owner_page.expect_response(re.compile(f"/api/v1/chats/{chat_id}$")) as answer:
        owner_page.keyboard.press("Enter")

    assert answer.value.status == 200, f"the owner's rename got {answer.value.status}"
    expect(chat_row(sidebar, renamed)).to_be_visible()
    with member.client() as client:
        assert client.get(f"/api/v1/chats/{chat_id}").json()["title"] == renamed


def test_the_owner_drags_a_members_chat_within_the_shared_folder_but_not_out(page_for, crew):
    owner, member, group_id, _ = crew
    folder_id = create_folder(owner, folder_name())
    inner_name = folder_name()
    inner_id = create_folder(owner, inner_name)
    with owner.client() as client:
        nested = client.post(
            f"/api/v1/folders/{inner_id}/update/parent", json={"parent_id": folder_id}
        )
    assert nested.status_code == 200, nested.text
    private_name = folder_name()
    create_folder(owner, private_name)
    share_over_api(owner, folder_id, group_id, "write")
    title = f"My mooring {uuid.uuid4().hex[:6]}"
    chat_id = members_chat_in(member, folder_id, title)
    owner_page = page_for(owner)
    sidebar = open_sidebar(owner_page)
    row = chat_row(sidebar, title)
    expect(row).to_be_visible()
    expect(row.locator("xpath=ancestor::div[@draggable][1]")).to_have_attribute("draggable", "true")

    with owner_page.expect_response(re.compile(f"/api/v1/chats/{chat_id}/folder")) as refused:
        drag(owner_page, row, folder_row(sidebar, private_name))
    assert refused.value.status == 401, f"the move out got {refused.value.status}"
    assert stored_folder_id(member, chat_id) == folder_id

    with owner_page.expect_response(re.compile(f"/api/v1/chats/{chat_id}/folder")) as moved:
        drag(owner_page, row, folder_row(sidebar, inner_name))
    assert moved.value.status == 200, f"the move within got {moved.value.status}"
    assert stored_folder_id(member, chat_id) == inner_id
