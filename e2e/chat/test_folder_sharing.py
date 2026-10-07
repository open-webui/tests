"""Journey: a folder shared with a group from its sidebar menu, and what a member then meets.

The owner, in a group allowed to share folders, opens Share in the folder's "More" menu and adds
the group in the Access List; the folder is shared for writing unless the owner turns the
group's Access level down to Read. A member of the group finds the folder in their own sidebar,
lists the owner's chats in it and opens one read-only, with no message input. A member allowed
to write starts a chat on the folder's page: the chat is filed in the folder, the folder's
system prompt reaches the model for it as for the owner's own chats, and the owner finds the
member's chat in the folder. A member who may only read gets no message input on the folder's
page. Dragged onto the shared folder in the sidebar, a writing member's own chat is filed there,
while a reading member's chat is not taken at all. Removing the group from the Access List takes
the folder out of the member's sidebar again.

Discriminates: passes on dev ebc6add67. In a frontend copy whose Share dialog never saves a change
the three tests that share or unshare in the dialog go red. In backend copies: listing no shared
folders turns the read share, unshare and both drag tests red at the member's sidebar; the
middleware's folder lookup returning no folder turns the writing member's test red (no folder
prompt); reporting every shared folder as writable turns the read-only input and drop tests red.
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


def folder_row(sidebar: Locator, name: str) -> Locator:
    return sidebar.get_by_role("button", name=re.compile(f"^{re.escape(name)}"))


def open_share_dialog(page: Page, name: str) -> Locator:
    row = folder_row(open_sidebar(page), name)
    row.hover()
    row.get_by_role("button").last.click()
    page.get_by_role("menu").get_by_role("button", name="Share").click()
    dialog = page.get_by_role("dialog").filter(has_text=f"Share: {name}")
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
    expect(member_page.get_by_text("Read only", exact=True)).to_be_visible()
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

    member_page = page_for(member)
    member_page.goto(f"/folders/{folder_id}")
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
    writer_page = page_for(owner)
    writer_page.goto(f"/folders/{folder_id}")
    expect(chat_input(writer_page)).to_be_visible()

    member_page = page_for(member)
    member_page.goto(f"/folders/{folder_id}")
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
    owner_page = page_for(owner)
    owner_page.goto(f"/folders/{folder_id}")
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
