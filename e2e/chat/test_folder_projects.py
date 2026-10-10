"""Journey: chat folders as projects, from the sidebar and the folder page.

A person creates a folder and a subfolder in the sidebar, gives a folder a system prompt, a
knowledge base and an uploaded file in its settings, drags chats into, between and out of
folders, renames and deletes folders, opens a folder's page and folds folders open and shut.
A pinned chat that also sits in a folder, dragged onto Chats or onto Pinned, lands there with the
stored pinned state that section means (open-webui/open-webui#31368, issue #31367).
A folder's icon, picked on its page, shows on its sidebar row, and a background image uploaded
in its edit dialog shows behind the folder page; both are stored and survive a reload.
Each outcome is read back from what the server stored or after a reload, and the folder's
settings from the request the scripted provider receives for a new chat started in the folder.

Discriminates: passes on the dev a5bc78300 build; in a frontend copy of the dev 176d31d1d build
each test fails, one edit each: the subfolder dialog creating at the top level, the folder dialog
saving without its prompt and files, a chat dropped on a folder not being moved, the in-place
rename saving the old name, the delete confirmation inverting its checkbox (both delete tests), a
chat started on the folder page sent without the folder, the expand toggle not being saved, the
icon pick saved empty and the edit dialog saving without the background image. In the a5bc78300
build with #31368 reverted, the pinned chat dropped on Chats stays under Pinned and the one
dropped on Pinned ends unpinned.
The folder settings test was retargeted for 8d0ff76f2, whose folder dialog picks knowledge and
uploads files in its Knowledge picker: it passes on dev 76ad6f97c (3 of 3) and fails in a build of
it whose folder dialog saves without its files.

The in-place rename test also pins open-webui/open-webui#31582, fixed by PR #31584: pressing Enter
saved the folder twice, so two update requests and two "Folder updated successfully" toasts
followed one rename. It fails on dev 176d31d1d.
"""

from __future__ import annotations

import base64
import re
import time
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.image_engines import PNG_BASE64
from harness.knowledge_bases import knowledge_base
from utils.chat_ui import expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SYSTEM_PROMPT = "Answer as the harbour master of Port Ellen."


def create_folder(owner, name: str, parent_id: str | None = None) -> str:
    with owner.client() as client:
        created = client.post("/api/v1/folders/", json={"name": name, "parent_id": parent_id})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def expand_folder(owner, folder_id: str) -> None:
    with owner.client() as client:
        expanded = client.post(
            f"/api/v1/folders/{folder_id}/update/expanded", json={"is_expanded": True}
        )
    assert expanded.status_code == 200, expanded.text


def create_chat(owner, title: str, folder_id: str | None = None) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/chats/new", json={"chat": {"title": title}, "folder_id": folder_id}
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def stored_folders(owner) -> dict[str, dict]:
    with owner.client() as client:
        listed = client.get("/api/v1/folders/")
    assert listed.status_code == 200, listed.text
    return {folder["name"]: folder for folder in listed.json()}


def stored_folder(owner, folder_id: str) -> dict:
    with owner.client() as client:
        found = client.get(f"/api/v1/folders/{folder_id}")
    assert found.status_code == 200, found.text
    return found.json()


def stored_chat(owner, chat_id: str) -> dict | None:
    with owner.client() as client:
        found = client.get(f"/api/v1/chats/{chat_id}")
    return found.json() if found.status_code == 200 else None


def open_sidebar(page: Page) -> Locator:
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    show_folders(sidebar)
    return sidebar


def show_folders(sidebar: Locator) -> None:
    section = sidebar.get_by_role("button", name="Folders", exact=True)
    expect(section).to_be_visible()
    if section.get_attribute("aria-expanded") != "true":
        section.click()


def reloaded(sidebar: Locator) -> Locator:
    sidebar.page.reload()  # the sidebar stays open
    show_folders(sidebar)
    return sidebar


def folder_row(sidebar: Locator, name: str) -> Locator:
    return sidebar.get_by_role("button", name=name, exact=True)


def expand_toggle(sidebar: Locator, name: str) -> Locator:
    return folder_row(sidebar, name).get_by_role("button").first


def folder_menu(sidebar: Locator, name: str) -> Locator:
    """Open the "More" menu on the folder's row, a tooltip-only button shown on hover."""
    row = folder_row(sidebar, name)
    row.hover()
    row.get_by_role("button").last.click()
    return sidebar.page.get_by_role("menu")


def folder_content(sidebar: Locator, folder_id: str) -> Locator:
    """The folder's row with everything filed under it."""
    row = sidebar.locator(f"#folder-{folder_id}-button")
    return row.locator("xpath=ancestor::div[@draggable][1]")


def chat_row(scope: Locator, title: str) -> Locator:
    return scope.get_by_role("button", name=title)


def submit_folder_dialog(page: Page, name: str) -> None:
    dialog = page.get_by_role("dialog")
    dialog.get_by_placeholder("Enter folder name").fill(name)
    dialog.get_by_role("button", name="Save").click()


def drag(page: Page, source: Locator, target: Locator) -> None:
    # a synthetic drag: a mouse drag across the sidebar rows sometimes never fires the drop
    transfer = page.evaluate_handle("() => new DataTransfer()")
    source.dispatch_event("dragstart", {"dataTransfer": transfer})
    target.dispatch_event("dragover", {"dataTransfer": transfer})
    target.dispatch_event("drop", {"dataTransfer": transfer})
    source.dispatch_event("dragend", {"dataTransfer": transfer})


def test_a_folder_and_a_subfolder_created_in_the_sidebar_are_kept(page_for, make_user):
    owner = make_user()
    page = page_for(owner)
    sidebar = open_sidebar(page)

    sidebar.get_by_role("button", name="New Folder").click()
    submit_folder_dialog(page, "Garden")
    expect(folder_row(sidebar, "Garden")).to_be_visible()
    folder_menu(sidebar, "Garden").get_by_role("button", name="Create Folder").click()
    submit_folder_dialog(page, "Seeds")
    expect(page.get_by_text("Folder created successfully")).to_be_visible()

    folders = stored_folders(owner)
    assert folders["Seeds"]["parent_id"] == folders["Garden"]["id"]
    reloaded(sidebar)
    expect(folder_row(sidebar, "Seeds")).to_have_count(0)
    expand_toggle(sidebar, "Garden").click()
    expect(folder_row(sidebar, "Seeds")).to_be_visible()


def test_the_folder_settings_reach_a_new_chat_started_in_the_folder(
    page_for, make_user, admin, upstream
):
    owner = make_user()
    folder_id = create_folder(owner, "Harbour")
    base_name = f"Tide tables {uuid.uuid4().hex[:8]}"
    readable = [{"principal_type": "user", "principal_id": owner.id, "permission": "read"}]
    with admin.client() as client, knowledge_base(client, base_name, readable) as knowledge_id:
        page = page_for(owner)
        sidebar = open_sidebar(page)
        folder_menu(sidebar, "Harbour").get_by_role("button", name="Edit").click()
        dialog = page.get_by_role("dialog")
        dialog.get_by_placeholder(re.compile("Write your model system prompt")).fill(SYSTEM_PROMPT)
        dialog.get_by_role("button", name=re.compile("^Knowledge")).and_(
            dialog.locator("button[aria-expanded]")
        ).click()
        page.get_by_placeholder("Search knowledge").fill(base_name)
        page.get_by_role("button", name=base_name).click()
        expect(dialog.get_by_text(base_name)).to_be_visible()
        with page.expect_file_chooser() as chooser:
            page.get_by_role("button", name="Upload Files").click()
        chooser.value.set_files(
            files=[{"name": "moorings.txt", "mimeType": "text/plain", "buffer": b"Berth 4."}]
        )
        expect(dialog.get_by_text("moorings.txt")).to_be_visible()
        expect(page.get_by_text("Uploading")).to_have_count(0)
        page.get_by_role("button", name="Done").click()
        dialog.get_by_role("button", name="Save").click()
        expect(page.get_by_text("Folder updated successfully")).to_be_visible()

        saved = stored_folder(owner, folder_id)["data"] or {}
        assert saved.get("system_prompt") == SYSTEM_PROMPT
        files = saved.get("files") or []
        attached = {(item["type"], item.get("name")): item["id"] for item in files}
        assert attached[("collection", base_name)] == knowledge_id
        file_id = attached[("file", "moorings.txt")]

        folder_row(sidebar, "Harbour").click()
        expect(page).to_have_url(re.compile(f"/folders/{folder_id}"))
        upstream.queue(reply.text("High tide at noon", match=reply.answering("when is high tide?")))
        send(page, "when is high tide?")
        expect_reply(page, "High tide at noon")
        expect(page).to_have_url(re.compile("/c/"))

    [request] = [r for r in upstream.chat_requests() if reply.answering("when is high tide?")(r)]
    system = "\n".join(m["content"] for m in request["messages"] if m["role"] == "system")
    assert system.startswith(SYSTEM_PROMPT), system
    assert f'type="collection" id="{knowledge_id}"' in system, system
    assert f'type="file" id="{file_id}"' in system, system
    chat_id = page.url.rsplit("/", 1)[-1]
    assert stored_chat(owner, chat_id)["folder_id"] == folder_id


def test_a_chat_dragged_into_between_and_out_of_folders_lands_each_time(page_for, make_user):
    owner = make_user()
    folder_ids = {name: create_folder(owner, name) for name in ("Alpha", "Beta")}
    for folder_id in folder_ids.values():
        expand_folder(owner, folder_id)
    chat_id = create_chat(owner, "Loose ends")
    page = page_for(owner)
    sidebar = open_sidebar(page)
    expect(sidebar.get_by_text("No chats")).to_have_count(2)

    for name, folder_id in folder_ids.items():
        with page.expect_response(re.compile(f"/api/v1/chats/{chat_id}/folder")):
            drag(page, chat_row(sidebar, "Loose ends"), folder_row(sidebar, name))
        expect(chat_row(folder_content(sidebar, folder_id), "Loose ends")).to_be_visible()
        expect(chat_row(sidebar, "Loose ends")).to_have_count(1)
        assert stored_chat(owner, chat_id)["folder_id"] == folder_id

    chats_section = sidebar.get_by_role("button", name="Chats", exact=True)
    with page.expect_response(re.compile(f"/api/v1/chats/{chat_id}/folder")):
        drag(page, chat_row(sidebar, "Loose ends"), chats_section)
    expect(sidebar.get_by_text("No chats")).to_have_count(2)
    expect(chat_row(sidebar, "Loose ends")).to_have_count(1)
    assert stored_chat(owner, chat_id)["folder_id"] is None


def pinned_section(sidebar: Locator) -> Locator:
    """The Pinned group inside the Chats section, found from its header button."""
    header = sidebar.get_by_role("button", name="Pinned", exact=True)
    return header.locator("xpath=ancestor::div[.//div[@id='sidebar-chat-group']][1]")


def pin_chat(owner, chat_id: str) -> None:
    with owner.client() as client:
        pinned = client.post(f"/api/v1/chats/{chat_id}/pin")
    assert pinned.status_code == 200, pinned.text


def wait_until_pinned(owner, chat_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    stored = stored_chat(owner, chat_id)
    while not stored["pinned"] and time.monotonic() < deadline:
        time.sleep(0.2)
        stored = stored_chat(owner, chat_id)
    return stored


def pinned_and_filed(owner, title: str, folder_id: str) -> str:
    chat_id = create_chat(owner, title, folder_id)
    pin_chat(owner, chat_id)
    stored = stored_chat(owner, chat_id)
    assert (stored["pinned"], stored["folder_id"]) == (True, folder_id)
    return chat_id


@pytest.mark.regression
def test_a_pinned_chat_in_a_folder_dragged_onto_chats_is_unpinned_and_listed_there(
    page_for, make_user
):
    owner = make_user()
    folder_id = create_folder(owner, "Alpha")
    expand_folder(owner, folder_id)
    chat_id = pinned_and_filed(owner, "Pinned and filed", folder_id)
    sidebar = open_sidebar(page_for(owner))
    pinned = pinned_section(sidebar)
    expect(chat_row(pinned, "Pinned and filed")).to_be_visible()

    drag(
        sidebar.page,
        chat_row(pinned, "Pinned and filed"),
        sidebar.get_by_role("button", name="Chats", exact=True),
    )

    chats_content = sidebar.locator("#sidebar-chats-content")
    expect(
        sidebar.get_by_role("button", name="Pinned", exact=True),
        "the chat dropped on Chats stayed under Pinned (open-webui/open-webui#31367)",
    ).to_have_count(0)
    expect(chat_row(chats_content, "Pinned and filed")).to_be_visible()
    stored = stored_chat(owner, chat_id)
    assert stored["pinned"] is False, "the chat dropped on Chats is still pinned (#31367)"
    assert stored["folder_id"] is None
    reloaded(sidebar)
    expect(chat_row(chats_content, "Pinned and filed")).to_be_visible()


@pytest.mark.regression
def test_a_pinned_chat_in_a_folder_dragged_onto_pinned_stays_pinned_outside_the_folder(
    page_for, make_user
):
    owner = make_user()
    folder_id = create_folder(owner, "Alpha")
    expand_folder(owner, folder_id)
    anchor_id = create_chat(owner, "Anchor")
    pin_chat(owner, anchor_id)
    chat_id = pinned_and_filed(owner, "Pinned and filed", folder_id)
    sidebar = open_sidebar(page_for(owner))
    pinned = pinned_section(sidebar)
    expect(chat_row(pinned, "Anchor")).to_be_visible()

    with sidebar.page.expect_response(re.compile(f"/api/v1/chats/{chat_id}/folder")):
        drag(
            sidebar.page,
            chat_row(pinned, "Pinned and filed"),
            sidebar.get_by_role("button", name="Pinned", exact=True),
        )

    stored = wait_until_pinned(owner, chat_id)
    assert stored["pinned"] is True, "the chat dropped on Pinned is not pinned (#31367)"
    assert stored["folder_id"] is None
    expect(chat_row(pinned, "Pinned and filed")).to_be_visible()
    reloaded(sidebar)
    expect(chat_row(pinned_section(sidebar), "Pinned and filed")).to_be_visible()


def test_a_folder_renamed_in_place_keeps_its_new_name(page_for, make_user):
    owner = make_user()
    folder_id = create_folder(owner, "Drafts")
    update_path = f"/api/v1/folders/{folder_id}/update"
    sidebar = open_sidebar(page_for(owner))

    folder_row(sidebar, "Drafts").dblclick()
    updates = []
    sidebar.page.on(
        "request",
        lambda request: updates.append(request) if request.url.endswith(update_path) else None,
    )
    rename_input = sidebar.get_by_role("textbox")
    rename_input.fill("Final copies")
    rename_input.press("Enter")
    toasts = sidebar.page.get_by_text("Folder updated successfully")
    expect(toasts.first).to_be_visible()
    expect(toasts, "one rename showed several toasts (open-webui/open-webui#31582)").to_have_count(
        1
    )
    assert len(updates) == 1, (
        f"one rename saved the folder {len(updates)} times (open-webui/open-webui#31582)"
    )

    assert stored_folder(owner, folder_id)["name"] == "Final copies"
    reloaded(sidebar)
    expect(folder_row(sidebar, "Final copies")).to_be_visible()
    expect(folder_row(sidebar, "Drafts")).to_have_count(0)


def delete_from_the_sidebar(sidebar: Locator, name: str, delete_contents: bool) -> None:
    folder_menu(sidebar, name).get_by_role("button", name="Delete").click()
    dialog = sidebar.page.get_by_role("dialog", name="Delete folder?")
    expect(dialog.get_by_text(f'Are you sure you want to delete "{name}"?')).to_be_visible()
    checkbox = dialog.get_by_role("checkbox")
    expect(checkbox).to_be_checked()  # deleting the contents is the default
    checkbox.set_checked(delete_contents)
    dialog.get_by_role("button", name="Confirm").click()
    expect(sidebar.page.get_by_text("Folder deleted successfully")).to_be_visible()


@pytest.fixture
def filed_owner(make_user):
    """An account with a folder "Trips" holding a chat and a subfolder holding another.

    Returns the account and the ids of the two chats.
    """
    owner = make_user()
    folder_id = create_folder(owner, "Trips")
    subfolder_id = create_folder(owner, "Spring", parent_id=folder_id)
    chat_ids = [
        create_chat(owner, "Ferry times", folder_id),
        create_chat(owner, "Hostel list", subfolder_id),
    ]
    return owner, chat_ids


def test_deleting_a_folder_with_its_contents_deletes_its_chats_and_subfolders(
    page_for, filed_owner
):
    owner, chat_ids = filed_owner
    sidebar = open_sidebar(page_for(owner))

    delete_from_the_sidebar(sidebar, "Trips", delete_contents=True)

    expect(folder_row(sidebar, "Trips")).to_have_count(0)
    assert stored_folders(owner) == {}
    assert [stored_chat(owner, chat_id) for chat_id in chat_ids] == [None, None]
    reloaded(sidebar)
    expect(chat_row(sidebar, "Ferry times")).to_have_count(0)


def test_deleting_a_folder_but_not_its_contents_moves_its_chats_to_the_chat_list(
    page_for, filed_owner
):
    owner, chat_ids = filed_owner
    sidebar = open_sidebar(page_for(owner))

    delete_from_the_sidebar(sidebar, "Trips", delete_contents=False)

    expect(folder_row(sidebar, "Trips")).to_have_count(0)
    assert stored_folders(owner) == {}
    stored = [stored_chat(owner, chat_id) for chat_id in chat_ids]
    assert None not in stored, "the chats were deleted along with the folder"
    assert [chat["folder_id"] for chat in stored] == [None, None]
    expect(chat_row(sidebar, "Ferry times")).to_be_visible()
    expect(chat_row(sidebar, "Hostel list")).to_be_visible()


def test_the_folder_page_lists_its_chats_and_files_a_new_chat_there(page_for, make_user, upstream):
    owner = make_user()
    folder_id = create_folder(owner, "Recipes")
    filed_chat_id = create_chat(owner, "Bread starter", folder_id)
    create_chat(owner, "Unfiled musings")
    page = page_for(owner)

    page.goto(f"/folders/{folder_id}")
    expect(page.get_by_text("Recipes", exact=True).first).to_be_visible()
    expect(page.get_by_role("link", name=re.compile("Bread starter"))).to_have_attribute(
        "href", f"/c/{filed_chat_id}"
    )
    expect(page.get_by_role("link", name=re.compile("Unfiled musings"))).to_have_count(0)

    upstream.queue(reply.text("Knead for ten minutes", match=reply.answering("how long to knead?")))
    send(page, "how long to knead?")
    expect_reply(page, "Knead for ten minutes")
    expect(page).to_have_url(re.compile("/c/"))
    new_chat_id = page.url.rsplit("/", 1)[-1]
    assert stored_chat(owner, new_chat_id)["folder_id"] == folder_id

    page.goto(f"/folders/{folder_id}")
    expect(page.get_by_role("link", name=re.compile("how long to knead?"))).to_have_attribute(
        "href", f"/c/{new_chat_id}"
    )


def test_a_folder_stays_open_or_shut_across_reloads(page_for, make_user):
    owner = make_user()
    folder_id = create_folder(owner, "Letters")
    create_chat(owner, "Dear Ada", folder_id)
    sidebar = open_sidebar(page_for(owner))
    expect(chat_row(sidebar, "Dear Ada")).to_have_count(0)

    with sidebar.page.expect_response(re.compile(f"/folders/{folder_id}/update/expanded")):
        expand_toggle(sidebar, "Letters").click()
    expect(chat_row(sidebar, "Dear Ada")).to_be_visible()
    assert stored_folder(owner, folder_id)["is_expanded"] is True
    reloaded(sidebar)
    expect(chat_row(sidebar, "Dear Ada")).to_be_visible()

    with sidebar.page.expect_response(re.compile(f"/folders/{folder_id}/update/expanded")):
        expand_toggle(sidebar, "Letters").click()
    expect(chat_row(sidebar, "Dear Ada")).to_have_count(0)
    assert stored_folder(owner, folder_id)["is_expanded"] is False
    reloaded(sidebar)
    expect(folder_row(sidebar, "Letters")).to_be_visible()
    expect(chat_row(sidebar, "Dear Ada")).to_have_count(0)


def test_a_folders_icon_and_background_image_show_and_are_stored(page_for, make_user, tmp_path):
    owner = make_user()
    folder_id = create_folder(owner, "Seascapes")
    picture = tmp_path / "backdrop.png"
    picture.write_bytes(base64.b64decode(PNG_BASE64))
    page = page_for(owner)
    page.goto(f"/folders/{folder_id}")
    background = page.locator("[style*='background-image']")
    expect(background).to_have_count(0)

    page.get_by_label("Change folder icon").click()
    page.get_by_placeholder("Search all emojis").fill("anchor")
    page.get_by_role("button", name="2693", exact=True).click()
    expect(page.get_by_text("Folder updated successfully")).to_be_visible()
    assert stored_folder(owner, folder_id)["meta"]["icon"] == "anchor"

    sidebar = open_sidebar(page)
    expect(sidebar.locator(f"#folder-{folder_id}-button").get_by_alt_text("anchor")).to_be_visible()

    row = sidebar.locator(f"#folder-{folder_id}-button")
    row.hover()
    row.get_by_role("button").last.click()
    page.get_by_role("menu").get_by_role("button", name="Edit").click()
    dialog = page.get_by_role("dialog")
    dialog.locator("#folder-background-image-input").set_input_files(picture)
    expect(dialog.get_by_role("button", name="Reset")).to_be_visible()
    dialog.get_by_role("button", name="Save").click()
    expect(background).to_have_count(1)

    stored = stored_folder(owner, folder_id)["meta"]
    assert stored["icon"] == "anchor", "saving the background dropped the icon"
    assert stored["background_image_url"].startswith("data:image/png;base64,")

    page.reload()
    expect(background).to_have_count(1)
    show_folders(sidebar)
    expect(sidebar.locator(f"#folder-{folder_id}-button").get_by_alt_text("anchor")).to_be_visible()
