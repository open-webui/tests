"""Journey: building, browsing, sharing and emptying a knowledge base in the workspace.

A fresh admin creates a knowledge base, finds it by name in the list and deletes it from the
row's menu. Inside a base it uploads several files at once, pastes text as a new file and
uploads a folder, whose subfolders come along; each file is listed once processed and opening
one shows its text. A directory made from the add-content menu is listed after a reload and holds
the text added inside it, and deleting it removes what it held, so a chat that attaches the base
is no longer sent that text. Organising a base is read back after a reload and over the API: a
directory renamed from its row keeps its files, a file dragged onto a directory is listed inside
it and no longer at the top level, a directory dragged onto another shows nested with its files,
and deleting a directory while keeping its contents moves the files up a level, where a chat that
attaches the base is still sent their text. A webpage added by its link is listed once processed
and its text reaches a chat that attaches the base; the page is a local service, on an instance
that may fetch loopback addresses. An admin exports a base from its row's menu and the downloaded
zip holds the base's files and their text. The base's search narrows the list to matching file
names, and a file removed from the base is no longer sent to a chat that attaches the base. The
owner renames the base, shares it with a group read-only and then with write access, and a group
member sees each change: the new name, the base marked read only with its controls locked, then
editable. An account outside the group does not see it. Resetting the base empties it. Syncing a
local directory into a base uploads it, and syncing it again after files changed on disk adds the
new file, replaces the changed one and removes the deleted one, with a summary that counts each,
so a chat that attaches the base is sent only the current text. With a Max Upload Size set, a file
over it is refused with the limit named while the rest upload, and a directory upload names how
many of its files failed.

Chromium's directory picker cannot be driven by Playwright, so the folder upload and sync tests
hide it and the page takes the file input it offers browsers without one.

Discriminates: passes on dev 176d31d1d. In a frontend copy, uploading only the first of the
chosen files fails the upload test, uploading a folder's files without their subfolder fails
the folder test, skipping the file's text fetch fails the preview test, a search input that
does not search fails the search test, an edit that is never saved fails the rename test, a
name field left editable for readers fails the read-only test, an access level select that
changes nothing fails the write test, a reset confirm that resets nothing fails the reset test
and a delete confirm that deletes nothing fails the create and delete test. In a backend copy,
leaving a removed file's chunks in the base's collection fails the removal test (the chat is
still sent its text), a directory delete route that answers success and deletes nothing fails
the directory test, a web page route that returns no text fails the webpage test, an export
that leaves the files out of the zip fails the export test, a directory rename that is never
stored fails the directory rename test, a file move that is never stored fails the file drag test,
a directory move that is never stored fails the directory drag test and a directory delete that
removes the files it was asked to keep fails the keep files test. In a frontend copy of dev
ebc6add67, a sync that skips removing stale files fails the sync test, an upload without its size
check fails the size test and a directory upload that stops counting failed files fails the
directory size test.
"""

from __future__ import annotations

import io
import json
import re
import uuid
import zipfile
from pathlib import Path

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.access import make_group
from harness.actors import Actor, admin_of
from harness.chat import ask
from harness.knowledge_bases import add_text_file
from harness.listener import listening, text_answer
from harness.web_retrieval import LOCAL_WEB_FETCH, RETRIEVAL_CONFIG

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

KNOWLEDGE_USER = {"workspace": {"knowledge": True}}
DESCRIPTION = "berths, tides and the harbour office"


@pytest.fixture
def curator(make_user):
    """A fresh admin, whose knowledge bases are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for knowledge in client.get("/api/v1/knowledge/").json().get("items", []):
            if knowledge["user_id"] == account.id:
                client.delete(f"/api/v1/knowledge/{knowledge['id']}/delete")


def _delete_bases_of(account: Actor) -> None:
    with account.client() as client:
        for knowledge in client.get("/api/v1/knowledge/").json().get("items", []):
            if knowledge["user_id"] == account.id:
                client.delete(f"/api/v1/knowledge/{knowledge['id']}/delete")


def _unique(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:6]}"


def _create_base(owner: Actor, name: str) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/knowledge/create",
            json={"name": name, "description": DESCRIPTION, "access_grants": []},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _add_files(owner: Actor, knowledge_id: str, files: dict[str, str]) -> None:
    with owner.client() as client:
        for filename, text in files.items():
            add_text_file(client, knowledge_id, filename, text)


def _open_base(page: Page, knowledge_id: str) -> Locator:
    page.goto(f"/workspace/knowledge/{knowledge_id}")
    base = page.get_by_role("main")
    expect(base.get_by_role("textbox", name="Knowledge Name")).to_be_visible()
    return base


def _file_row(base: Locator, filename: str) -> Locator:
    return base.get_by_role("listitem").filter(has_text=filename)


def _add_content(page: Page, base: Locator, item: str) -> None:
    # the menu's trigger wraps the labelled button
    base.get_by_role("button", name="Add Content").last.click()
    page.get_by_role("menu").get_by_role("button", name=item).click()


def _stored_file_names(owner: Actor, knowledge_id: str) -> list[str]:
    with owner.client() as client:
        listed = client.get(f"/api/v1/knowledge/{knowledge_id}/files")
    assert listed.status_code == 200, listed.text
    return sorted(item["filename"] for item in listed.json()["items"])


def _search_list(page: Page, name: str) -> None:
    page.goto("/workspace/knowledge")
    with page.expect_response(
        lambda response: "query=" in response.url and "page=1" in response.url
    ):
        page.get_by_role("textbox", name="Search Knowledge").fill(name)


def _list_row(page: Page, name: str) -> Locator:
    """The base's row in the list, once the list stopped loading further pages of matches."""
    _search_list(page, name)
    row = page.get_by_role("main").get_by_role("button").filter(has_text=name)
    expect(row).to_be_visible()
    # a next page loading re-renders the rows, which closes an open menu
    expect(page.get_by_text("Loading...")).to_have_count(0)
    return row


def test_a_knowledge_base_is_created_found_in_the_list_and_deleted(page_for, curator):
    name = _unique("Harbour")
    page = page_for(curator)
    page.goto("/workspace/knowledge/create")
    creating = page.get_by_role("dialog")
    creating.get_by_role("textbox", name="Name your knowledge base").fill(name)
    creating.get_by_role("textbox", name="Describe your knowledge base and objectives").fill(
        DESCRIPTION
    )
    creating.get_by_role("button", name="Create Knowledge").click()
    expect(page).to_have_url(re.compile(r"/workspace/knowledge/[0-9a-f-]+$"))
    knowledge_id = page.url.rsplit("/", 1)[-1]
    expect(page.get_by_role("textbox", name="Knowledge Name")).to_have_value(name)

    row = _list_row(page, name)
    expect(row).to_contain_text(DESCRIPTION)
    # the menu's trigger wraps the labelled button
    row.get_by_role("button", name="More Options").last.click()
    page.get_by_role("menu").get_by_role("button", name="Delete").click()
    page.get_by_role("dialog", name="Confirm your action").get_by_role(
        "button", name="Confirm"
    ).click()
    expect(page.get_by_text("Knowledge deleted successfully.")).to_be_visible()
    expect(row).to_have_count(0)

    with curator.client() as client:
        assert client.get(f"/api/v1/knowledge/{knowledge_id}").status_code != 200


def test_uploaded_files_and_pasted_text_are_listed_once_processed(page_for, curator):
    knowledge_id = _create_base(curator, _unique("Harbour"))
    page = page_for(curator)
    base = _open_base(page, knowledge_id)

    with page.expect_file_chooser() as chooser:
        _add_content(page, base, "Upload files")
    chooser.value.set_files(
        [
            {"name": "tides.txt", "mimeType": "text/plain", "buffer": b"High tide at 06:40.\n"},
            {"name": "berths.txt", "mimeType": "text/plain", "buffer": b"Berth 9 is free.\n"},
        ]
    )
    expect(base.get_by_text("2 files")).to_be_visible()

    _add_content(page, base, "Add text content")
    writing = page.get_by_role("dialog")
    writing.get_by_placeholder("Title").fill("office hours")
    writing.get_by_placeholder("Write something...").fill("The office opens at eight.")
    writing.get_by_role("button", name="Save").click()
    expect(base.get_by_text("3 files")).to_be_visible()

    for filename in ("tides.txt", "berths.txt", "office hours.txt"):
        expect(_file_row(base, filename)).to_have_count(1)
    assert _stored_file_names(curator, knowledge_id) == [
        "berths.txt",
        "office hours.txt",
        "tides.txt",
    ]


def _without_directory_picker(page: Page) -> None:
    # Playwright cannot drive Chromium's directory picker; the page then offers a file input
    page.add_init_script(
        "delete Window.prototype.showDirectoryPicker; delete window.showDirectoryPicker;"
    )


def test_an_uploaded_folder_keeps_its_subfolders(page_for, curator, tmp_path: Path):
    folder = tmp_path / "harbour"
    (folder / "moorings").mkdir(parents=True)
    (folder / "tides.txt").write_text("High tide at 06:40.\n")
    (folder / "moorings" / "berths.txt").write_text("Berth 9 is free.\n")
    knowledge_id = _create_base(curator, _unique("Harbour"))
    page = page_for(curator)
    _without_directory_picker(page)
    base = _open_base(page, knowledge_id)

    with page.expect_file_chooser() as chooser:
        _add_content(page, base, "Upload directory")
    chooser.value.set_files(str(folder))

    base.get_by_role("button", name="harbour").click()
    expect(_file_row(base, "tides.txt")).to_be_visible()
    base.get_by_role("button", name="moorings").click()
    expect(_file_row(base, "berths.txt")).to_be_visible()
    expect(_file_row(base, "tides.txt")).to_have_count(0)
    assert _stored_file_names(curator, knowledge_id) == ["berths.txt", "tides.txt"]


def test_opening_a_file_shows_its_text(page_for, curator):
    knowledge_id = _create_base(curator, _unique("Harbour"))
    _add_files(curator, knowledge_id, {"tides.txt": "High tide at 06:40, low tide at 12:55."})
    page = page_for(curator)
    base = _open_base(page, knowledge_id)

    _file_row(base, "tides.txt").get_by_role("button", name=re.compile("tides.txt")).click()

    expect(page.get_by_role("link", name="tides.txt")).to_be_visible()
    expect(page.get_by_role("textbox", name="File content")).to_have_value(
        "High tide at 06:40, low tide at 12:55."
    )


def test_searching_the_base_lists_only_matching_files(page_for, curator):
    knowledge_id = _create_base(curator, _unique("Harbour"))
    _add_files(
        curator,
        knowledge_id,
        {"tides.txt": "High tide.", "moorings.txt": "Berth 9.", "fuel.txt": "Cards only."},
    )
    page = page_for(curator)
    base = _open_base(page, knowledge_id)
    expect(base.get_by_text("3 files")).to_be_visible()

    base.get_by_role("textbox", name="Search Collection").fill("moor")

    expect(_file_row(base, "tides.txt")).to_have_count(0)
    expect(_file_row(base, "fuel.txt")).to_have_count(0)
    expect(_file_row(base, "moorings.txt")).to_be_visible()


def test_a_removed_file_is_no_longer_sent_to_a_chat(page_for, curator, upstream):
    knowledge_id = _create_base(curator, _unique("Harbour"))
    _add_files(
        curator,
        knowledge_id,
        {"gate.txt": "The gate code is 4242.", "fuel.txt": "The fuel dock takes cards only."},
    )
    page = page_for(curator)
    base = _open_base(page, knowledge_id)

    _file_row(base, "gate.txt").get_by_role("button").last.click()
    page.get_by_role("menu").get_by_role("button", name="Delete").click()
    expect(page.get_by_text("File removed successfully.")).to_be_visible()
    expect(_file_row(base, "gate.txt")).to_have_count(0)
    expect(_file_row(base, "fuel.txt")).to_be_visible()

    question = "what is the gate code?"
    upstream.queue(reply.text("I cannot tell.", match=reply.answering(question)))
    with curator.client() as client:
        ask(client, question, files=[{"type": "collection", "id": knowledge_id}])
    sent = json.dumps(next(filter(reply.answering(question), upstream.chat_requests())))
    assert "fuel dock takes cards" in sent
    assert "4242" not in sent


def _sent_to_the_model(owner: Actor, upstream, knowledge_id: str, question: str) -> str:
    upstream.queue(reply.text("Noted.", match=reply.answering(question)))
    with owner.client() as client:
        ask(client, question, files=[{"type": "collection", "id": knowledge_id}])
    return json.dumps(next(filter(reply.answering(question), upstream.chat_requests())))


def test_a_new_directory_holds_its_text_and_deleting_it_removes_that_text(
    page_for, curator, upstream
):
    name = _unique("Harbour")
    knowledge_id = _create_base(curator, name)
    _add_files(curator, knowledge_id, {"fuel.txt": "The fuel dock takes cards only."})
    page = page_for(curator)
    base = _open_base(page, knowledge_id)

    _add_content(page, base, "New directory")
    page.get_by_role("dialog").get_by_placeholder("Directory name").fill("moorings")
    page.get_by_role("dialog").get_by_role("button", name="Create").click()
    expect(page.get_by_text("Directory created.")).to_be_visible()
    page.reload()
    base = page.get_by_role("main")
    directory = base.get_by_role("button", name="moorings").last
    expect(directory).to_be_visible()

    directory.click()
    _add_content(page, base, "Add text content")
    writing = page.get_by_role("dialog")
    writing.get_by_placeholder("Title").fill("gate")
    writing.get_by_placeholder("Write something...").fill("The gate code is 4242.")
    writing.get_by_role("button", name="Save").click()
    expect(_file_row(base, "gate.txt")).to_be_visible()
    page.reload()
    base = page.get_by_role("main")
    expect(_file_row(base, "fuel.txt")).to_be_visible()
    expect(_file_row(base, "gate.txt")).to_have_count(0)
    base.get_by_role("button", name="moorings").last.click()
    expect(_file_row(base, "gate.txt")).to_be_visible()
    expect(_file_row(base, "fuel.txt")).to_have_count(0)

    # the root crumb carries the base's name
    base.get_by_role("button", name=name, exact=True).click()
    # a directory row is no list item, and its menu button has no label
    row = base.locator("div[draggable=true]").filter(has_text="moorings")
    row.get_by_role("button").last.click()
    page.get_by_role("menu").get_by_role("button", name="Delete").click()
    page.get_by_role("dialog", name="Delete directory?").get_by_role(
        "button", name="Confirm"
    ).click()
    expect(page.get_by_text("Directory deleted.")).to_be_visible()
    page.reload()
    base = page.get_by_role("main")
    expect(_file_row(base, "fuel.txt")).to_be_visible()
    expect(base.get_by_role("button", name="moorings")).to_have_count(0)

    assert _stored_file_names(curator, knowledge_id) == ["fuel.txt"]
    question = "what is the gate code?"
    sent = _sent_to_the_model(curator, upstream, knowledge_id, question)
    assert "fuel dock takes cards" in sent
    assert "4242" not in sent, "the chat is still sent the text of a deleted directory"


def _create_directory(owner: Actor, knowledge_id: str, name: str) -> str:
    with owner.client() as client:
        created = client.post(f"/api/v1/knowledge/{knowledge_id}/dirs/create", json={"name": name})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _file_in_directory(
    owner: Actor, knowledge_id: str, directory_id: str, filename: str, text: str
):
    _add_files(owner, knowledge_id, {filename: text})
    with owner.client() as client:
        listed = client.get(f"/api/v1/knowledge/{knowledge_id}/files", params={"directory_id": ""})
        file_id = next(i["id"] for i in listed.json()["items"] if i["filename"] == filename)
        moved = client.post(
            f"/api/v1/knowledge/{knowledge_id}/file/move",
            json={"file_id": file_id, "directory_id": directory_id},
        )
    assert moved.status_code == 200, moved.text


def _level(owner: Actor, knowledge_id: str, directory_id: str = "") -> tuple[list[str], list[str]]:
    """The directory names and file names listed at one level; the empty id is the top level."""
    with owner.client() as client:
        listed = client.get(
            f"/api/v1/knowledge/{knowledge_id}/files", params={"directory_id": directory_id}
        )
    assert listed.status_code == 200, listed.text
    body = listed.json()
    return (
        sorted(entry["name"] for entry in body["directories"]),
        sorted(entry["filename"] for entry in body["items"]),
    )


def _directory_row(base: Locator, name: str) -> Locator:
    # a directory row is no list item, and its menu button has no label
    return base.locator("div[draggable=true]").filter(has_text=name)


def test_a_renamed_directory_keeps_its_files_under_the_new_name(page_for, curator):
    knowledge_id = _create_base(curator, _unique("Harbour"))
    moorings = _create_directory(curator, knowledge_id, "moorings")
    _file_in_directory(curator, knowledge_id, moorings, "gate.txt", "The gate code is 4242.")
    page = page_for(curator)
    base = _open_base(page, knowledge_id)

    row = _directory_row(base, "moorings")
    row.focus()
    row.get_by_role("button").last.click()
    page.get_by_role("menu").get_by_role("button", name="Rename").click()
    # the row no longer shows its name as text while it is edited
    renaming = base.locator("div[draggable=true]").get_by_role("textbox")
    renaming.fill("pontoons")
    renaming.press("Enter")
    # Enter and the blur that follows both save, so the toast shows twice
    expect(page.get_by_text("Directory renamed.").first).to_be_visible()

    page.reload()
    base = page.get_by_role("main")
    expect(base.get_by_role("button", name="pontoons").last).to_be_visible()
    expect(base.get_by_role("button", name="moorings")).to_have_count(0)
    base.get_by_role("button", name="pontoons").last.click()
    expect(_file_row(base, "gate.txt")).to_be_visible()
    assert _level(curator, knowledge_id) == (["pontoons"], [])
    assert _level(curator, knowledge_id, moorings) == ([], ["gate.txt"])


def test_a_file_dragged_onto_a_directory_is_listed_inside_it(page_for, curator):
    knowledge_id = _create_base(curator, _unique("Harbour"))
    moorings = _create_directory(curator, knowledge_id, "moorings")
    _add_files(
        curator,
        knowledge_id,
        {"gate.txt": "The gate code is 4242.", "fuel.txt": "The fuel dock takes cards only."},
    )
    page = page_for(curator)
    base = _open_base(page, knowledge_id)
    expect(_file_row(base, "gate.txt")).to_be_visible()

    _file_row(base, "gate.txt").drag_to(_directory_row(base, "moorings"))
    expect(page.get_by_text("File moved.")).to_be_visible()

    page.reload()
    base = page.get_by_role("main")
    expect(_file_row(base, "fuel.txt")).to_be_visible()
    expect(_file_row(base, "gate.txt")).to_have_count(0)
    base.get_by_role("button", name="moorings").last.click()
    expect(_file_row(base, "gate.txt")).to_be_visible()
    expect(_file_row(base, "fuel.txt")).to_have_count(0)
    assert _level(curator, knowledge_id) == (["moorings"], ["fuel.txt"])
    assert _level(curator, knowledge_id, moorings) == ([], ["gate.txt"])


def test_a_directory_dragged_onto_another_shows_nested_with_its_files(page_for, curator):
    knowledge_id = _create_base(curator, _unique("Harbour"))
    moorings = _create_directory(curator, knowledge_id, "moorings")
    harbour = _create_directory(curator, knowledge_id, "quay")
    _file_in_directory(curator, knowledge_id, moorings, "gate.txt", "The gate code is 4242.")
    page = page_for(curator)
    base = _open_base(page, knowledge_id)
    expect(_directory_row(base, "quay")).to_be_visible()

    _directory_row(base, "moorings").drag_to(_directory_row(base, "quay"))
    expect(page.get_by_text("Directory moved.")).to_be_visible()

    page.reload()
    base = page.get_by_role("main")
    expect(_directory_row(base, "quay")).to_be_visible()
    expect(_directory_row(base, "moorings")).to_have_count(0)
    base.get_by_role("button", name="quay").last.click()
    expect(_directory_row(base, "moorings")).to_be_visible()
    base.get_by_role("button", name="moorings").last.click()
    expect(_file_row(base, "gate.txt")).to_be_visible()
    assert _level(curator, knowledge_id) == (["quay"], [])
    assert _level(curator, knowledge_id, harbour) == (["moorings"], [])
    assert _level(curator, knowledge_id, moorings) == ([], ["gate.txt"])


def test_deleting_a_directory_but_keeping_its_contents_moves_the_files_up(
    page_for, curator, upstream
):
    knowledge_id = _create_base(curator, _unique("Harbour"))
    moorings = _create_directory(curator, knowledge_id, "moorings")
    _add_files(curator, knowledge_id, {"fuel.txt": "The fuel dock takes cards only."})
    _file_in_directory(curator, knowledge_id, moorings, "gate.txt", "The gate code is 4242.")
    page = page_for(curator)
    base = _open_base(page, knowledge_id)

    row = _directory_row(base, "moorings")
    row.focus()
    row.get_by_role("button").last.click()
    page.get_by_role("menu").get_by_role("button", name="Delete").click()
    deleting = page.get_by_role("dialog", name="Delete directory?")
    # the checkbox has no label of its own, and it is the dialog's only one
    deleting.get_by_role("checkbox").uncheck()
    deleting.get_by_role("button", name="Confirm").click()
    expect(page.get_by_text("Directory deleted.")).to_be_visible()

    page.reload()
    base = page.get_by_role("main")
    expect(_file_row(base, "gate.txt")).to_be_visible()
    expect(_file_row(base, "fuel.txt")).to_be_visible()
    expect(base.get_by_role("button", name="moorings")).to_have_count(0)
    assert _level(curator, knowledge_id) == ([], ["fuel.txt", "gate.txt"])
    sent = _sent_to_the_model(curator, upstream, knowledge_id, "what is the gate code?")
    assert "4242" in sent, "the files kept from a deleted directory are no longer sent to a chat"


@pytest.fixture(scope="module")
def pages():
    with listening() as service:
        service.route("GET", "/tides", text_answer("<p>High tide at the harbour is at noon</p>"))
        yield service


@pytest.mark.slow
def test_a_webpage_added_by_its_link_is_listed_and_its_text_reaches_a_chat(
    page_for, instance_with, pages
):
    launched = instance_with(LOCAL_WEB_FETCH)
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    owner = admin_of(launched)
    knowledge_id = _create_base(owner, _unique("Harbour"))
    link = f"http://localhost:{pages.port}/tides"
    page = page_for(owner)
    base = _open_base(page, knowledge_id)

    _add_content(page, base, "Add webpage")
    page.get_by_role("textbox", name="Webpage URLs").fill(link)
    page.get_by_role("button", name="Add", exact=True).click()

    expect(page.get_by_text("File added successfully.")).to_be_visible(timeout=30_000)
    page.reload()
    base = page.get_by_role("main")
    expect(base.get_by_role("listitem")).to_have_count(1)
    expect(base.get_by_text("1 file")).to_be_visible()
    question = "when is high tide?"
    sent = _sent_to_the_model(owner, launched.upstream, knowledge_id, question)
    assert "High tide at the harbour is at noon" in sent
    _delete_bases_of(owner)


def test_an_admin_exports_a_base_as_a_zip_of_its_files_and_their_text(
    page_for, curator, tmp_path: Path
):
    name = _unique("Harbour")
    knowledge_id = _create_base(curator, name)
    _add_files(
        curator,
        knowledge_id,
        {"tides.txt": "High tide at 06:40, low tide at 12:55.", "fuel.txt": "Cards only."},
    )
    page = page_for(curator)

    row = _list_row(page, name)
    row.get_by_role("button", name="More Options").last.click()
    with page.expect_download() as download:
        page.get_by_role("menu").get_by_role("button", name="Export").click()
    saved = tmp_path / "export.zip"
    download.value.save_as(saved)

    expect(page.get_by_text("Knowledge exported successfully")).to_be_visible()
    assert download.value.suggested_filename == f"{name}.zip"
    with zipfile.ZipFile(io.BytesIO(saved.read_bytes())) as archive:
        exported = {entry: archive.read(entry).decode() for entry in archive.namelist()}
    assert exported == {
        "tides.txt": "High tide at 06:40, low tide at 12:55.",
        "fuel.txt": "Cards only.",
    }


@pytest.fixture
def team(admin, make_user) -> tuple[Actor, Actor, str]:
    """A group member and an outsider, both allowed the knowledge workspace, and the group name."""
    member, outsider = make_user(), make_user()
    group_id = make_group(admin, [member], KNOWLEDGE_USER)
    make_group(admin, [outsider], KNOWLEDGE_USER)
    with admin.client() as client:
        group = client.get(f"/api/v1/groups/id/{group_id}")
    assert group.status_code == 200, group.text
    return member, outsider, group.json()["name"]


def _share_with_group(page: Page, base: Locator, group_name: str) -> Locator:
    base.get_by_role("button", name="Access").click()
    dialog = page.get_by_role("dialog").filter(has_text="Access Control")
    dialog.get_by_role("button", name="Add Access").click()
    picker = page.get_by_role("dialog").filter(has_text="Add Access").last
    picker.get_by_placeholder("Search").fill(group_name)
    picker.get_by_role("button", name=group_name).click()
    picker.get_by_role("button", name="Add", exact=True).click()
    expect(page.get_by_text("Saved").first).to_be_visible()
    expect(dialog.get_by_role("combobox", name="Access level")).to_have_value("read")
    return dialog


def test_a_renamed_base_shows_its_new_name_to_a_group_member(page_for, curator, team):
    member, _, group_name = team
    knowledge_id = _create_base(curator, _unique("Harbour"))
    page = page_for(curator)
    base = _open_base(page, knowledge_id)
    _share_with_group(page, base, group_name)
    page.keyboard.press("Escape")

    new_name = _unique("Marina")
    base.get_by_role("textbox", name="Knowledge Name").fill(new_name)
    base.get_by_role("textbox", name="Knowledge Description").fill("pontoons and the fuel dock")
    expect(page.get_by_text("Knowledge updated successfully")).to_be_visible()

    member_page = page_for(member)
    row = _list_row(member_page, new_name)
    expect(row).to_contain_text("pontoons and the fuel dock")
    row.click()
    expect(member_page.get_by_role("textbox", name="Knowledge Name")).to_have_value(new_name)
    expect(member_page.get_by_role("textbox", name="Knowledge Description")).to_have_value(
        "pontoons and the fuel dock"
    )


def test_a_group_with_read_access_sees_the_base_read_only(page_for, curator, team):
    member, outsider, group_name = team
    name = _unique("Harbour")
    knowledge_id = _create_base(curator, name)
    _add_files(curator, knowledge_id, {"tides.txt": "High tide at 06:40."})
    page = page_for(curator)
    _share_with_group(page, _open_base(page, knowledge_id), group_name)

    member_page = page_for(member)
    expect(_list_row(member_page, name)).to_contain_text("Read Only")
    base = _open_base(member_page, knowledge_id)
    expect(base.get_by_role("textbox", name="Knowledge Name")).to_be_disabled()
    expect(base.get_by_role("textbox", name="Knowledge Description")).to_be_disabled()
    expect(base.get_by_text("Read Only")).to_be_visible()
    expect(base.get_by_role("button", name="Add Content")).to_have_count(0)
    _file_row(base, "tides.txt").get_by_role("button", name=re.compile("tides.txt")).click()
    content = member_page.get_by_role("textbox", name="File content")
    expect(content).to_have_value("High tide at 06:40.")
    expect(content).to_be_disabled()

    outsider_page = page_for(outsider)
    _search_list(outsider_page, name)
    expect(outsider_page.get_by_text("No knowledge found")).to_be_visible()


def test_granting_write_lets_a_group_member_edit_the_base(page_for, curator, team):
    member, _, group_name = team
    name = _unique("Harbour")
    knowledge_id = _create_base(curator, name)
    page = page_for(curator)
    dialog = _share_with_group(page, _open_base(page, knowledge_id), group_name)

    with page.expect_response(lambda response: "/access/update" in response.url) as saved:
        dialog.get_by_role("combobox", name="Access level").select_option("write")
    assert saved.value.ok

    member_page = page_for(member)
    expect(_list_row(member_page, name)).not_to_contain_text("Read Only")
    base = _open_base(member_page, knowledge_id)
    expect(base.get_by_role("textbox", name="Knowledge Name")).to_be_enabled()
    expect(base.get_by_role("button", name="Access")).to_be_visible()
    with member_page.expect_file_chooser() as chooser:
        _add_content(member_page, base, "Upload files")
    chooser.value.set_files(
        {"name": "fuel.txt", "mimeType": "text/plain", "buffer": b"Cards only.\n"}
    )
    expect(member_page.get_by_text("File added successfully.")).to_be_visible()
    expect(_file_row(base, "fuel.txt")).to_be_visible()
    assert _stored_file_names(curator, knowledge_id) == ["fuel.txt"]


def test_resetting_the_base_empties_it(page_for, curator):
    knowledge_id = _create_base(curator, _unique("Harbour"))
    _add_files(curator, knowledge_id, {"tides.txt": "High tide.", "fuel.txt": "Cards only."})
    page = page_for(curator)
    base = _open_base(page, knowledge_id)
    expect(base.get_by_text("2 files")).to_be_visible()

    _add_content(page, base, "Reset")
    page.get_by_role("dialog", name="Reset knowledge base?").get_by_role(
        "button", name="Confirm"
    ).click()

    expect(page.get_by_text("Knowledge base has been reset")).to_be_visible()
    expect(base.get_by_text("No content found")).to_be_visible()
    assert _stored_file_names(curator, knowledge_id) == []


def _sync_directory(page: Page, base: Locator, folder: Path, file_count: int) -> None:
    with page.expect_file_chooser() as chooser:
        _add_content(page, base, "Sync directory")
    chooser.value.set_files(str(folder))
    confirming = page.get_by_role("dialog", name="Confirm your action")
    expect(confirming).to_contain_text(f"{file_count} files selected.")
    confirming.get_by_role("button", name="Confirm").click()


def test_syncing_a_directory_mirrors_its_changes_into_the_base(
    page_for, curator, upstream, tmp_path: Path
):
    folder = tmp_path / "harbour"
    (folder / "moorings").mkdir(parents=True)
    (folder / "tides.txt").write_text("High tide at 06:40.\n")
    (folder / "fuel.txt").write_text("The fuel dock takes cards only.\n")
    (folder / "moorings" / "berths.txt").write_text("Berth 9 is free.\n")
    knowledge_id = _create_base(curator, _unique("Harbour"))
    page = page_for(curator)
    _without_directory_picker(page)
    base = _open_base(page, knowledge_id)

    _sync_directory(page, base, folder, 3)
    expect(
        page.get_by_text("Sync complete: 3 added, 0 modified, 0 deleted, 0 unmodified")
    ).to_be_visible()
    assert _stored_file_names(curator, knowledge_id) == ["berths.txt", "fuel.txt", "tides.txt"]

    (folder / "tides.txt").write_text("High tide at 07:15.\n")
    (folder / "fuel.txt").unlink()
    (folder / "gate.txt").write_text("The gate code is 4242.\n")
    _sync_directory(page, base, folder, 3)

    expect(
        page.get_by_text("Sync complete: 1 added, 1 modified, 1 deleted, 1 unmodified")
    ).to_be_visible()
    base.get_by_role("button", name="harbour").click()
    expect(_file_row(base, "gate.txt")).to_be_visible()
    expect(_file_row(base, "tides.txt")).to_have_count(1)
    expect(_file_row(base, "fuel.txt")).to_have_count(0)
    assert _stored_file_names(curator, knowledge_id) == ["berths.txt", "gate.txt", "tides.txt"]
    sent = _sent_to_the_model(curator, upstream, knowledge_id, "when is high tide?")
    assert "07:15" in sent and "4242" in sent, sent
    assert "06:40" not in sent, "the base still holds the text the sync replaced"
    assert "fuel dock" not in sent, "the base still holds a file deleted from the directory"


@pytest.fixture
def one_megabyte_uploads(admin, preserve):
    """A Max Upload Size of 1 MB on the shared instance, restored afterwards."""
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        saved = client.post(RETRIEVAL_CONFIG[1], json={"FILE_MAX_SIZE": 1})
    assert saved.status_code == 200, saved.text


LARGE_FILE = "x" * (2 * 1024 * 1024)


def test_a_file_over_the_max_upload_size_is_refused_with_the_limit_named(
    page_for, curator, one_megabyte_uploads
):
    knowledge_id = _create_base(curator, _unique("Harbour"))
    page = page_for(curator)
    base = _open_base(page, knowledge_id)

    with page.expect_file_chooser() as chooser:
        _add_content(page, base, "Upload files")
    chooser.value.set_files(
        [
            {"name": "atlas.txt", "mimeType": "text/plain", "buffer": LARGE_FILE.encode()},
            {"name": "tides.txt", "mimeType": "text/plain", "buffer": b"High tide at 06:40.\n"},
        ]
    )

    expect(page.get_by_text("File size should not exceed 1 MB.")).to_be_visible()
    expect(page.get_by_text("File added successfully.")).to_be_visible()
    expect(_file_row(base, "tides.txt")).to_be_visible()
    expect(_file_row(base, "atlas.txt")).to_have_count(0)
    assert _stored_file_names(curator, knowledge_id) == ["tides.txt"]


def test_a_directory_upload_reports_the_files_over_the_max_upload_size(
    page_for, curator, one_megabyte_uploads, tmp_path: Path
):
    folder = tmp_path / "harbour"
    folder.mkdir()
    (folder / "atlas.txt").write_text(LARGE_FILE)
    (folder / "tides.txt").write_text("High tide at 06:40.\n")
    knowledge_id = _create_base(curator, _unique("Harbour"))
    page = page_for(curator)
    _without_directory_picker(page)
    base = _open_base(page, knowledge_id)

    with page.expect_file_chooser() as chooser:
        _add_content(page, base, "Upload directory")
    chooser.value.set_files(str(folder))

    expect(page.get_by_text("Upload failed for 1 of 2 files.")).to_be_visible()
    assert _stored_file_names(curator, knowledge_id) == ["tides.txt"]
