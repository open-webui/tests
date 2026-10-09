"""Journey: browsing a knowledge base's files and opening them in the file viewer.

The base's page lists its files in a side panel next to a viewer. Sorting by name puts the files
in name order and the direction button reverses it; a base of more files than one page shows a
Load more button that lists the rest. A search lists matching files from every folder with the
folder each one sits in, and with Search file content on it also finds a file by its text. Opening
a text file previews it and its Indexed text tab shows what the server indexed. A TypeScript file,
which browsers upload labelled as video, previews as its code, and an HTML file previews without
running its scripts. A PDF is not driven here: the viewer's PDF renderer draws no pages in the
suite's Chromium for any PDF. A file whose stored upload is gone opens on its indexed text, without
a Preview tab, and still downloads as that text. The owner edits the indexed text and saves it;
when the embedding provider fails the save shows the error and keeps the edit open, and leaving
an unsaved edit for another file asks first. A file uploaded through the Add Content menu,
removed from its row's menu and uploaded again is listed again. A reader of a shared base opens
its files but sees no Add Content menu, no Edit button and no Rename or Remove entries in a
row's menu.

Discriminates: passes on the dev 550311b21 build. In a frontend build whose sort menu ignores the
chosen key, whose Load more button loads nothing, whose search rows leave out the folder path,
whose Search file content switch changes nothing, whose viewer never fetches the indexed text,
whose HTML preview may run scripts, whose viewer shows an error instead of falling back to the
indexed text when the original is gone, whose save swallows a failed save, whose file switch
never asks about unsaved edits, whose remove entry removes nothing, that offers Edit to readers
and that previews a video-labelled code file as video (96ea26d09 undone), the matching test
fails; the indexed text mutation also fails the owner's save test, which then has nothing to edit.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import grant, make_group
from harness.actors import Actor
from harness.knowledge_bases import add_text_file
from integration.retrieval.test_knowledge_file_browser import (  # noqa: F401 (fixture)
    create_folder,
    failing_embeddings,
    indexed_text,
    move_file,
    stored_upload,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

KNOWLEDGE_USER = {"workspace": {"knowledge": True}}


@pytest.fixture
def curator(make_user):
    """A fresh admin, whose knowledge bases are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for knowledge in client.get("/api/v1/knowledge/").json().get("items", []):
            if knowledge["user_id"] == account.id:
                client.delete(f"/api/v1/knowledge/{knowledge['id']}/delete")


def create_base(owner: Actor, grants: list[dict] | None = None) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/knowledge/create",
            json={
                "name": f"Harbour {uuid.uuid4().hex[:6]}",
                "description": "",
                "access_grants": grants or [],
            },
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def add_files(owner: Actor, knowledge_id: str, files: dict[str, str]) -> dict[str, str]:
    with owner.client() as client:
        return {
            name: add_text_file(client, knowledge_id, name, text) for name, text in files.items()
        }


def add_upload(owner: Actor, knowledge_id: str, name: str, mime: str, data: bytes) -> str:
    with owner.client() as client:
        uploaded = client.post(
            "/api/v1/files/",
            params={"process": "true", "process_in_background": "false"},
            files={"file": (name, data, mime)},
        )
        assert uploaded.status_code == 200, uploaded.text
        file_id = uploaded.json()["id"]
        added = client.post(f"/api/v1/knowledge/{knowledge_id}/file/add", json={"file_id": file_id})
    assert added.status_code == 200, added.text
    return file_id


def open_base(page: Page, knowledge_id: str) -> Page:
    page.goto(f"/workspace/knowledge/{knowledge_id}")
    expect(page.get_by_role("textbox", name="Search Collection")).to_be_visible()
    return page


def entry(page: Page, name: str) -> Locator:
    return page.get_by_role("button", name=re.compile(rf"^{re.escape(name)}"))


def listed_names(page: Page) -> list[str]:
    return [
        text.split()[0] for text in page.locator("button[data-knowledge-entry]").all_inner_texts()
    ]


def sort_menu(page: Page) -> Locator:
    # the menu's trigger wraps the labelled button
    page.get_by_role("button", name="Filter and sort").last.click()
    return page.get_by_role("menu")


def search(page: Page, text: str) -> None:
    page.get_by_role("textbox", name="Search Collection").fill(text)


def open_file(page: Page, name: str) -> None:
    entry(page, name).click()
    expect(page.get_by_role("button", name="Indexed text")).to_be_visible()


def test_sorting_by_name_lists_the_files_in_name_order_both_ways(page_for, curator):
    knowledge_id = create_base(curator)
    add_files(curator, knowledge_id, {"buoys.txt": "b", "charts.txt": "c", "anchors.txt": "a"})
    page = open_base(page_for(curator), knowledge_id)
    expect(entry(page, "anchors.txt")).to_be_visible()

    menu = sort_menu(page)
    menu.get_by_role("menuitemradio", name=re.compile("Name$")).click()
    expect(page.locator("button[data-knowledge-entry]").first).to_have_text(
        re.compile(r"^\s*charts\.txt")
    )
    assert listed_names(page) == ["charts.txt", "buoys.txt", "anchors.txt"]

    menu.get_by_role("button", name="Descending").click()
    expect(page.locator("button[data-knowledge-entry]").first).to_have_text(
        re.compile(r"^\s*anchors\.txt")
    )
    assert listed_names(page) == ["anchors.txt", "buoys.txt", "charts.txt"]


def test_a_base_of_more_than_one_page_loads_the_rest_on_request(page_for, curator):
    knowledge_id = create_base(curator)
    add_files(
        curator,
        knowledge_id,
        {f"berth-{number:02}.txt": f"Berth {number}." for number in range(35)},
    )
    page = open_base(page_for(curator), knowledge_id)
    rows = page.locator("button[data-knowledge-entry]")
    expect(rows).to_have_count(30)

    page.get_by_role("button", name="Load more").click()

    expect(rows).to_have_count(35)
    expect(page.get_by_role("button", name="Load more")).to_have_count(0)
    assert sorted(listed_names(page)) == [f"berth-{number:02}.txt" for number in range(35)]


def test_a_search_lists_files_from_every_folder_with_their_folder(page_for, curator):
    knowledge_id = create_base(curator)
    files = add_files(
        curator, knowledge_id, {"berth-plan.txt": "Berth four.", "berth-fees.txt": "Ten a night."}
    )
    with curator.client() as client:
        harbour = create_folder(client, knowledge_id, "Harbour")
        berths = create_folder(client, knowledge_id, "Berths", harbour)
        assert move_file(client, knowledge_id, files["berth-plan.txt"], berths).status_code == 200
    page = open_base(page_for(curator), knowledge_id)
    expect(entry(page, "berth-plan.txt")).to_have_count(0)  # folded away inside Harbour

    search(page, "berth")

    expect(page.get_by_text("Search results")).to_be_visible()
    expect(entry(page, "berth-plan.txt")).to_contain_text("Harbour/Berths")
    expect(entry(page, "berth-fees.txt")).to_be_visible()
    expect(entry(page, "berth-fees.txt")).not_to_contain_text("Harbour")


def test_searching_file_content_finds_a_file_by_its_text(page_for, curator):
    knowledge_id = create_base(curator)
    add_files(
        curator,
        knowledge_id,
        {"notes.txt": "Drop anchor in the outer anchorage.", "fees.txt": "Ten a night."},
    )
    page = open_base(page_for(curator), knowledge_id)

    search(page, "anchorage")
    expect(page.get_by_text("No results found")).to_be_visible()

    sort_menu(page).get_by_role("menuitemcheckbox", name=re.compile("Search file content")).click()
    expect(entry(page, "notes.txt")).to_be_visible()
    expect(entry(page, "fees.txt")).to_have_count(0)


def test_a_text_file_previews_and_shows_its_indexed_text(page_for, curator):
    knowledge_id = create_base(curator)
    add_files(curator, knowledge_id, {"tides.txt": "High tide at noon."})
    page = open_base(page_for(curator), knowledge_id)
    expect(page.get_by_text("Select a file to preview")).to_be_visible()

    open_file(page, "tides.txt")
    expect(page.get_by_role("button", name="Preview")).to_have_attribute("aria-pressed", "true")
    expect(page.get_by_text("High tide at noon.")).to_be_visible()

    with page.expect_response(lambda response: response.url.endswith("/data/content")) as indexed:
        page.get_by_role("button", name="Indexed text").click()
    assert indexed.value.ok
    expect(page.get_by_role("button", name="Indexed text")).to_have_attribute(
        "aria-pressed", "true"
    )
    expect(page.get_by_text("High tide at noon.")).to_be_visible()


def test_a_typescript_file_the_browser_calls_video_previews_as_its_code(page_for, curator):
    knowledge_id = create_base(curator)
    # browsers label .ts uploads as MPEG transport stream video
    add_upload(curator, knowledge_id, "berths.ts", "video/mp2t", b"export const berths = 12;\n")
    page = open_base(page_for(curator), knowledge_id)

    open_file(page, "berths.ts")

    expect(page.get_by_role("textbox").filter(has_text="export const berths = 12;")).to_be_visible()
    expect(page.locator("video")).to_have_count(0)


def test_an_html_file_previews_without_running_its_scripts(page_for, curator):
    knowledge_id = create_base(curator)
    html = b"<p>Departures board</p><script>document.body.dataset.ran = 'yes';</script>"
    add_upload(curator, knowledge_id, "board.html", "text/html", html)
    page = open_base(page_for(curator), knowledge_id)

    open_file(page, "board.html")

    frame = page.locator("iframe[title='HTML Preview']")
    expect(frame).to_be_visible()
    expect(
        page.frame_locator("iframe[title='HTML Preview']").get_by_text("Departures board")
    ).to_be_visible()
    assert "allow-scripts" not in (frame.get_attribute("sandbox") or "")
    expect(
        page.frame_locator("iframe[title='HTML Preview']").locator("body[data-ran]")
    ).to_have_count(0)


def test_a_file_whose_upload_is_gone_opens_on_its_indexed_text(
    page_for, curator, instance, tmp_path: Path
):
    knowledge_id = create_base(curator)
    files = add_files(curator, knowledge_id, {"tides.txt": "High tide at noon."})
    stored_upload(instance, files["tides.txt"]).unlink()
    page = open_base(page_for(curator), knowledge_id)

    entry(page, "tides.txt").click()

    expect(page.get_by_role("button", name="Indexed text")).to_have_attribute(
        "aria-pressed", "true"
    )
    expect(page.get_by_text("High tide at noon.")).to_be_visible()
    expect(page.get_by_role("button", name="Preview")).to_have_count(0)
    expect(page.get_by_role("alert")).to_have_count(0)

    page.get_by_role("button", name="File actions").last.click()
    with page.expect_download() as download:
        page.get_by_role("menu").get_by_role("button", name="Download").click()
    saved = tmp_path / "download"
    download.value.save_as(saved)
    assert saved.read_text() == "High tide at noon."


def edit_indexed_text(page: Page, text: str) -> None:
    page.get_by_role("button", name="Edit").click()
    editor = page.get_by_role("textbox").filter(has_text="High tide at noon.")
    editor.click()
    page.keyboard.press("ControlOrMeta+a")
    page.keyboard.type(text)


def test_the_owner_edits_and_saves_the_indexed_text(page_for, curator):
    knowledge_id = create_base(curator)
    files = add_files(curator, knowledge_id, {"tides.txt": "High tide at noon."})
    page = open_base(page_for(curator), knowledge_id)
    open_file(page, "tides.txt")

    edit_indexed_text(page, "High tide at one.")
    page.get_by_role("button", name="Save").click()

    expect(page.get_by_text("File content updated successfully.")).to_be_visible()
    expect(page.get_by_role("button", name="Edit")).to_be_visible()
    with curator.client() as client:
        assert indexed_text(client, files["tides.txt"]) == "High tide at one."


def test_a_save_while_embedding_fails_shows_the_error_and_keeps_the_edit(
    page_for, curator, request
):
    knowledge_id = create_base(curator)
    add_files(curator, knowledge_id, {"tides.txt": "High tide at noon."})
    request.getfixturevalue("failing_embeddings")
    page = open_base(page_for(curator), knowledge_id)
    open_file(page, "tides.txt")
    edit_indexed_text(page, "High tide at one.")

    page.get_by_role("button", name="Save").click()

    expect(page.get_by_text("not fully indexed")).to_be_visible()
    expect(page.get_by_text("File content updated successfully.")).to_have_count(0)
    expect(page.get_by_role("button", name="Save")).to_be_visible()


def test_leaving_an_unsaved_edit_for_another_file_asks_first(page_for, curator):
    knowledge_id = create_base(curator)
    add_files(
        curator, knowledge_id, {"tides.txt": "High tide at noon.", "fees.txt": "Ten a night."}
    )
    page = open_base(page_for(curator), knowledge_id)
    open_file(page, "tides.txt")
    edit_indexed_text(page, "High tide at one.")

    entry(page, "fees.txt").click()
    dialog = page.get_by_role("dialog").filter(has_text="Discard unsaved changes?")
    expect(dialog).to_be_visible()
    dialog.get_by_role("button", name="Cancel").click()
    expect(page.get_by_text("High tide at one.")).to_be_visible()

    entry(page, "fees.txt").click()
    page.get_by_role("dialog").filter(has_text="Discard unsaved changes?").get_by_role(
        "button", name="Confirm"
    ).click()
    expect(page.get_by_text("Ten a night.")).to_be_visible()


def stored_names(owner: Actor, knowledge_id: str, expected: list[str]) -> list[str]:
    """The base's file names once they match `expected`, or as they stand after ten seconds."""
    deadline = time.monotonic() + 10
    while True:
        with owner.client() as client:
            listed = client.get(f"/api/v1/knowledge/{knowledge_id}/files")
        assert listed.status_code == 200, listed.text
        names = sorted(item["filename"] for item in listed.json()["items"])
        if names == expected or time.monotonic() > deadline:
            return names
        time.sleep(0.5)


def upload_through_menu(page: Page, name: str, text: bytes) -> None:
    page.get_by_role("button", name="Add Content").last.click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("button", name="Upload files").click()
    chooser.value.set_files({"name": name, "mimeType": "text/plain", "buffer": text})


def test_a_file_removed_from_the_base_can_be_uploaded_again(page_for, curator):
    knowledge_id = create_base(curator)
    page = open_base(page_for(curator), knowledge_id)

    upload_through_menu(page, "tides.txt", b"High tide at noon.\n")
    expect(entry(page, "tides.txt")).to_be_visible()
    expect(entry(page, "tides.txt")).not_to_contain_text("Processing")
    assert stored_names(curator, knowledge_id, ["tides.txt"]) == ["tides.txt"]

    page.locator("[data-knowledge-row]").filter(has=entry(page, "tides.txt")).get_by_role(
        "button", name="More"
    ).last.click()
    page.get_by_role("menu").get_by_role("button", name="Remove from knowledge").click()
    page.get_by_role("dialog").get_by_role("button", name="Confirm").click()
    expect(entry(page, "tides.txt")).to_have_count(0)
    assert stored_names(curator, knowledge_id, []) == []

    upload_through_menu(page, "tides.txt", b"High tide at noon.\n")
    expect(entry(page, "tides.txt")).to_be_visible()
    expect(entry(page, "tides.txt")).not_to_contain_text("Processing")
    assert stored_names(curator, knowledge_id, ["tides.txt"]) == ["tides.txt"]


def test_a_reader_opens_files_but_is_offered_no_changes(page_for, curator, admin, make_user):
    reader = make_user()
    make_group(admin, [reader], KNOWLEDGE_USER)
    knowledge_id = create_base(curator, [grant("user", reader.id, "read")])
    add_files(curator, knowledge_id, {"tides.txt": "High tide at noon."})
    page = open_base(page_for(reader), knowledge_id)

    open_file(page, "tides.txt")
    expect(page.get_by_text("High tide at noon.")).to_be_visible()
    expect(page.get_by_role("button", name="Edit")).to_have_count(0)
    expect(page.get_by_role("button", name="Add Content")).to_have_count(0)

    page.locator("[data-knowledge-row]").filter(has=entry(page, "tides.txt")).get_by_role(
        "button", name="More"
    ).last.click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("button", name="Download")).to_be_visible()
    expect(menu.get_by_role("button", name="Rename")).to_have_count(0)
    expect(menu.get_by_role("button", name="Remove from knowledge")).to_have_count(0)
