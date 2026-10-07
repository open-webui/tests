"""Journey: what a folder's project settings do to the turns of the chats in it, and only those.

A folder's system prompt goes in front of every turn of a chat filed in it, the first and the
later ones, and in front of no chat outside it. A chat moved into the folder from the chat
header's Move menu takes the prompt from its next turn, and dragged back onto Chats in the sidebar
it loses it again. Nested folders do not merge their settings: a chat in a subfolder gets the
subfolder's prompt and not its parent's, and a subfolder dragged under another folder takes its
chats along and keeps its own prompt for them. The folder's knowledge base and uploaded file are
what the model's knowledge search reads in the folder, and the answer shows each as a citation
that opens on its text; the same search in a chat outside the folder never reads the folder's
file. Each turn is read from the request the scripted provider received.

Discriminates: passes on dev ebc6add67; in a backend copy with the middleware's folder lookup
returning no folder every test goes red (no prompt, no folder knowledge cited), and with the
folder's prompt taken from its top-level ancestor the two nested tests go red.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from harness.knowledge_bases import add_text_file, knowledge_base
from utils.chat_ui import chat_input, conversation, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

HARBOUR_PROMPT = "Answer as the harbour master of Port Ellen."
GARDEN_PROMPT = "Answer as the head gardener of Kew."
SEEDS_PROMPT = "Answer as the seed keeper of the vault."
MOORINGS = "Berth 4 is kept free for the lifeboat at all tides."
TIDES = "Spring tides at Port Ellen rise four metres above chart datum."


def create_folder(owner, name: str, system_prompt: str | None = None, parent_id=None) -> str:
    data = {"system_prompt": system_prompt} if system_prompt else None
    with owner.client() as client:
        created = client.post(
            "/api/v1/folders/", json={"name": name, "parent_id": parent_id, "data": data}
        )
        assert created.status_code == 200, created.text
        folder_id = created.json()["id"]
        expanded = client.post(
            f"/api/v1/folders/{folder_id}/update/expanded", json={"is_expanded": True}
        )
        assert expanded.status_code == 200, expanded.text
    return folder_id


def seeded_chat(owner, title: str, folder_id: str | None = None) -> str:
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": "ahoy"}, {"role": "assistant", "content": "ahoy there"}],
        )
        titled = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": title}})
        assert titled.status_code == 200, titled.text
        if folder_id:
            moved = client.post(f"/api/v1/chats/{chat_id}/folder", json={"folder_id": folder_id})
            assert moved.status_code == 200, moved.text
    return chat_id


def stored_folder_id(owner, chat_id: str) -> str | None:
    with owner.client() as client:
        found = client.get(f"/api/v1/chats/{chat_id}")
    assert found.status_code == 200, found.text
    return found.json()["folder_id"]


def ask(page: Page, upstream, prompt: str, answer: str) -> None:
    upstream.queue(reply.text(answer, match=reply.answering(prompt)))
    send(page, prompt)
    expect_reply(page, answer)


def system_sent(upstream, prompt: str) -> str:
    """The system text of the one provider request that answered `prompt`."""
    [request] = [body for body in upstream.chat_requests() if reply.answering(prompt)(body)]
    return "\n".join(
        str(message["content"]) for message in request["messages"] if message["role"] == "system"
    )


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


def folder_content(sidebar: Locator, folder_id: str) -> Locator:
    """The folder's row with everything filed under it."""
    row = sidebar.locator(f"#folder-{folder_id}-button")
    return row.locator("xpath=ancestor::div[@draggable][1]")


def drag(page: Page, source: Locator, target: Locator) -> None:
    # a synthetic drag: a mouse drag across the sidebar rows sometimes never fires the drop
    transfer = page.evaluate_handle("() => new DataTransfer()")
    source.dispatch_event("dragstart", {"dataTransfer": transfer})
    target.dispatch_event("dragover", {"dataTransfer": transfer})
    target.dispatch_event("drop", {"dataTransfer": transfer})
    source.dispatch_event("dragend", {"dataTransfer": transfer})


def test_the_folder_prompt_leads_every_turn_in_the_folder_and_no_chat_outside(
    page_for, make_user, upstream
):
    owner = make_user()
    folder_id = create_folder(owner, "Harbour", HARBOUR_PROMPT)
    page = page_for(owner)
    page.goto(f"/folders/{folder_id}")

    ask(page, upstream, "when is high tide?", "High tide at noon.")
    expect(page).to_have_url(re.compile("/c/"))
    ask(page, upstream, "and low tide?", "Low tide at six.")
    page.goto("/")
    ask(page, upstream, "what is for lunch?", "Soup.")

    assert system_sent(upstream, "when is high tide?").startswith(HARBOUR_PROMPT)
    assert system_sent(upstream, "and low tide?").startswith(HARBOUR_PROMPT), (
        "the folder's prompt was left off the chat's second turn"
    )
    assert HARBOUR_PROMPT not in system_sent(upstream, "what is for lunch?"), (
        "a chat outside the folder was sent the folder's prompt"
    )


def test_a_chat_moved_in_takes_the_prompt_and_dragged_out_loses_it(page_for, make_user, upstream):
    owner = make_user()
    folder_id = create_folder(owner, "Harbour", HARBOUR_PROMPT)
    title = f"Loose ends {uuid.uuid4().hex[:6]}"
    chat_id = seeded_chat(owner, title)
    page = page_for(owner)
    page.goto(f"/c/{chat_id}")
    ask(page, upstream, "before the move?", "Still loose.")

    page.get_by_label("Chat actions").click()
    page.get_by_role("menu").get_by_role("button", name="Move").hover()
    page.get_by_role("menu").get_by_role("button", name="Harbour").click()
    expect(page.get_by_text("Chat moved successfully")).to_be_visible()
    assert stored_folder_id(owner, chat_id) == folder_id
    ask(page, upstream, "after the move?", "Filed now.")

    sidebar = open_sidebar(page)
    filed = folder_content(sidebar, folder_id).get_by_role("button", name=title)
    expect(filed).to_be_visible()
    with page.expect_response(re.compile(f"/api/v1/chats/{chat_id}/folder")):
        drag(page, filed, sidebar.get_by_role("button", name="Chats", exact=True))
    assert stored_folder_id(owner, chat_id) is None
    ask(page, upstream, "after leaving?", "Loose again.")

    assert HARBOUR_PROMPT not in system_sent(upstream, "before the move?")
    assert system_sent(upstream, "after the move?").startswith(HARBOUR_PROMPT), (
        "a chat moved into the folder was not sent its prompt on the next turn"
    )
    assert HARBOUR_PROMPT not in system_sent(upstream, "after leaving?"), (
        "a chat dragged out of the folder was still sent its prompt"
    )


def test_a_subfolder_chat_gets_the_subfolders_prompt_and_not_its_parents(
    page_for, make_user, upstream
):
    owner = make_user()
    garden_id = create_folder(owner, "Garden", GARDEN_PROMPT)
    seeds_id = create_folder(owner, "Seeds", SEEDS_PROMPT, parent_id=garden_id)
    page = page_for(owner)

    page.goto(f"/folders/{seeds_id}")
    ask(page, upstream, "which beans keep longest?", "Broad beans.")
    page.goto(f"/folders/{garden_id}")
    ask(page, upstream, "when to prune roses?", "In March.")

    in_seeds = system_sent(upstream, "which beans keep longest?")
    assert in_seeds.startswith(SEEDS_PROMPT), in_seeds
    assert GARDEN_PROMPT not in in_seeds, "the parent folder's prompt reached the subfolder chat"
    in_garden = system_sent(upstream, "when to prune roses?")
    assert in_garden.startswith(GARDEN_PROMPT), in_garden
    assert SEEDS_PROMPT not in in_garden


def test_a_subfolder_dragged_under_another_folder_keeps_its_chats_and_its_prompt(
    page_for, make_user, upstream
):
    owner = make_user()
    garden_id = create_folder(owner, "Garden", GARDEN_PROMPT)
    seeds_id = create_folder(owner, "Seeds", SEEDS_PROMPT, parent_id=garden_id)
    archive_id = create_folder(owner, "Archive")
    title = f"Bean trials {uuid.uuid4().hex[:6]}"
    chat_id = seeded_chat(owner, title, seeds_id)
    page = page_for(owner)
    page.goto("/")
    sidebar = open_sidebar(page)
    seeds_row = sidebar.get_by_role("button", name="Seeds", exact=True)
    expect(seeds_row).to_be_visible()

    with page.expect_response(lambda response: response.url.endswith("/update/parent")):
        drag(page, seeds_row, sidebar.get_by_role("button", name="Archive", exact=True))

    archive = folder_content(sidebar, archive_id)
    expect(archive.get_by_role("button", name="Seeds", exact=True)).to_be_visible()
    expect(folder_content(sidebar, garden_id).get_by_role("button", name="Seeds")).to_have_count(0)
    with owner.client() as client:
        assert client.get(f"/api/v1/folders/{seeds_id}").json()["parent_id"] == archive_id
    page.reload()
    expect(archive.get_by_role("button", name=title)).to_be_visible()

    archive.get_by_role("button", name=title).click()
    expect(page).to_have_url(re.compile(f"/c/{chat_id}"))
    ask(page, upstream, "still sowing beans?", "Every spring.")
    after_move = system_sent(upstream, "still sowing beans?")
    assert after_move.startswith(SEEDS_PROMPT), after_move
    assert GARDEN_PROMPT not in after_move


@pytest.fixture
def harbour_knowledge(admin, make_user):
    """An account with a folder holding a knowledge base and an uploaded file of its own.

    Yields the account, the folder id, the knowledge base's file name and the folder file's name.
    """
    owner = make_user()
    tag = uuid.uuid4().hex[:8]
    tides_name, moorings_name = f"tides-{tag}.txt", f"moorings-{tag}.txt"
    readable = [{"principal_type": "user", "principal_id": owner.id, "permission": "read"}]
    with admin.client() as client, knowledge_base(client, f"Tides {tag}", readable) as base_id:
        add_text_file(client, base_id, tides_name, TIDES)
        with owner.client() as client:
            uploaded = client.post(
                "/api/v1/files/",
                params={"process": "true", "process_in_background": "false"},
                files={"file": (moorings_name, MOORINGS.encode(), "text/plain")},
            )
            assert uploaded.status_code == 200, uploaded.text
            files = [
                {"type": "collection", "id": base_id, "name": f"Tides {tag}"},
                {"type": "file", "id": uploaded.json()["id"], "name": moorings_name},
            ]
            created = client.post(
                "/api/v1/folders/", json={"name": "Harbour", "data": {"files": files}}
            )
            assert created.status_code == 200, created.text
        yield owner, created.json()["id"], tides_name, moorings_name


def search_and_answer(page: Page, upstream, prompt: str, answer: str) -> None:
    upstream.queue(
        reply.tool_call("query_knowledge_files", {"query": "berth"}, match=reply.answering(prompt)),
        reply.text(answer, match=reply.answering(prompt)),
    )
    send(page, prompt)
    expect_reply(page, answer)


def search_results(upstream, prompt: str) -> str:
    """What the knowledge search handed back to the model for `prompt`."""
    return "\n".join(
        str(message["content"])
        for body in upstream.chat_requests()
        if reply.answering(prompt)(body)
        for message in body["messages"]
        if message["role"] == "tool"
    )


def open_citation(page: Page, name: str) -> Locator:
    conversation(page).get_by_role("button", name=f"View source: {name}").click()
    citation = page.get_by_role("dialog")
    expect(citation).to_be_visible()
    return citation


def test_the_folders_knowledge_and_file_are_cited_in_its_chats_only(
    page_for, harbour_knowledge, upstream
):
    owner, folder_id, tides_name, moorings_name = harbour_knowledge
    page = page_for(owner)
    page.goto(f"/folders/{folder_id}")

    search_and_answer(page, upstream, "where may I moor?", "Not at berth 4.")
    sources = conversation(page).get_by_role("button", name=re.compile(r"^Toggle \d+ sources?$"))
    sources.click()
    citation = open_citation(page, moorings_name)
    expect(citation).to_contain_text(MOORINGS)
    page.keyboard.press("Escape")
    expect(citation).to_be_hidden()
    expect(open_citation(page, tides_name)).to_contain_text(TIDES)
    page.keyboard.press("Escape")

    page.goto("/")
    search_and_answer(page, upstream, "where may I moor, outside?", "No idea.")
    assert MOORINGS in search_results(upstream, "where may I moor?")
    assert MOORINGS not in search_results(upstream, "where may I moor, outside?"), (
        "the knowledge search outside the folder read the folder's file"
    )
