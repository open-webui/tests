"""Journey: the model browses, searches and reads knowledge bases and the files of a chat.

A chat on a model with no knowledge attached is offered tools over every knowledge base the
account can read: list and search the bases, find and grep their files, read a file by lines or
characters and query the chunks. A model with knowledge attached gets `list_knowledge` and the
scoped search in their place, and still shows a knowledge base only to accounts that may read
it; an attached note comes back whole. On a model that reads attached files through tools
(`file_context` off) the chat's own files are listed, grepped and queried, and a file the
account cannot read is left out even when the chat names it.

The scripted model calls each tool and the test reads the result it was sent back.

Discriminates: in a backend copy, `view_knowledge_file` skipping its access check turned the
stranger read red; `grep_knowledge_files` searching without the account filter turned the grep
test red; `list_knowledge` listing attached bases without the read check turned the attached
stranger test red; `search_knowledge_files` skipping the attachment check turned the attached
search test red; `_get_accessible_chat_files` skipping `_has_read_access_to_file` turned the
foreign chat file test red; and the knowledge toggle ignored turned the toggle test red.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from typing import Iterator

import pytest

from harness.knowledge_bases import add_text_file, knowledge_base
from harness.python_tools import EVERYONE_READS
from harness.tool_calls import offered_tools, run_tool
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

HERON_LINES = [
    "Grey herons nest in colonies.",
    "They stand still for minutes.",
    "A grey heron can live twenty years.",
    "Nests are reused every spring.",
]
LEDGER_TEXT = "heron budget: 4000\nsecret line"
KNOWLEDGE_TOOLS = {
    "list_knowledge_bases",
    "search_knowledge_bases",
    "query_knowledge_bases",
    "grep_knowledge_files",
    "search_knowledge_files",
    "query_knowledge_files",
    "view_knowledge_file",
}


def call(actor, upstream, tool: str, model: str = MOCK_MODEL_ID, files=None, **arguments):
    """The tool result the model got for `actor` in a chat with `files`, parsed when JSON."""
    chat = {"model": model, **({"files": files} if files else {})}
    with actor.client() as client:
        result = run_tool(client, upstream, tool, arguments, **chat)
    try:
        return json.loads(result)
    except ValueError:
        return result


@pytest.fixture
def library(admin, make_user):
    """A field guide of two files shared with a reader, and a ledger nobody was granted.

    The admin owns both, so the reader sees the guide only through its grant; names are unique
    per test.
    """
    reader, stranger = make_user(), make_user()
    tag = uuid.uuid4().hex[:8]
    reader_grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
    with (
        admin.client() as client,
        knowledge_base(client, f"Field guide {tag}", [reader_grant]) as guide_id,
        knowledge_base(client, f"Ledger {tag}") as ledger_id,
    ):
        client.post(
            f"/api/v1/knowledge/{guide_id}/update",
            json={
                "name": f"Field guide {tag}",
                "description": f"wading birds {tag}",
                "access_grants": [reader_grant],
            },
        ).raise_for_status()
        herons = add_text_file(client, guide_id, f"herons-{tag}.md", "\n".join(HERON_LINES))
        grebes = add_text_file(client, guide_id, f"grebes-{tag}.txt", "Grebes dive.")
        ledger = add_text_file(client, ledger_id, f"ledger-{tag}.md", LEDGER_TEXT)
        yield {
            "reader": reader,
            "stranger": stranger,
            "tag": tag,
            "guide_id": guide_id,
            "ledger_id": ledger_id,
            "herons": herons,
            "grebes": grebes,
            "ledger": ledger,
        }


# --- no knowledge attached: every readable knowledge base -----------------------------------


def test_the_knowledge_bases_listed_are_the_readable_ones(library, upstream):
    listed = call(library["reader"], upstream, "list_knowledge_bases", count=100)

    by_id = {entry["id"]: entry for entry in listed}
    assert library["guide_id"] in by_id and library["ledger_id"] not in by_id, listed
    assert by_id[library["guide_id"]]["file_count"] == 2
    assert by_id[library["guide_id"]]["description"] == f"wading birds {library['tag']}"


def test_a_knowledge_base_is_found_by_its_description(library, upstream):
    found = call(library["reader"], upstream, "search_knowledge_bases", query=library["tag"])
    stranger_found = call(
        library["stranger"], upstream, "search_knowledge_bases", query="wading birds"
    )

    assert [entry["id"] for entry in found] == [library["guide_id"]]
    assert library["guide_id"] not in {entry["id"] for entry in stranger_found}


def test_files_are_found_by_name_within_readable_knowledge_bases(library, upstream):
    reader, tag = library["reader"], library["tag"]

    found = call(reader, upstream, "search_knowledge_files", query=f"herons-{tag}")
    scoped = call(
        reader, upstream, "search_knowledge_files", query=tag, knowledge_id=library["guide_id"]
    )
    refused = call(
        reader, upstream, "search_knowledge_files", query=tag, knowledge_id=library["ledger_id"]
    )

    assert [(entry["id"], entry["knowledge_id"]) for entry in found] == [
        (library["herons"], library["guide_id"])
    ]
    assert {entry["id"] for entry in scoped} == {library["herons"], library["grebes"]}
    assert refused == {"error": f"Access denied to knowledge base {library['ledger_id']}"}


def test_grep_searches_only_readable_files(library, upstream):
    reader = library["reader"]

    hits = call(reader, upstream, "grep_knowledge_files", pattern="heron").splitlines()
    counted = call(
        reader,
        upstream,
        "grep_knowledge_files",
        pattern="GREY",
        case_insensitive=True,
        count_only=True,
    )

    assert [hit.split("  ", 1)[1] for hit in hits] == [
        f"herons-{library['tag']}.md:1: Grey herons nest in colonies.",
        f"herons-{library['tag']}.md:3: A grey heron can live twenty years.",
    ], f"grep reached a file the account cannot read: {hits}"
    assert counted.splitlines()[-1] == "[2 total matches]"


def test_a_knowledge_file_is_read_by_lines_and_by_characters(library, upstream):
    reader, herons = library["reader"], library["herons"]

    by_lines = call(
        reader, upstream, "view_knowledge_file", file_id=herons, start_line=2, end_line=3
    )
    first_page = call(reader, upstream, "view_knowledge_file", file_id=herons, max_chars=10)
    next_page = call(
        reader, upstream, "view_knowledge_file", file_id=herons, offset=10, max_chars=10
    )

    assert (
        by_lines["content"]
        == "2: They stand still for minutes.\n3: A grey heron can live twenty years."
    )
    assert (by_lines["total_lines"], by_lines["next_start_line"]) == (4, 4)
    assert by_lines["knowledge_id"] == library["guide_id"]
    assert first_page["content"] + next_page["content"] == "\n".join(HERON_LINES)[:20]
    assert (first_page["next_offset"], next_page["offset"]) == (10, 10)


def test_a_stranger_cannot_read_a_knowledge_file(library, upstream):
    refused = call(library["stranger"], upstream, "view_knowledge_file", file_id=library["herons"])

    assert refused == {"error": "Access denied"}, f"a stranger read a knowledge file: {refused}"


def test_querying_returns_chunks_of_readable_knowledge_bases_only(library, upstream):
    reader = library["reader"]

    chunks = call(reader, upstream, "query_knowledge_files", query="heron", count=10)
    of_ledger = call(
        reader,
        upstream,
        "query_knowledge_files",
        query="heron",
        knowledge_ids=[library["ledger_id"]],
    )

    sources = {chunk["file_id"] for chunk in chunks}
    assert {library["herons"], library["grebes"]} <= sources, chunks
    assert library["ledger"] not in sources
    assert of_ledger == [], f"a foreign knowledge base was queried: {of_ledger}"


def test_knowledge_bases_are_ranked_among_readable_ones_only(library, upstream):
    ranked = call(library["stranger"], upstream, "query_knowledge_bases", query="wading birds")

    ids = {entry["id"] for entry in ranked}
    assert library["guide_id"] not in ids, f"a foreign knowledge base was ranked: {ranked}"


# --- knowledge attached to the model --------------------------------------------------------


@contextlib.contextmanager
def preset(admin, knowledge: list[dict], **meta) -> Iterator[str]:
    """A model every account may chat with, carrying `knowledge` and `meta`; yields its id."""
    model_id = f"preset-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": model_id,
        "meta": {"knowledge": knowledge, **meta},
        "params": {},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        try:
            yield model_id
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def attached(library, admin):
    """A preset with the field guide and a note of the reader's attached."""
    with library["reader"].client() as client:
        note = client.post(
            "/api/v1/notes/create",
            json={"title": "Lake note", "data": {"content": {"md": "herons at dusk"}}},
        )
    assert note.status_code == 200, note.text
    knowledge = [
        {"type": "collection", "id": library["guide_id"], "name": "Field guide"},
        {"type": "note", "id": note.json()["id"], "name": "Lake note"},
    ]
    with preset(admin, knowledge) as model_id:
        yield {**library, "model": model_id, "note_id": note.json()["id"]}


def test_list_knowledge_shows_what_is_attached(attached, upstream):
    reader, model = attached["reader"], attached["model"]

    summary = call(reader, upstream, "list_knowledge", model=model)
    detail = call(
        reader, upstream, "list_knowledge", model=model, knowledge_id=attached["guide_id"], count=1
    )

    assert [entry["id"] for entry in summary["knowledge_bases"]] == [attached["guide_id"]]
    assert "files" not in summary["knowledge_bases"][0]
    assert summary["notes"] == [{"id": attached["note_id"], "title": "Lake note"}]
    listing = detail["knowledge_bases"][0]
    assert (listing["files_count"], listing["files_total"], listing["has_more"]) == (1, 2, True)


def test_an_attached_knowledge_base_stays_hidden_from_a_stranger(attached, upstream):
    listed = call(attached["stranger"], upstream, "list_knowledge", model=attached["model"])

    assert listed == {"knowledge_bases": [], "files": [], "notes": []}, (
        f"a stranger saw knowledge they cannot read through the model: {listed}"
    )


def test_the_attached_search_stays_within_the_attachment(attached, upstream):
    reader, model = attached["reader"], attached["model"]

    found = call(reader, upstream, "search_knowledge_files", model=model, query="grebes")
    outside = call(
        reader,
        upstream,
        "search_knowledge_files",
        model=model,
        query="x",
        knowledge_id=attached["ledger_id"],
    )

    assert [entry["id"] for entry in found] == [attached["grebes"]]
    assert outside == {
        "error": f"Knowledge base {attached['ledger_id']} is not attached to this model"
    }


def test_an_attached_note_is_returned_whole_with_the_chunks(attached, upstream):
    chunks = call(
        attached["reader"],
        upstream,
        "query_knowledge_files",
        model=attached["model"],
        query="herons",
    )

    assert chunks[0] == {
        "content": "herons at dusk",
        "source": "Lake note",
        "note_id": attached["note_id"],
        "type": "note",
    }
    assert {chunk.get("file_id") for chunk in chunks[1:]} <= {
        attached["herons"],
        attached["grebes"],
    }


def test_the_knowledge_toggle_withdraws_the_knowledge_tools(admin, make_user, upstream):
    account = make_user()
    with preset(admin, [], builtinTools={"knowledge": False}) as model_id:
        with account.client() as client:
            without = offered_tools(client, upstream, model=model_id)
            with_them = offered_tools(client, upstream)

    assert not KNOWLEDGE_TOOLS & without, f"the knowledge toggle is off, yet: {without}"
    assert KNOWLEDGE_TOOLS <= with_them


# --- files attached to the chat -------------------------------------------------------------


@pytest.fixture
def file_reader(admin):
    """A model that reads a chat's files through tools rather than having them in context."""
    capabilities = {"file_upload": True, "file_context": False}
    with preset(admin, [], capabilities=capabilities) as model_id:
        yield model_id


def chat_files(*file_ids: str) -> list[dict]:
    return [{"type": "file", "id": file_id, "name": file_id} for file_id in file_ids]


def test_the_chats_files_are_listed_grepped_and_queried(library, file_reader, upstream):
    reader, herons = library["reader"], library["herons"]
    files = chat_files(herons)

    listed = call(reader, upstream, "list_chat_files", model=file_reader, files=files)
    grepped = call(
        reader, upstream, "grep_chat_files", model=file_reader, files=files, pattern="nest"
    )
    queried = call(
        reader, upstream, "query_chat_files", model=file_reader, files=files, query="heron"
    )
    unattached = call(
        reader,
        upstream,
        "grep_chat_files",
        model=file_reader,
        files=files,
        pattern="x",
        file_id=library["grebes"],
    )

    assert [(entry["id"], entry["filename"]) for entry in listed] == [
        (herons, f"herons-{library['tag']}.md")
    ]
    assert grepped == f"{herons}  herons-{library['tag']}.md:1: Grey herons nest in colonies."
    assert {chunk["file_id"] for chunk in queried} == {herons}
    assert unattached == {"error": "File not found"}


def test_a_chat_naming_someone_elses_file_cannot_read_it(library, file_reader, upstream):
    files = chat_files(library["ledger"])

    listed = call(library["reader"], upstream, "list_chat_files", model=file_reader, files=files)
    grepped = call(
        library["reader"],
        upstream,
        "grep_chat_files",
        model=file_reader,
        files=files,
        pattern="secret",
    )

    assert listed == [], f"a foreign file was listed as the chat's own: {listed}"
    assert grepped == {"error": "No accessible files found"}
