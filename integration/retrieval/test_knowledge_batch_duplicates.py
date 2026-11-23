"""Regressions: batch add to a knowledge base differs from the single add.

Duplicates (PR #31336, issue #31333):

`POST /api/v1/knowledge/{id}/files/batch/add` linked a file whose extracted text was already in the
knowledge base under another file, where the single add answers "Duplicate content detected", so the
same passages were embedded twice. The batch now runs the single add's check, also against the
earlier files of the same call, and reports a duplicate as a failed file while the rest still go
through. Chunks it writes carry the content hash, so a later single add sees them too.

Discriminates: passes on dev 015dbc861. In a backend copy, `process_files_batch` without the hash
check and without the hash in the chunk metadata (ecd0ff67e reverted) fails every duplicate case
here (both files linked); the re-add and distinct-content cases pass either way.

Pages (issue open-webui/open-webui#34876, fix branch
Classic298:fix/knowledge-batch-add-keeps-chunks): the single add copies the file's own chunks
into the knowledge base, each with the PDF page it came from, but the batch re-split the file's
plain text as one document. Its chunks ran across pages and had no page, so a chat on that
knowledge base gave the model no page to cite (`RAG_SOURCE_METADATA_KEYS`) and the citations none
to show. The page case fails on dev d599f7a08 (one chunk with all three pages and no page) and
passes on the fix branch. The file is renamed before it is added, and the citations must show
the new name; that check also passes with the fix's name override dropped, since citations take
the name from the file record.
"""

from __future__ import annotations

import json
import re
import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.actors import admin_of
from harness.chat import ask
from harness.file_context import file_context_ticked
from harness.knowledge_bases import knowledge_base
from harness.sample_documents import pdf

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def _upload(client: httpx.Client, text: str) -> str:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (f"{uuid.uuid4().hex[:8]}.txt", text.encode(), "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]


def _text() -> str:
    return f"quarterly harbour report {uuid.uuid4().hex}"


def _batch_add(client: httpx.Client, knowledge_id: str, *file_ids: str) -> dict:
    added = client.post(
        f"/api/v1/knowledge/{knowledge_id}/files/batch/add",
        json=[{"file_id": file_id} for file_id in file_ids],
    )
    assert added.status_code == 200, added.text
    return added.json()


def _add(client: httpx.Client, knowledge_id: str, file_id: str) -> httpx.Response:
    return client.post(f"/api/v1/knowledge/{knowledge_id}/file/add", json={"file_id": file_id})


def _linked(client: httpx.Client, knowledge_id: str) -> list[str]:
    listed = client.get(f"/api/v1/knowledge/{knowledge_id}/files")
    assert listed.status_code == 200, listed.text
    return [item["id"] for item in listed.json()["items"]]


@pytest.fixture
def base(admin):
    with admin.client() as client, knowledge_base(client) as knowledge_id:
        yield client, knowledge_id


def test_a_batch_refuses_content_already_in_the_knowledge_base(base):
    client, knowledge_id = base
    text = _text()
    first, copy = _upload(client, text), _upload(client, text)
    assert _add(client, knowledge_id, first).status_code == 200

    _batch_add(client, knowledge_id, copy)

    assert _linked(client, knowledge_id) == [first], "the batch linked duplicate content (#31336)"


def test_a_batch_links_one_of_two_files_with_the_same_text(base):
    client, knowledge_id = base
    text = _text()
    first, copy = _upload(client, text), _upload(client, text)

    _batch_add(client, knowledge_id, first, copy)

    # the batch is read in database order, so either file may be the one kept
    assert len(_linked(client, knowledge_id)) == 1


def test_a_batch_sibling_of_a_duplicate_still_goes_through(base):
    client, knowledge_id = base
    text = _text()
    first, copy, other = _upload(client, text), _upload(client, text), _upload(client, _text())

    _batch_add(client, knowledge_id, first, copy, other)

    linked = _linked(client, knowledge_id)
    assert other in linked and len(linked) == 2


def test_a_single_add_refuses_content_a_batch_added(base):
    client, knowledge_id = base
    text = _text()
    first, copy = _upload(client, text), _upload(client, text)
    _batch_add(client, knowledge_id, first)

    refused = _add(client, knowledge_id, copy)

    assert refused.status_code == 400, refused.text
    assert "Duplicate content" in refused.text
    assert _linked(client, knowledge_id) == [first]


def test_a_batch_accepts_the_same_file_again_and_files_with_distinct_text(base):
    client, knowledge_id = base
    first, second = _upload(client, _text()), _upload(client, _text())
    _batch_add(client, knowledge_id, first)

    again = _batch_add(client, knowledge_id, first, second)

    assert sorted(_linked(client, knowledge_id)) == sorted([first, second])
    assert not again.get("warnings")


@pytest.fixture(scope="module")
def pages_shown(instance_with):
    """An instance that gives the model each source chunk's page."""
    return instance_with({"RAG_SOURCE_METADATA_KEYS": "page"})


def _chunks(client: httpx.Client, knowledge_id: str) -> list[tuple]:
    """(page, page label, text) of every chunk the knowledge base is searched by."""
    queried = client.post(
        "/api/v1/retrieval/query/collection",
        json={"collection_names": [knowledge_id], "query": "harbour", "k": 10},
    )
    assert queried.status_code == 200, queried.text
    result = queried.json()
    return sorted(
        (metadata.get("page"), metadata.get("page_label"), text)
        for text, metadata in zip(result["documents"][0], result["metadatas"][0])
    )


@pytest.mark.slow
def test_a_batch_keeps_the_pdf_pages_the_single_add_keeps(pages_shown):
    tag = uuid.uuid4().hex[:8]
    pages = [f"harbour {tag} on page {word}" for word in ("one", "two", "three")]
    admin = admin_of(pages_shown)
    with (
        admin.client() as client,
        knowledge_base(client, "Single") as single,
        knowledge_base(client, "Batch") as batch,
    ):
        uploaded = client.post(
            "/api/v1/files/",
            params={"process": "true", "process_in_background": "false"},
            files={"file": ("survey.pdf", pdf(pages), "application/pdf")},
        )
        assert uploaded.status_code == 200, uploaded.text
        file_id = uploaded.json()["id"]
        renamed = client.post(f"/api/v1/files/{file_id}/rename", json={"filename": "report.pdf"})
        assert renamed.status_code == 200, renamed.text
        assert _add(client, single, file_id).status_code == 200
        assert not _batch_add(client, batch, file_id).get("warnings")

        single_chunks = _chunks(client, single)
        assert single_chunks == [(index, str(index + 1), pages[index]) for index in range(3)], (
            f"the single add no longer keeps one chunk per PDF page: {single_chunks}"
        )
        batch_chunks = _chunks(client, batch)
        assert batch_chunks == single_chunks, (
            f"the batch gave the knowledge base {batch_chunks}, where the single add keeps one "
            f"chunk per page with its page: {single_chunks} (#34876)"
        )

        prompt = f"Which page mentions harbour {tag}?"
        with file_context_ticked(admin):
            pages_shown.upstream.queue(reply.text("page 2", match=reply.answering(prompt)))
            _, message = ask(client, prompt, files=[{"type": "collection", "id": batch}])
    assert message.get("done") and not message.get("error"), message

    sent = [
        json.dumps(body["messages"])
        for body in pages_shown.upstream.chat_requests()
        if prompt in json.dumps(body["messages"])
    ]
    assert sent, "the chat never reached the model"
    shown = sorted(int(page) for page in re.findall(r'<source [^>]*page=\\"(\d+)\\"', sent[-1]))
    assert shown == [1, 2, 3], (
        f"the model was given pages {shown} for a batch-added PDF with one line on each of three "
        "pages, so RAG_SOURCE_METADATA_KEYS gives it nothing to cite (#34876)"
    )
    cited = sorted(
        metadata.get("page")
        for source in message.get("sources") or []
        for metadata in source.get("metadata") or []
        if metadata.get("file_id") == file_id
    )
    assert cited == [0, 1, 2], (
        f"the reply's citations carry pages {cited}, not one per page (#34876)"
    )
    names = {
        metadata.get("name")
        for source in message.get("sources") or []
        for metadata in source.get("metadata") or []
    }
    assert names == {"report.pdf"}, f"the citations name {names} for a file renamed to report.pdf"
