"""Journey: an admin connects a Qdrant or Milvus collection in Integrations and a chat uses it.

The admin adds a knowledge connection in Admin Settings > Integrations, which can only be saved
after its test query returned points, reloads to see it listed, edits it without retyping the
API key and switches it off. A chat that attaches the resulting knowledge base with `#` is sent
the points Qdrant found; once the connection is switched off the chat is sent none. A local
service plays Qdrant, so the tests read the query it got and the context the model was sent.
The reply lists a point whose source is a web address by its title, and its title and Content
links open that page; a point with a plain source shows its name as text (open-webui#32231).
The settings offer no way to delete a connection or its source. For Milvus the admin picks the
provider, which fills the column names Milvus needs (a vector field among them, without which the
test does not run), names a database and adds the source; a chat that attaches it is sent the rows
a local gRPC service played as Milvus found.

Discriminates: passes on dev 176d31d1d; in a frontend copy, with the source form sending an
empty API key on an edit the edit test fails (Qdrant gets no key), with the Create button enabled
before a test passed the create and the no-points tests fail, and with the switch sending the
old enabled state the switch test fails (it stays on). In a backend copy, with the retrieval of
an external knowledge base skipped the chat tests fail (Qdrant and Milvus are never queried), and
with the Milvus search leaving out the database name the Milvus add test fails. In a frontend copy,
with the test form no longer asking for a vector field on Milvus the no-vector-field test fails.
On dev 5032e834e the Qdrant chat test passes; it fails in a frontend build with ca3a6cf26
(open-webui#32232) reverted (the source is listed by its address) and in a backend copy with
1369027db (open-webui#32233) reverted (the title links to `/api/v1/files/21/content`).
"""

from __future__ import annotations

import json
import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.external_knowledge import (
    CONNECTIONS,
    EMBEDDING,
    MILVUS_SOURCE_CONFIG,
    MILVUS_TOKEN,
    QDRANT_API_KEY,
    external_connection,
    external_knowledge_base,
    milvus_row,
    point,
    queries_to,
    searches_of,
    serve_milvus,
    serve_qdrant,
)
from utils.chat_ui import chat_input, conversation, expect_reply, listed_source, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

BOATHOUSE_PAGE = "https://harbour.example/boathouse"
POINTS = [
    point(
        21,
        "The boathouse key hangs behind the oar rack.",
        0.93,
        source=BOATHOUSE_PAGE,
        name="Boathouse rules",
    ),
    point(22, "Oars are counted every Friday.", 0.74, source="oars.md"),
]
QUESTION = "where is the boathouse key?"
MILVUS_ROWS = [
    milvus_row("row-a", "The chandlery opens at six.", [0.3, -0.2, 0.1], source="chandlery.md"),
    milvus_row("row-b", "The fuel dock takes cards only.", EMBEDDING, source="fuel.md"),
]
FUEL_QUESTION = "how do I pay at the fuel dock?"


@pytest.fixture
def curator(make_user):
    """A fresh admin, whose external knowledge is deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for knowledge in client.get(
            "/api/v1/knowledge/search", params={"source": "external"}
        ).json()["items"]:
            if knowledge["user_id"] == account.id:
                client.delete(f"/api/v1/knowledge/{knowledge['id']}/delete")
        for connection in client.get(CONNECTIONS).json()["items"]:
            if connection["created_by"] == account.id:
                client.delete(f"{CONNECTIONS}/{connection['id']}")


def _open_integrations(page: Page) -> Locator:
    """The Integrations tab of the admin settings, freshly loaded from the server."""
    page.goto("/admin/settings/integrations")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_text("External Knowledge Sources", exact=True)).to_be_visible()
    return settings


def _add_button(settings: Locator) -> Locator:
    # the plus button has no accessible name, so it is found next to the section title
    title = settings.get_by_text("External Knowledge Sources", exact=True)
    return title.locator("xpath=ancestor::div[.//button][1]").get_by_role("button")


def _row(settings: Locator, name: str) -> Locator:
    # the innermost block holding both the name and its Configure button
    return (
        settings.locator("div")
        .filter(has=settings.page.get_by_text(name))
        .filter(has=settings.page.get_by_role("button", name="Configure"))
        .last
    )


def _source_form(page: Page, title: str) -> Locator:
    return page.get_by_role("dialog").filter(has=page.get_by_role("heading", name=title))


def _fill_new_source(form: Locator, name: str, endpoint: str, collection: str, query: str) -> None:
    form.get_by_label("Name", exact=True).fill(name)
    form.get_by_label("Endpoint").fill(endpoint)
    form.get_by_label("API Key / Token").fill(QDRANT_API_KEY)
    form.get_by_label("Collection").fill(collection)
    form.get_by_label("Test Query").fill(query)


def test_a_source_is_added_once_its_test_query_passes_and_survives_a_reload(
    page_for, curator, listener
):
    name = f"Boathouse {uuid.uuid4().hex[:6]}"
    collection = "boathouse_docs"
    endpoint = serve_qdrant(listener, collection, POINTS)["endpoint"]
    page = page_for(curator)
    settings = _open_integrations(page)
    _add_button(settings).click()
    form = _source_form(page, "Add Knowledge Connection")
    _fill_new_source(form, name, endpoint, collection, QUESTION)
    expect(form.get_by_role("button", name="Create")).to_be_disabled()

    form.get_by_role("button", name="Verify Connection").click()
    expect(page.get_by_text("Test succeeded.").first).to_be_visible()
    form.get_by_role("button", name="Create").click()
    expect(page.get_by_text("Knowledge source created.").first).to_be_visible()

    settings = _open_integrations(page)
    expect(settings.get_by_text(name)).to_be_visible()
    expect(settings.get_by_text(f"qdrant · {collection}")).to_be_visible()
    # the button test and the create each query Qdrant, both with the key
    assert [query["limit"] for query in queries_to(listener, collection)] == [5, 5]
    calls = listener.requests_to(f"/collections/{collection}/points/query")
    assert {call.headers.get("api-key") for call in calls} == {QDRANT_API_KEY}
    with curator.client() as client:
        [saved] = [
            connection
            for connection in client.get(CONNECTIONS).json()["items"]
            if connection["endpoint"] == endpoint
        ]
    assert saved["auth_configured"] is True and saved["enabled"] is True


def test_a_source_whose_test_query_finds_nothing_cannot_be_created(page_for, curator, listener):
    collection = "empty_docs"
    endpoint = serve_qdrant(listener, collection, [])["endpoint"]
    page = page_for(curator)
    _add_button(_open_integrations(page)).click()
    form = _source_form(page, "Add Knowledge Connection")
    _fill_new_source(form, f"Empty {uuid.uuid4().hex[:6]}", endpoint, collection, QUESTION)

    form.get_by_role("button", name="Verify Connection").click()

    expect(page.get_by_text("Test returned no results.").first).to_be_visible()
    expect(form.get_by_role("button", name="Create")).to_be_disabled()
    assert len(queries_to(listener, collection)) == 1


def test_an_edited_source_is_saved_under_its_new_name_with_the_key_kept(
    page_for, curator, listener
):
    collection = "boathouse_docs"
    name = f"Boathouse {uuid.uuid4().hex[:6]}"
    renamed = f"Slipway {uuid.uuid4().hex[:6]}"
    form_data = serve_qdrant(listener, collection, POINTS)
    page = page_for(curator)
    with (
        curator.client() as client,
        external_connection(client, form_data) as connection_id,
        external_knowledge_base(client, connection_id, collection, name),
    ):
        settings = _open_integrations(page)
        row = _row(settings, name)
        row.get_by_role("button", name="Configure").click()
        form = _source_form(page, "Edit Knowledge Connection")
        expect(form.get_by_label("Name", exact=True)).to_have_value(name)
        expect(form.get_by_label("Endpoint")).to_have_value(form_data["endpoint"])
        expect(form.get_by_label("Collection")).to_have_value(collection)
        expect(form.get_by_label("API Key / Token")).to_have_value("")
        form.get_by_label("Name", exact=True).fill(renamed)
        form.get_by_label("Test Query").fill(QUESTION)
        form.get_by_role("button", name="Verify Connection").click()
        expect(page.get_by_text("Test succeeded.").first).to_be_visible()
        form.get_by_role("button", name="Save").click()
        expect(page.get_by_text("Knowledge source updated.").first).to_be_visible()

        settings = _open_integrations(page)
        expect(settings.get_by_text(renamed)).to_be_visible()
        expect(settings.get_by_text(name)).to_have_count(0)

    calls = listener.requests_to(f"/collections/{collection}/points/query")
    assert len(calls) == 2, "the button test and the save each query Qdrant"
    assert {call.headers.get("api-key") for call in calls} == {QDRANT_API_KEY}


def _attach_and_ask(page: Page, knowledge_name: str, prefix: str) -> None:
    page.goto("/")
    chat_input(page).click()
    page.keyboard.type(f"#{prefix}")
    page.get_by_role("tooltip").get_by_role("button", name=knowledge_name).click()
    send(page, QUESTION)


@pytest.mark.usefixtures("file_context_on")
def test_a_chat_with_the_source_attached_is_sent_the_points_qdrant_found(
    page_for, curator, listener, upstream
):
    collection = "boathouse_docs"
    name = f"Boathouse {uuid.uuid4().hex[:6]}"
    form_data = serve_qdrant(listener, collection, POINTS)
    upstream.queue(reply.text("Behind the oar rack.", match=reply.answering(QUESTION)))
    page = page_for(curator)
    with (
        curator.client() as client,
        external_connection(client, form_data) as connection_id,
        external_knowledge_base(client, connection_id, collection, name),
    ):
        _attach_and_ask(page, name, "Boathouse")
        expect_reply(page, "Behind the oar rack.")

    [query] = queries_to(listener, collection)
    assert query["query"] == {"nearest": [0.1, 0.2, 0.3]}
    [chat_request] = [body for body in upstream.chat_requests() if reply.answering(QUESTION)(body)]
    sent = json.dumps(chat_request["messages"])
    assert "The boathouse key hangs behind the oar rack." in sent
    assert "Oars are counted every Friday." in sent

    # a web address source is listed by its title and links to that page, a plain one is text
    chat = conversation(page)
    chat.get_by_role("button", name="Toggle 2 sources").click()
    listed = listed_source(page, "Boathouse rules")
    expect(listed).to_be_visible()
    listed.click()
    citation = page.get_by_role("dialog")
    expect(citation.get_by_role("link", name="Boathouse rules")).to_have_attribute(
        "href", BOATHOUSE_PAGE
    )
    content_link = citation.get_by_role("link", name="Content")
    expect(content_link).to_have_attribute("href", re.compile(f"^{re.escape(BOATHOUSE_PAGE)}"))
    citation.get_by_role("button", name="Close citation modal").click()
    listed_source(page, "oars.md").click()
    expect(citation.get_by_text("oars.md", exact=True)).to_be_visible()
    expect(citation.get_by_role("link")).to_have_count(0)


def test_a_switched_off_source_stays_off_after_a_reload_and_a_chat_gets_no_points(
    page_for, curator, listener, upstream
):
    collection = "boathouse_docs"
    name = f"Boathouse {uuid.uuid4().hex[:6]}"
    form_data = serve_qdrant(listener, collection, POINTS)
    upstream.queue(reply.text("I do not know.", match=reply.answering(QUESTION)))
    page = page_for(curator)
    with (
        curator.client() as client,
        external_connection(client, form_data) as connection_id,
        external_knowledge_base(client, connection_id, collection, name),
    ):
        settings = _open_integrations(page)
        row = _row(settings, name)
        row.get_by_role("switch").click()
        expect(row.get_by_role("switch")).not_to_be_checked()

        settings = _open_integrations(page)
        row = _row(settings, name)
        expect(row.get_by_role("switch")).not_to_be_checked()
        assert client.get(f"{CONNECTIONS}/{connection_id}").json()["enabled"] is False

        _attach_and_ask(page, name, "Boathouse")
        expect_reply(page, "I do not know.")

    assert queries_to(listener, collection) == []
    [chat_request] = [body for body in upstream.chat_requests() if reply.answering(QUESTION)(body)]
    assert "oar rack" not in json.dumps(chat_request["messages"])


def test_a_milvus_source_is_added_with_the_columns_and_database_it_was_given(
    page_for, curator, milvus_service
):
    name = f"Fuelquay {uuid.uuid4().hex[:6]}"
    collection = "fuel_docs"
    endpoint = serve_milvus(milvus_service, collection, MILVUS_ROWS)["endpoint"]
    page = page_for(curator)
    _add_button(_open_integrations(page)).click()
    form = _source_form(page, "Add Knowledge Connection")
    form.get_by_label("Provider").select_option("milvus")
    expect(form.get_by_label("Content Field")).to_have_value("data.text")
    expect(form.get_by_label("Vector Field")).to_have_value("vector")
    expect(form.get_by_label("Metadata Field")).to_have_value("metadata")
    form.get_by_label("Name", exact=True).fill(name)
    form.get_by_label("Endpoint").fill(endpoint)
    form.get_by_label("API Key / Token").fill(MILVUS_TOKEN)
    form.get_by_label("Database").fill("harbour")
    form.get_by_label("Collection").fill(collection)
    form.get_by_label("Test Query").fill(FUEL_QUESTION)
    expect(form.get_by_role("button", name="Create")).to_be_disabled()

    form.get_by_role("button", name="Verify Connection").click()
    expect(page.get_by_text("Test succeeded.").first).to_be_visible()
    form.get_by_role("button", name="Create").click()
    expect(page.get_by_text("Knowledge source created.").first).to_be_visible()

    settings = _open_integrations(page)
    expect(settings.get_by_text(name)).to_be_visible()
    expect(settings.get_by_text(f"milvus · {collection}")).to_be_visible()
    # the button test and the create each search Milvus, on the vector column
    assert [search[1:3] for search in searches_of(milvus_service, collection)] == [
        ("vector", 5),
        ("vector", 5),
    ]
    assert any(call.get("dbname") == "harbour" for call in milvus_service.call_metadata)
    with curator.client() as client:
        [saved] = [
            connection
            for connection in client.get(CONNECTIONS).json()["items"]
            if connection["endpoint"] == endpoint
        ]
    assert saved["provider"] == "milvus" and saved["config"]["db_name"] == "harbour"
    assert saved["auth_configured"] is True


def test_a_milvus_source_without_a_vector_field_is_not_tried(page_for, curator, milvus_service):
    collection = "fuel_docs"
    endpoint = serve_milvus(milvus_service, collection, MILVUS_ROWS)["endpoint"]
    page = page_for(curator)
    _add_button(_open_integrations(page)).click()
    form = _source_form(page, "Add Knowledge Connection")
    form.get_by_label("Provider").select_option("milvus")
    form.get_by_label("Name", exact=True).fill(f"Fuel dock {uuid.uuid4().hex[:6]}")
    form.get_by_label("Endpoint").fill(endpoint)
    form.get_by_label("Collection").fill(collection)
    form.get_by_label("Test Query").fill(FUEL_QUESTION)
    form.get_by_label("Vector Field").fill("")

    form.get_by_role("button", name="Verify Connection").click()

    expect(page.get_by_text("Fill the source fields and test query first.").first).to_be_visible()
    expect(form.get_by_role("button", name="Create")).to_be_disabled()
    assert searches_of(milvus_service, collection) == []


@pytest.mark.usefixtures("file_context_on")
def test_a_chat_with_the_milvus_source_attached_is_sent_the_rows_milvus_found(
    page_for, curator, milvus_service, upstream
):
    collection = "fuel_docs"
    name = f"Fuelquay {uuid.uuid4().hex[:6]}"
    form_data = serve_milvus(milvus_service, collection, MILVUS_ROWS)
    upstream.queue(reply.text("Cards only.", match=reply.answering(FUEL_QUESTION)))
    page = page_for(curator)
    with (
        curator.client() as client,
        external_connection(client, form_data) as connection_id,
        external_knowledge_base(client, connection_id, collection, name, MILVUS_SOURCE_CONFIG),
    ):
        page.goto("/")
        chat_input(page).click()
        page.keyboard.type("#Fuelquay")
        page.get_by_role("tooltip").get_by_role("button", name=name).click()
        send(page, FUEL_QUESTION)
        expect_reply(page, "Cards only.")

    assert [search[1] for search in searches_of(milvus_service, collection)] == ["vector"]
    [chat_request] = [
        body for body in upstream.chat_requests() if reply.answering(FUEL_QUESTION)(body)
    ]
    sent = json.dumps(chat_request["messages"])
    assert "The fuel dock takes cards only." in sent
    assert "The chandlery opens at six." in sent
