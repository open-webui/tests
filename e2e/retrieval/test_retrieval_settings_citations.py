"""Journey: the retrieval settings an admin saves in Admin Settings > Documents, seen in a chat.

An admin changes one setting on the Documents tab and saves; a person then attaches a file of
field notes in a chat, asks about it and opens the reply's source, which lists the passages the
model was given. Top K decides how many passages are cited. Hybrid Search with a Relevance
Threshold cites only the passages that match the question, and switched off again cites the
rest as well. An External reranker picked in the panel decides the cited passage, which shows the
relevance the reranker gave it. Chunk Size decides how much of the file each passage holds, and
Chunk Overlap repeats a sentence in the two neighbouring passages that share it. Full Context
Mode gives the model the whole file however low Top K is. A RAG Template frames the context the
model is sent, and one that places the context twice is flagged in the panel and sends it twice.

The instance embeds through a local service whose vectors follow a few keywords, so a question
about herons is nearest the passages about herons; a local service also plays the reranker.

Discriminates: passes on dev ebc6add67; in a frontend build whose Documents form sends the
retrieval settings it loaded in place of the edited ones, every test fails.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor, admin_of, create_user
from harness.keyword_embeddings import keyword_embedding_env, serve_keyword_embeddings
from harness.listener import ReceivedRequest, json_answer, listening
from harness.web_retrieval import RETRIEVAL_CONFIG
from utils.chat_ui import REPLY_TIMEOUT_MS, chat_input, conversation, expect_reply, send

pytestmark = [
    pytest.mark.journey,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

KEYWORDS = ["heron", "otter", "grebe"]
PARAGRAPHS = [
    "The grey heron waits in the shallows at dawn.",
    "Otters slide down the muddy bank after rain.",
    "A second heron stalks frogs along the ditch.",
    "Great crested grebes dance on the open water.",
    "Reed warblers sing from the tall stems.",
    "Coots squabble over weed near the jetty.",
]
HERONS = [PARAGRAPHS[0], PARAGRAPHS[2]]
OTTERS = PARAGRAPHS[1]
GREBES = PARAGRAPHS[3]
RELEVANCE = re.compile(r"^\d+\.\d\d%$")
NOTES = ("field-notes.txt", "\n\n".join(PARAGRAPHS))
# one paragraph per chunk: two paragraphs together run past it
BASELINE = {
    "CONTENT_EXTRACTION_ENGINE": "",
    "BYPASS_EMBEDDING_AND_RETRIEVAL": False,
    "TEXT_SPLITTER": "",
    "ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER": False,
    "CHUNK_SIZE": 60,
    "CHUNK_OVERLAP": 0,
    "CHUNK_MIN_SIZE_TARGET": 0,
    "TOP_K": 1,
    "TOP_K_RERANKER": 6,
    "RELEVANCE_THRESHOLD": 0.0,
    "ENABLE_RAG_HYBRID_SEARCH": False,
    "RAG_FULL_CONTEXT": False,
    "RAG_RERANKING_ENGINE": "",
    "RAG_RERANKING_MODEL": "",
    "RAG_TEMPLATE": "",
}


@pytest.fixture(scope="module")
def embedder():
    with listening() as service:
        serve_keyword_embeddings(service, KEYWORDS)
        yield service


@pytest.fixture
def ranked(instance_with, embedder, preserve):
    """The keyword-embedding instance on the baseline settings, restored after the test."""
    launched = instance_with(keyword_embedding_env(embedder))
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    preserve(RETRIEVAL_CONFIG, on=launched)
    with launched.client() as client:
        saved = client.post(RETRIEVAL_CONFIG[1], json=BASELINE)
    assert saved.status_code == 200, saved.text
    return launched


def open_documents(page: Page) -> Locator:
    page.goto("/admin/settings/documents")
    settings = page.get_by_role("dialog")
    # the tab loads the embedding and retrieval settings before it renders
    expect(settings.get_by_text("Allowed File Extensions", exact=True)).to_be_visible(
        timeout=15_000
    )
    return settings


@pytest.fixture
def documents(ranked, page_for) -> Locator:
    """The admin's Documents tab on the keyword-embedding instance."""
    return open_documents(page_for(admin_of(ranked)))


@pytest.fixture
def reader(ranked) -> Actor:
    return create_user(ranked)


def field(settings: Locator, label: str, kind: str = "input") -> Locator:
    return settings.get_by_text(label, exact=True).locator(f"xpath=following-sibling::div//{kind}")


def set_switch(settings: Locator, name: str, turn_on: bool) -> None:
    switch = settings.get_by_role("switch", name=name, exact=True)
    if (switch.get_attribute("aria-checked") == "true") != turn_on:
        switch.click()
    expect(switch).to_have_attribute("aria-checked", "true" if turn_on else "false")


def save(settings: Locator) -> None:
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(settings.page.get_by_text("Settings saved successfully!").first).to_be_visible()


def ask_about_the_notes(page_for, ranked, reader: Actor, question: str) -> Page:
    """Attach the field notes in a new chat, ask `question` and wait for the reply."""
    page = page_for(reader)
    expect(chat_input(page)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    page.get_by_role("main").get_by_label("More").click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("button", name="Upload Files").click()
    name, text = NOTES
    chooser.value.set_files({"name": name, "mimeType": "text/plain", "buffer": text.encode()})
    expect(page.get_by_role("button", name=name)).to_be_visible()
    ranked.upstream.queue(reply.text("Noted.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Noted.")
    return page


def sent_to_the_model(ranked, question: str) -> str:
    request = next(filter(reply.answering(question), ranked.upstream.chat_requests()))
    return "\n".join(str(message.get("content")) for message in request["messages"])


def cited_passages(page: Page) -> Locator:
    """Open the reply's one source; returns the dialog listing its passages."""
    chat = conversation(page)
    chat.get_by_role("button", name="Toggle 1 source").click()
    chat.get_by_role("button", name=f"View source: {NOTES[0]}").click()
    citation = page.get_by_role("dialog")
    expect(citation.get_by_role("link", name=NOTES[0])).to_be_visible()
    return citation


def passage_count(citation: Locator) -> Locator:
    # each passage shows its relevance
    return citation.get_by_text(RELEVANCE)


def unique_question(text: str) -> str:
    return f"{text} {uuid.uuid4().hex[:6]}"


def test_top_k_decides_how_many_passages_are_cited(documents, page_for, ranked, reader):
    field(documents, "Top K").fill("2")
    save(documents)

    page = ask_about_the_notes(page_for, ranked, reader, unique_question("Where do herons hunt?"))

    citation = cited_passages(page)
    expect(passage_count(citation)).to_have_count(2)
    for heron in HERONS:
        expect(citation.get_by_text(heron, exact=True)).to_be_visible()
    expect(citation.get_by_text(OTTERS, exact=True)).to_have_count(0)


def test_hybrid_search_with_a_threshold_cites_only_matching_passages(
    documents, page_for, ranked, reader
):
    field(documents, "Top K").fill("6")
    set_switch(documents, "Hybrid Search", turn_on=True)
    field(documents, "Top K Reranker").fill("6")
    field(documents, "Relevance Threshold").fill("0.5")
    save(documents)

    question = unique_question("Where do herons hunt?")
    page = ask_about_the_notes(page_for, ranked, reader, question)

    citation = cited_passages(page)
    expect(passage_count(citation)).to_have_count(2)
    for heron in HERONS:
        expect(citation.get_by_text(heron, exact=True)).to_be_visible()
    sent = sent_to_the_model(ranked, question)
    assert [paragraph for paragraph in PARAGRAPHS if paragraph in sent] == HERONS, sent

    documents = open_documents(documents.page)
    set_switch(documents, "Hybrid Search", turn_on=False)
    save(documents)
    page = ask_about_the_notes(page_for, ranked, reader, unique_question("Where do herons hunt?"))
    expect(passage_count(cited_passages(page))).to_have_count(6)


@pytest.fixture
def reranker():
    """A reranker that scores the grebe paragraph highest; records what it was asked."""

    def rerank(request: ReceivedRequest):
        documents = request.json()["documents"]
        scores = [0.9 if "grebe" in document else 0.1 for document in documents]
        results = [{"index": index, "relevance_score": score} for index, score in enumerate(scores)]
        return json_answer({"results": results})

    with listening() as service:
        service.route("POST", "/rerank", rerank)
        yield service


def test_an_external_reranker_picks_the_cited_passage(
    documents, page_for, ranked, reader, reranker
):
    field(documents, "Top K").fill("6")
    set_switch(documents, "Hybrid Search", turn_on=True)
    documents.get_by_role("combobox", name="Select a reranking model engine").select_option(
        "external"
    )
    # the embedding engine's URL and key fields carry the same labels
    field(documents, "API Base URL").last.fill(f"{reranker.base_url}/rerank")
    documents.get_by_placeholder("API Key", exact=True).last.fill("rerank-key")
    field(documents, "Reranking Model").fill("grebe-ranker")
    field(documents, "Top K Reranker").fill("1")
    save(documents)

    question = unique_question("Where do herons hunt?")
    page = ask_about_the_notes(page_for, ranked, reader, question)

    citation = cited_passages(page)
    expect(passage_count(citation)).to_have_count(1)
    expect(citation.get_by_text(GREBES, exact=True)).to_be_visible()
    expect(citation.get_by_text("90.00%")).to_be_visible()
    asked = reranker.requests_to("/rerank")
    assert asked, "the external reranker was never asked"
    assert asked[-1].headers.get("Authorization") == "Bearer rerank-key"
    assert asked[-1].json()["model"] == "grebe-ranker"


def test_chunk_size_decides_how_much_each_cited_passage_holds(documents, page_for, ranked, reader):
    field(documents, "Chunk Size").fill("1000")
    save(documents)

    page = ask_about_the_notes(page_for, ranked, reader, unique_question("Where do herons hunt?"))

    citation = cited_passages(page)
    expect(passage_count(citation)).to_have_count(1)
    for paragraph in PARAGRAPHS:
        expect(citation.get_by_text(paragraph, exact=True)).to_be_visible()


def test_chunk_overlap_repeats_a_sentence_in_neighbouring_passages(
    documents, page_for, ranked, reader
):
    field(documents, "Top K").fill("2")
    field(documents, "Chunk Size").fill("100")
    field(documents, "Chunk Overlap").fill("50")
    save(documents)

    page = ask_about_the_notes(page_for, ranked, reader, unique_question("Where do otters play?"))

    citation = cited_passages(page)
    expect(passage_count(citation)).to_have_count(2)
    # with two paragraphs to a chunk, only the overlap puts the otters in a second one
    expect(citation.get_by_text(OTTERS, exact=True)).to_have_count(2)


def test_full_context_mode_gives_the_model_the_whole_file(documents, page_for, ranked, reader):
    set_switch(documents, "Full Context Mode", turn_on=True)
    expect(documents.get_by_text("Top K", exact=True)).to_have_count(0)
    save(documents)

    question = unique_question("Where do herons hunt?")
    page = ask_about_the_notes(page_for, ranked, reader, question)

    sent = sent_to_the_model(ranked, question)
    assert [paragraph for paragraph in PARAGRAPHS if paragraph in sent] == PARAGRAPHS, sent
    citation = cited_passages(page)
    for paragraph in PARAGRAPHS:
        expect(citation.get_by_text(re.compile(re.escape(paragraph))).first).to_be_visible()


def test_a_rag_template_frames_what_the_model_is_sent(documents, page_for, ranked, reader):
    template = "Answer only from these notes:\n{{CONTEXT}}\nThe question was: {{QUERY}}"
    field(documents, "RAG Template", kind="textarea").fill(template)
    save(documents)

    question = unique_question("Which grebe dances?")
    ask_about_the_notes(page_for, ranked, reader, question)

    sent = sent_to_the_model(ranked, question)
    assert "Answer only from these notes:" in sent, sent
    assert re.search(f"The question was: .*{re.escape(question)}", sent, re.DOTALL), sent
    assert GREBES in sent, sent


def test_a_template_placing_the_context_twice_is_flagged_and_sends_it_twice(
    documents, page_for, ranked, reader
):
    template = "Notes:\n{{CONTEXT}}\nQuestion: {{QUERY}}\nNotes again:\n{{CONTEXT}}"
    field(documents, "RAG Template", kind="textarea").fill(template)
    expect(
        documents.get_by_text("This template contains multiple context placeholders")
    ).to_be_visible()
    save(documents)

    question = unique_question("Which grebe dances?")
    ask_about_the_notes(page_for, ranked, reader, question)

    sent = sent_to_the_model(ranked, question)
    assert sent.count(GREBES) == 2, sent
