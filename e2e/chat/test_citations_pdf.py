"""Journey: a reply that cites a PDF opens it beside the chat, at the cited page, passage marked.

A person attaches a PDF in the chat, or asks a knowledge base that holds one, and the reply cites
it inline as `[1]`. The marker shows as a chip with the file's name, right where the model put it.
Clicking it opens the PDF in a tab named after the file beside the reply, scrolled to the page the
cited passage was read from, with that passage highlighted and the page current in the page strip.
Resting the pointer on the highlight shows the passage and its page in a card at the cited text.
The highlight covers the cited text at any width of the tab. Closing the tab gives the chat
its room back, and the chip opens the PDF again at the same page, also after a reload. A passage
cited from several pages is marked on each, and the chip takes the reader back to the most relevant
one wherever they scrolled since. Two cited PDFs each open on their own page, by a single chip or
from a group marker's list. The source list opens the PDF as its chip does (since 072d0b51d, for
open-webui/open-webui#34831; before, it opened a dialog listing each passage with its page). The
same works by keyboard, on a phone in the Tabs drawer, in a side chat, and in the read-only view
of a shared link, which shows the dialog. A PDF read by MinerU opens at
the page MinerU placed the passage on.

Dev commits 9b6241e54 (MinerU keeps the page of each passage) and cc2e033b0 (a cited PDF opens in
the chat's Preview tab with the passage marked). The real-upload and MinerU tests run on an
instance that embeds through a local keyword service, so the question picks one page; the rest
store the chat with its sources as retrieval leaves them, on a real uploaded PDF.

Discriminates: passes on the dev c1f3b6fd1 build, but for the reds below. In copies of that build
with the cited page dropped (the PDF opens at page 1), with no passage marked, or with the PDF
branch of the citation click removed (the dialog opens instead), every test that opens the PDF
fails; the source list and shared link tests pass there. In a backend copy whose MinerU local
loader returns the text as one piece again (9b6241e54 undone), the MinerU test fails: the PDF
opens at page 1. In a copy of the dev d599f7a08 build that draws the highlights 150 pixels left of
the text, the two PDFs test fails at its first tab width.

The several pages test was red on dev c1f3b6fd1: clicking the chip again after moving to another
page left the PDF where it was (open-webui/open-webui#34834, fixed by 83e1d38d8), and in a side
chat (a saved chat dragged in beside another) the chip opened nothing
(open-webui/open-webui#34828). Since 072d0b51d a PDF cited in a side chat opens in that chat's
tab group, over the chat, so the reader goes back to the chat's tab to use the chip again.
"""

from __future__ import annotations

import contextlib
import json
import re
import uuid
from typing import Callable, Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import sample_documents as samples
from harness import upstream as reply
from harness.access import grant
from harness.actors import Actor, admin_of, create_user
from harness.chat_history import seed_chat
from harness.file_context import file_context_ticked
from harness.keyword_embeddings import keyword_embedding_env, serve_keyword_embeddings
from harness.knowledge_bases import knowledge_base
from harness.listener import json_answer, listening
from harness.web_retrieval import RETRIEVAL_CONFIG
from utils.cached_chat import attach, attach_from_menu
from utils.chat_surface import ChatSurface
from utils.chat_ui import chat_input, conversation, expect_reply, last_reply, listed_source, send
from utils.citation_marks import expect_card_at, expect_marks_on_cited_text
from utils.citations_pdf import (
    cited_pdf_panel,
    cited_pdf_tab,
    expect_marked,
    expect_page_shown,
    page_strip,
    passage_card,
    pdf_source,
    pdf_viewer,
    unique_name,
    upload_pdf,
)
from utils.citations_ui import group_card, leave_marker
from utils.phone import PHONE, drawer_page, tap_on_screen

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

KEYWORDS = ["gates", "heron", "otter"]
HARBOUR = [
    "The harbour gates open at six every morning.",
    "Grey herons nest on the north pier each spring.",
    "Otters fish below the old sea wall at dusk.",
]
HERON_ANSWER = "They nest on the north pier [1]. End of notes."
# one page per passage, and only the closest passage is cited
RANKED_SETTINGS = {
    "CONTENT_EXTRACTION_ENGINE": "",
    "BYPASS_EMBEDDING_AND_RETRIEVAL": False,
    "TEXT_SPLITTER": "",
    "ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER": False,
    "CHUNK_SIZE": 200,
    "CHUNK_OVERLAP": 0,
    "CHUNK_MIN_SIZE_TARGET": 0,
    "TOP_K": 1,
    "RELEVANCE_THRESHOLD": 0.0,
    "ENABLE_RAG_HYBRID_SEARCH": False,
    "RAG_FULL_CONTEXT": False,
    "RAG_TEMPLATE": "",
    "RAG_FULL_TEXT_CHAR_THRESHOLD": 0,
}
TIDES = [
    "High tide reaches the slipway before noon.",
    "The ferry waits for the evening tide to turn.",
    "Spring tides cover the lower quay twice a month.",
]
BIRDS = [
    "Cormorants dry their wings on the buoys.",
    "Terns dive for sand eels off the point.",
    "Gulls follow the trawlers back to port.",
]


@pytest.fixture(scope="module")
def embedder():
    with listening() as service:
        serve_keyword_embeddings(service, KEYWORDS)
        yield service


@pytest.fixture
def ranked(instance_with, embedder, preserve):
    """The keyword-embedding instance on the settings above, with File Context ticked."""
    launched = instance_with(keyword_embedding_env(embedder))
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    preserve(RETRIEVAL_CONFIG, on=launched)
    with launched.client() as client:
        saved = client.post(RETRIEVAL_CONFIG[1], json=RANKED_SETTINGS)
    assert saved.status_code == 200, saved.text
    with file_context_ticked(admin_of(launched)):
        yield launched


@pytest.fixture
def knowledge_of(ranked) -> Iterator[Callable[[Actor, str], str]]:
    """`knowledge_of(person, name)` shares a knowledge base holding the report as `name` with the
    person; returns the base's title."""
    with contextlib.ExitStack() as stack:

        def make(person: Actor, name: str) -> str:
            client = stack.enter_context(admin_of(ranked).client())
            title = f"Harbour handbook {uuid.uuid4().hex[:6]}"
            readers = [grant("user", person.id, "read")]
            knowledge_id = stack.enter_context(knowledge_base(client, title, readers))
            uploaded = client.post(
                "/api/v1/files/",
                params={"process": "true", "process_in_background": "false"},
                files={"file": (name, samples.pdf(HARBOUR), "application/pdf")},
            )
            assert uploaded.status_code == 200, f"uploading {name} failed: {uploaded.text}"
            added = client.post(
                f"/api/v1/knowledge/{knowledge_id}/file/add",
                json={"file_id": uploaded.json()["id"]},
            )
            assert added.status_code == 200, f"adding {name} failed: {added.text}"
            return title

        yield make


def attach_in_chat(page: Page, name: str) -> None:
    attach(page, name, samples.pdf(HARBOUR), "application/pdf")
    expect(page.locator("form").get_by_role("button", name=name)).to_be_visible()


def ask_about_the_herons(ranked, page: Page, attach_the_report: Callable[[Page], None]) -> None:
    expect(chat_input(page)).to_be_visible()
    attach_the_report(page)
    question = f"where do herons nest? {uuid.uuid4().hex[:6]}"
    ranked.upstream.queue(reply.text(HERON_ANSWER, match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "End of notes.")


def source_chip(chat: Page | ChatSurface, name: str) -> Locator:
    return last_reply(chat).get_by_role("button", name=f"View source: {name}", exact=True)


def open_cited_pdf(chat: Page | ChatSurface, name: str) -> Locator:
    """Click the reply's chip for `name`; returns the viewer of the tab it opens."""
    page = chat.page if isinstance(chat, ChatSurface) else chat
    source_chip(chat, name).click()
    expect(cited_pdf_tab(page, name)).to_have_attribute("aria-selected", "true")
    return pdf_viewer(cited_pdf_panel(page, name))


def seed_cited_chat(account: Actor, answer: str, sources: list[dict]) -> str:
    with account.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "what do my notes say?"},
                {"role": "assistant", "content": answer, "sources": sources},
            ],
        )
    return chat_id


def tides_chat(account: Actor) -> tuple[str, str]:
    """A chat citing the tide tables on page 3, the most relevant, and page 1; (chat id, name)."""
    name = unique_name("tide-tables")
    with account.client() as client:
        file_id = upload_pdf(client, name, TIDES)
    source = pdf_source(file_id, name, [(TIDES[2], 3, 0.9), (TIDES[0], 1, 0.6)])
    return seed_cited_chat(
        account, "Spring tides cover the quay [1]. End of notes.", [source]
    ), name


@pytest.mark.slow
@pytest.mark.parametrize("attached", ["in the chat", "in a knowledge base"])
def test_a_cited_pdf_opens_beside_the_reply_at_the_cited_page_with_the_passage_marked(
    ranked, page_for, knowledge_of, attached
):
    person = create_user(ranked)
    name = unique_name("field-report")
    if attached == "in the chat":
        attach_the_report = lambda page: attach_in_chat(page, name)  # noqa: E731
    else:
        handbook = knowledge_of(person, name)
        attach_the_report = lambda page: attach_from_menu(page, "Attach Knowledge", handbook)  # noqa: E731
    page = page_for(person)
    ask_about_the_herons(ranked, page, attach_the_report)

    chip = source_chip(page, name)
    expect(chip).to_have_count(1)
    expect(last_reply(page).get_by_role("button", name=re.compile(r"^View source"))).to_have_count(
        1
    )
    # the chip stands where the model wrote [1], in place of the marker
    expect(last_reply(page)).to_contain_text(
        re.compile(rf"They nest on the north pier\s*{re.escape(name)}\s*\. End of notes\.")
    )
    expect(conversation(page).get_by_role("button", name="Toggle 1 source")).to_be_visible()

    viewer = open_cited_pdf(page, name)
    # beside the chat, so the reply stays readable
    expect(last_reply(page)).to_be_in_viewport()
    expect_page_shown(viewer, HARBOUR[1], [HARBOUR[0], HARBOUR[2]])
    expect(page_strip(cited_pdf_panel(page, name), 2)).to_have_attribute("aria-current", "true")
    expect_marked(viewer, HARBOUR[1])
    card = passage_card(viewer, HARBOUR[1])
    expect(card).to_contain_text("Page 2")
    page.mouse.move(0, 0)
    expect(card).to_be_hidden()

    page.get_by_role("button", name=f"Close {name}", exact=True).click()
    expect(cited_pdf_tab(page, name)).to_have_count(0)
    expect(chat_input(page)).to_be_visible()
    viewer = open_cited_pdf(page, name)
    expect_page_shown(viewer, HARBOUR[1], [HARBOUR[0], HARBOUR[2]])

    page.reload()
    expect_reply(page, "End of notes.")
    viewer = open_cited_pdf(page, name)
    expect_page_shown(viewer, HARBOUR[1], [HARBOUR[0], HARBOUR[2]])
    expect_marked(viewer, HARBOUR[1])


def test_a_pdf_cited_from_several_pages_marks_each_and_returns_to_the_most_relevant(
    chat_for, make_user
):
    account = make_user()
    chat_id, name = tides_chat(account)
    chat = chat_for(account, chat_id=chat_id)
    expect_reply(chat, "End of notes.")

    viewer = open_cited_pdf(chat, name)
    expect_page_shown(viewer, TIDES[2], [TIDES[0], TIDES[1]])
    panel = cited_pdf_panel(chat.page, name)
    expect(page_strip(panel, 3)).to_have_attribute("aria-current", "true")
    expect_marked(viewer, TIDES[0], TIDES[2])
    expect(passage_card(viewer, TIDES[2])).to_contain_text("Page 3")

    page_strip(panel, 1).click()
    expect_page_shown(viewer, TIDES[0], [TIDES[1], TIDES[2]])
    expect(passage_card(viewer, TIDES[0])).to_contain_text("Page 1")
    chat.page.mouse.move(0, 0)

    # the citation leads back to its page wherever the reader scrolled since
    if chat.is_side:
        chat.show()
    source_chip(chat, name).click()
    expect(cited_pdf_tab(chat.page, name)).to_have_attribute("aria-selected", "true")
    expect_page_shown(viewer, TIDES[2], [TIDES[0], TIDES[1]])


def test_the_source_list_opens_the_pdf_at_the_most_relevant_page_with_each_cited_page_marked(
    page_for, make_user
):
    account = make_user()
    chat_id, name = tides_chat(account)
    page = page_for(account, path=f"/c/{chat_id}")
    expect_reply(page, "End of notes.")

    conversation(page).get_by_role("button", name="Toggle 1 source").click()
    listed_source(page, name).click()
    expect(cited_pdf_tab(page, name)).to_have_attribute("aria-selected", "true")
    panel = cited_pdf_panel(page, name)
    viewer = pdf_viewer(panel)
    expect_page_shown(viewer, TIDES[2], [TIDES[0], TIDES[1]])
    expect(page_strip(panel, 3)).to_have_attribute("aria-current", "true")
    expect_marked(viewer, TIDES[0], TIDES[2])


def test_each_of_two_cited_pdfs_opens_on_its_own_page(page_for, make_user):
    account = make_user()
    tides, birds = unique_name("tide-tables"), unique_name("bird-log")
    with account.client() as client:
        tides_id = upload_pdf(client, tides, TIDES)
        birds_id = upload_pdf(client, birds, BIRDS)
    sources = [
        pdf_source(tides_id, tides, [(TIDES[1], 2, 0.8)]),
        pdf_source(birds_id, birds, [(BIRDS[2], 3, 0.7)]),
    ]
    chat_id = seed_cited_chat(
        account, "The ferry waits [1] and gulls follow [2]. Both [1][2]. End of notes.", sources
    )
    page = page_for(account, path=f"/c/{chat_id}")
    expect_reply(page, "End of notes.")

    viewer = open_cited_pdf(page, tides)
    expect_page_shown(viewer, TIDES[1], [TIDES[0], TIDES[2]])
    expect_marked(viewer, TIDES[1])
    # the highlight covers the cited line at any width of the tab, and its card sits at the line
    line = viewer.get_by_text(TIDES[1], exact=True)
    for width in (2600, 1280, 1920):
        page.set_viewport_size({"width": width, "height": 1080})
        expect_marks_on_cited_text(viewer, line)
    card = passage_card(viewer, TIDES[1])
    expect(card).to_contain_text("Page 2")
    expect_card_at(card, line)
    page.mouse.move(0, 0)

    viewer = open_cited_pdf(page, birds)
    expect_page_shown(viewer, BIRDS[2], [BIRDS[0], BIRDS[1]])
    expect_marked(viewer, BIRDS[2])
    expect(page.get_by_text(TIDES[1], exact=True)).to_have_count(0)
    # the chip's card quotes the passage too, and the group card opens over it
    leave_marker(page)
    expect(group_card(page)).to_have_count(0)

    # the group marker lists both files, and each entry opens its own
    group = last_reply(page).get_by_role("button", name=re.compile(r"\+1 more sources$")).last
    group.click()
    listed = page.locator("[data-link-preview-content]")
    listed.get_by_role("button", name=f"View source: {tides}").click()
    expect(cited_pdf_tab(page, tides)).to_have_attribute("aria-selected", "true")
    viewer = pdf_viewer(cited_pdf_panel(page, tides))
    expect_page_shown(viewer, TIDES[1], [TIDES[0], TIDES[2]])
    expect(page.get_by_text(BIRDS[2], exact=True)).to_have_count(0)


def test_a_cited_pdf_opens_at_its_page_by_keyboard(page_for, make_user):
    account = make_user()
    chat_id, name = tides_chat(account)
    page = page_for(account, path=f"/c/{chat_id}")
    expect_reply(page, "End of notes.")

    # back from the source list toggle to the chip before it
    conversation(page).get_by_role("button", name="Toggle 1 source").focus()
    page.keyboard.press("Shift+Tab")
    expect(source_chip(page, name)).to_be_focused()
    page.keyboard.press("Enter")
    expect(cited_pdf_tab(page, name)).to_have_attribute("aria-selected", "true")
    panel = cited_pdf_panel(page, name)
    viewer = pdf_viewer(panel)
    expect_page_shown(viewer, TIDES[2], [TIDES[0], TIDES[1]])
    expect_marked(viewer, TIDES[0], TIDES[2])

    page_strip(panel, 1).focus()
    page.keyboard.press("Enter")
    expect_page_shown(viewer, TIDES[0], [TIDES[1], TIDES[2]])

    page.get_by_role("button", name=f"Close {name}", exact=True).focus()
    page.keyboard.press("Enter")
    expect(cited_pdf_tab(page, name)).to_have_count(0)
    source_chip(page, name).focus()
    page.keyboard.press("Space")
    expect(cited_pdf_tab(page, name)).to_have_attribute("aria-selected", "true")
    expect_page_shown(pdf_viewer(cited_pdf_panel(page, name)), TIDES[2], [TIDES[0], TIDES[1]])


def test_a_cited_pdf_opens_at_its_page_on_a_phone(page_for, make_user):
    account = make_user()
    chat_id, name = tides_chat(account)
    page = page_for(account, path=f"/c/{chat_id}", **PHONE)
    expect_reply(page, "End of notes.")

    tap_on_screen(source_chip(page, name))
    drawer = page.get_by_role("dialog", name="Tabs")
    expect(cited_pdf_tab(page, name)).to_have_attribute("aria-selected", "true")
    viewer = pdf_viewer(drawer_page(page))
    expect_page_shown(viewer, TIDES[2], [TIDES[0], TIDES[1]])
    expect_marked(viewer, TIDES[0], TIDES[2])

    drawer.get_by_role("button", name="Hide tabs").tap()
    expect(drawer).to_be_hidden()
    expect(source_chip(page, name)).to_be_in_viewport()


def test_a_shared_link_shows_each_cited_passage_with_its_page(page_for, make_user):
    owner, reader = make_user(), make_user()
    chat_id, name = tides_chat(owner)
    with owner.client() as client:
        shared = client.post(f"/api/v1/chats/{chat_id}/share")
        assert shared.status_code == 200, shared.text
        granted = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update",
            json={"access_grants": [grant("user", reader.id, "read")]},
        )
        assert granted.status_code == 200, granted.text

    page = page_for(reader, path=f"/s/{shared.json()['share_id']}")
    expect(conversation(page).get_by_text("End of notes.")).to_be_visible()
    conversation(page).get_by_role("button", name=f"View source: {name}").click()
    dialog = page.get_by_role("dialog")
    expect(dialog).to_contain_text(re.compile(rf"\(page\s+3\)\s*{re.escape(TIDES[2])}"))
    expect(dialog).to_contain_text(re.compile(rf"\(page\s+1\)\s*{re.escape(TIDES[0])}"))
    # a read-only view has no tab to open the PDF in
    expect(cited_pdf_tab(page, name)).to_have_count(0)


def mineru_parses_the_report(listener) -> None:
    """A MinerU 3.x stand-in: the report as one Markdown text, with the page of each paragraph."""
    markdown = "# Field report\n\n" + "\n\n".join(HARBOUR)
    content_list = [{"type": "text", "text": "Field report", "text_level": 1, "page_idx": 0}] + [
        {"type": "text", "text": line, "page_idx": index} for index, line in enumerate(HARBOUR)
    ]
    parsed = {"md_content": markdown, "content_list": json.dumps(content_list)}
    listener.route(
        "POST",
        "/file_parse",
        json_answer({"backend": "pipeline", "version": "3.0.0", "results": {"report": parsed}}),
    )


@pytest.mark.slow
def test_a_pdf_read_by_mineru_opens_at_the_page_mineru_found_the_passage_on(
    ranked, page_for, listener
):
    mineru_parses_the_report(listener)
    with ranked.client() as client:
        saved = client.post(
            RETRIEVAL_CONFIG[1],
            json={
                "CONTENT_EXTRACTION_ENGINE": "mineru",
                "MINERU_API_MODE": "local",
                "MINERU_API_URL": listener.base_url,
                "MINERU_API_KEY": "",
                "MINERU_PARAMS": {},
                "MINERU_FILE_EXTENSIONS": ["pdf"],
            },
        )
    assert saved.status_code == 200, saved.text
    name = unique_name("field-report")
    page = page_for(create_user(ranked))
    ask_about_the_herons(ranked, page, lambda chat: attach_in_chat(chat, name))
    assert listener.requests_to("/file_parse"), "the PDF was never sent to MinerU"

    viewer = open_cited_pdf(page, name)
    expect_page_shown(viewer, HARBOUR[1], [HARBOUR[0], HARBOUR[2]])
    expect_marked(viewer, HARBOUR[1])
    expect(passage_card(viewer, HARBOUR[1])).to_contain_text("Page 2")
