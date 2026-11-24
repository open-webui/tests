"""Regression: a citation from the model's knowledge search showed no relevance percentage.

Fix PR open-webui/open-webui#31307 (merged as `1c813902e`, issue open-webui/open-webui#29776)
in `utils/middleware.py`. With native function calling the citations built from
`query_knowledge_files` dropped the distance each chunk came back with, so opening one showed
the cited text without the relevance badge that classic retrieval shows for the same knowledge
base. A person read the badge in the citation dialog. Since 072d0b51d a cited file opens in a tab
beside the chat with the cited passage marked, and nothing there shows its relevance any more:
red on dev 306f9be78 (open-webui/open-webui#34913).

Twin of integration/tools/test_knowledge_tool_citation_scores.py.

The test runs in the main chat and in every way a side chat starts and every state it can be in
(`utils/chat_surface.py`); a knowledge tool's source shows on a side chat answer on dev c15f28649
and on dev e791cbdde alike. Red in the temporary side chat since dev 072d0b51d: clicking the cited
file there opens neither its tab nor the citation dialog (open-webui/open-webui#34921).

In a temporary side chat, the cited file opens beside the chat with the cited passage marked,
from the marker in the reply and from the source list alike, as in a saved side chat. Red on dev
dc9886799 (neither click opens anything, open-webui/open-webui#34921), passes on fix 76dd3f935
(branch fix/temporary-side-chat-source-opens, PR pending).

Discriminates: passes on dev bc2416c5d; with the fix reverted in a backend copy and the dev
build unchanged, the citation dialog shows the text with no relevance percentage. The temporary
side chat test passes on the build of 76dd3f935 (five runs, and two in parallel) and fails on
dev dc9886799 (the marker opens no tab).
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.knowledge_bases import add_text_file, knowledge_base
from utils.chat_surface import EVERYWHERE, TEMPORARY_SIDE, main_chat
from utils.chat_ui import conversation, expect_reply, listed_source, send
from utils.citations_files import expect_marked, file_panel, file_tab

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

NOTES = "Grey herons nest in colonies near water and hunt fish in the shallows."
PERCENTAGE = re.compile(r"\d+\.\d\d%")
PROMPT = "search my notes for herons"


@pytest.fixture
def shared_notes(admin, make_user):
    """A knowledge base holding one file, readable by a fresh account."""
    reader = make_user()
    tag = uuid.uuid4().hex[:8]
    grant = [{"principal_type": "user", "principal_id": reader.id, "permission": "read"}]
    filename = f"herons-{tag}.txt"
    with admin.client() as client, knowledge_base(client, f"Birds {tag}", grant) as knowledge_id:
        add_text_file(client, knowledge_id, filename, NOTES)
        yield reader, knowledge_id, filename


# several models: each answer's sources come with that answer, as with one model
@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_a_knowledge_tool_citation_shows_its_relevance(chat_for, shared_notes, upstream):
    reader, knowledge_id, filename = shared_notes
    arguments = {"query": "heron", "knowledge_ids": [knowledge_id]}
    upstream.queue(
        reply.tool_call("query_knowledge_files", arguments, match=reply.answering(PROMPT)),
        reply.text("Herons nest in colonies.", match=reply.answering(PROMPT)),
    )
    chat = chat_for(reader)

    send(chat, PROMPT)
    expect_reply(chat, "Herons nest in colonies.")

    sources = conversation(chat).get_by_role("button", name="Toggle 1 source")
    expect(sources, f"the reply in the {chat.place} lists no source").to_be_visible()
    sources.click()
    listed_source(chat, filename).click()
    expect_marked(file_panel(chat.page, filename).get_by_text(NOTES, exact=True))
    expect(
        chat.page.get_by_text(PERCENTAGE), "the opened source shows no relevance for its passage"
    ).to_be_visible()


def _expect_open_beside_the_chat(chat, filename: str, clicked: str) -> None:
    """The cited file is the selected tab beside the chat, its passage marked; then closed."""
    page = chat.page
    expect(
        file_tab(page, filename), f"{clicked} in the {chat.place} opened no tab"
    ).to_have_attribute("aria-selected", "true")
    expect_marked(file_panel(page, filename).get_by_text(NOTES, exact=True))
    expect(main_chat(page).input, "the opened file hides the chat").to_be_visible()
    expect(page.get_by_role("dialog")).to_have_count(0)
    page.get_by_role("button", name=f"Close {filename}", exact=True).click()
    expect(file_tab(page, filename)).to_have_count(0)


@pytest.mark.chat_surfaces(TEMPORARY_SIDE)
def test_a_knowledge_tool_source_opens_beside_a_temporary_side_chat(
    chat_for, shared_notes, upstream
):
    reader, knowledge_id, filename = shared_notes
    prompt = f"{PROMPT} {uuid.uuid4().hex[:6]}"
    arguments = {"query": "heron", "knowledge_ids": [knowledge_id]}
    upstream.queue(
        reply.tool_call("query_knowledge_files", arguments, match=reply.answering(prompt)),
        reply.text("Herons nest in colonies [1].", match=reply.answering(prompt)),
    )
    chat = chat_for(reader)

    send(chat, prompt)
    expect_reply(chat, "Herons nest in colonies")

    conversation(chat).get_by_role("button", name=f"View source: {filename}", exact=True).click()
    _expect_open_beside_the_chat(chat, filename, "the marker")

    sources = conversation(chat).get_by_role("button", name="Toggle 1 source")
    expect(sources, f"the reply in the {chat.place} lists no source").to_be_visible()
    sources.click()
    listed_source(chat, filename).click()
    _expect_open_beside_the_chat(chat, filename, "the source list")
