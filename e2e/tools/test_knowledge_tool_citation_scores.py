"""Regression: a citation from the model's knowledge search showed no relevance percentage.

Fix PR open-webui/open-webui#31307 (merged as `1c813902e`, issue open-webui/open-webui#29776)
in `utils/middleware.py`. With native function calling the citations built from
`query_knowledge_files` dropped the distance each chunk came back with, so opening one showed
the cited text without the relevance badge that classic retrieval shows for the same knowledge
base. A person reads the badge in the citation dialog.

Twin of integration/tools/test_knowledge_tool_citation_scores.py.

The test runs in the main chat and in every way a side chat starts and every state it can be in
(`utils/chat_surface.py`); a knowledge tool's source shows on a side chat answer on dev c15f28649
and on dev e791cbdde alike.

Discriminates: passes on dev bc2416c5d; with the fix reverted in a backend copy and the dev
build unchanged, the citation dialog shows the text with no relevance percentage.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.knowledge_bases import add_text_file, knowledge_base
from utils.chat_surface import EVERYWHERE
from utils.chat_ui import conversation, expect_reply, listed_source, send

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
    citation = chat.page.get_by_role("dialog")
    expect(citation).to_contain_text(NOTES)
    expect(citation.get_by_text(PERCENTAGE)).to_be_visible()
