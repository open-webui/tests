"""Journey: the sources a reply was grounded on, as the chat shows them.

A person attaches text files in the chat input and asks a question; the question is answered
from the files' text, and the reply lists the files as its sources. Each inline marker such as
`[2]` names the file the model was told was source 2, and opening it shows that file's text and
its relevance. The sources are still there after a reload and in the read-only view of a shared
link. A model whose Citations capability is off shows neither the source list nor the markers.

Discriminates: passes on dev 176d31d1d; in a frontend copy, the source dialog taking marker `[n]`
as the n+1th source turned the marker test red, the source list shown whatever the model's
capability turned the capability test red and the source list left out of read-only views turned
the shared link test red; in a backend copy, the reply stored without the sources it streamed
turned the reload test red.

Every test but the shared link runs in the main chat and in a side chat
(`utils/chat_surface.py`); the shared link is a page of its own. A side chat answer listed no
sources from an attached file until a reload (open-webui/open-webui#34017, fix PR
open-webui/open-webui#34064), so the side chat variants are red on dev c15f28649.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.access import grant
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.chat_surface import ChatSurface
from utils.chat_ui import (
    chat_input,
    conversation,
    expect_reply,
    last_reply,
    listed_source,
    open_input_menu,
    send,
    start_from_link,
)

pytestmark = [
    pytest.mark.journey,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    # retrieval puts the file text into the prompt only with File Context ticked, and the
    # passages carry a relevance only when a small file is not sent whole
    pytest.mark.usefixtures("file_context_on", "chunked_retrieval"),
]

HERONS = ("herons.txt", "Grey herons wait motionless in the reeds at dawn.")
OTTERS = ("otters.txt", "Otters slide down the muddy bank after rain.")
SOURCE_TOGGLE = re.compile(r"^Toggle \d+ sources?$")
PERCENTAGE = re.compile(r"\d+\.\d\d%")
SOURCE_TAG = re.compile(r'<source id="(\d+)" name="([^"]+)"')
NO_CITATIONS = {
    "file_context": True,
    "vision": True,
    "file_upload": True,
    "web_search": True,
    "citations": False,
    "status_updates": True,
}


def _attach(chat: Page | ChatSurface, *files: tuple[str, str]) -> None:
    page = chat.page if isinstance(chat, ChatSurface) else chat
    expect(chat_input(chat)).to_be_visible()
    menu = open_input_menu(chat)
    with page.expect_file_chooser() as chooser:
        menu.get_by_role("menuitem", name="Upload Files").click()
    chooser.value.set_files(
        [{"name": name, "mimeType": "text/plain", "buffer": text.encode()} for name, text in files]
    )


def _ask_with_files(
    page: Page | ChatSurface, upstream, answer: str, *files: tuple[str, str]
) -> str:
    """Attach `files`, ask about them and wait for `answer`; returns the question asked."""
    question = f"what do my notes say? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    _attach(page, *files)
    send(page, question)
    # an inline marker shows as the source's name
    expect_reply(page, answer.split(" [")[0])
    return question


def _source_ids_sent(upstream, question: str) -> dict[str, str]:
    """The id the model was told each source has, by source number."""
    [request] = [
        request for request in upstream.chat_requests() if reply.answering(question)(request)
    ]
    context = "\n".join(str(message.get("content")) for message in request["messages"])
    return dict(SOURCE_TAG.findall(context))


def _open_listed_source(chat: Page | ChatSurface, name: str, count: int = 1):
    page = chat.page if isinstance(chat, ChatSurface) else chat
    label = "Toggle 1 source" if count == 1 else f"Toggle {count} sources"
    conversation(chat).get_by_role("button", name=label).click()
    listed_source(chat, name).click()
    return page.get_by_role("dialog")


def test_each_inline_marker_opens_the_source_it_names(chat_for, make_user, upstream):
    chat = chat_for(make_user())
    question = _ask_with_files(
        chat, upstream, "Herons wait [1] and otters slide [2].", HERONS, OTTERS
    )
    source_ids = _source_ids_sent(upstream, question)
    assert set(source_ids.values()) == {HERONS[0], OTTERS[0]}, source_ids
    texts = dict((HERONS, OTTERS))

    expect(conversation(chat).get_by_role("button", name="Toggle 2 sources")).to_be_visible()
    markers = last_reply(chat).get_by_role("button", name=re.compile(r"^View source: "))
    expect(markers).to_have_count(2)
    for position in (1, 2):
        name = source_ids[str(position)]
        marker = markers.nth(position - 1)
        expect(marker).to_have_accessible_name(f"View source: {name}")
        marker.click()
        citation = chat.page.get_by_role("dialog")
        expect(citation.get_by_role("link", name=name)).to_be_visible()
        expect(citation).to_contain_text(texts[name])
        expect(citation.get_by_text(PERCENTAGE)).to_be_visible()
        citation.get_by_role("button", name="Close citation modal").click()
        expect(citation).to_be_hidden()


def test_the_sources_are_still_listed_after_a_reload(chat_for, make_user, upstream):
    chat = chat_for(make_user())
    _ask_with_files(chat, upstream, "They wait in the reeds.", HERONS)
    expect(chat.page).to_have_url(re.compile(r"/c/[0-9a-f-]+$"))

    chat = chat.reload()
    expect_reply(chat, "They wait in the reeds.")
    citation = _open_listed_source(chat, HERONS[0])
    expect(citation).to_contain_text(HERONS[1])


def test_a_shared_link_shows_the_sources(page_for, make_user, upstream):
    owner, viewer = make_user(), make_user()
    page = page_for(owner)
    _ask_with_files(page, upstream, "They wait in the reeds.", HERONS)
    expect(page).to_have_url(re.compile(r"/c/[0-9a-f-]+$"))
    chat_id = page.url.rsplit("/", 1)[-1]
    with owner.client() as client:
        shared = client.post(f"/api/v1/chats/{chat_id}/share")
        assert shared.status_code == 200, shared.text
        granted = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update",
            json={"access_grants": [grant("user", viewer.id, "read")]},
        )
        assert granted.status_code == 200, granted.text

    viewer_page = page_for(viewer, path=f"/s/{shared.json()['share_id']}")
    expect(conversation(viewer_page).get_by_text("They wait in the reeds.")).to_be_visible()
    citation = _open_listed_source(viewer_page, HERONS[0])
    expect(citation).to_contain_text(HERONS[1])


@pytest.fixture
def model_without_citations(admin):
    """A preset on the scripted model with its Citations capability off, open to everyone."""
    model_id = f"no-citations-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": f"No citations {model_id[-8:]}",
        "meta": {"capabilities": NO_CITATIONS},
        "params": {},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_a_model_with_citations_off_shows_no_sources(
    chat_for, make_user, upstream, model_without_citations
):
    chat = chat_for(make_user())
    start_from_link(chat, f"/?models={model_without_citations}")
    question = _ask_with_files(chat, upstream, "Herons wait in the reeds [1].", HERONS)
    assert _source_ids_sent(upstream, question) == {"1": HERONS[0]}

    reply_shown = last_reply(chat)
    expect(reply_shown).to_contain_text("Herons wait in the reeds.")
    expect(reply_shown).not_to_contain_text("[1]")
    expect(reply_shown.get_by_role("button", name=re.compile(r"^View source: "))).to_have_count(0)
    expect(conversation(chat).get_by_role("button", name=SOURCE_TOGGLE)).to_have_count(0)
