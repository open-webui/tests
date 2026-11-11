"""The Attach Webpage modal in the composer: links go in, pages come out as attachments.

The plus menu offers Attach Webpage under Add Context (since dev 834c47ee0); the modal takes one
link per line, refuses text that is not an http(s) link, and the page behind each link is fetched
by the server and shown in the composer as an attachment named by its link. The text of the page
reaches the model with the next message, and a page the server cannot read is refused with a toast
and leaves no attachment. A link in the `load-url` parameter of a chat link is attached the same
way once the person confirms it in the Open link dialog, and nothing is fetched before that (since
0ffd86967). Without the web upload permission the menu entry does nothing for a user and still
opens for an admin; without the file upload permission an attached page stays and reaches the
model. The pages are a local service, on an instance that may fetch loopback addresses.

Discriminates: passes on the 176d31d1d build; with the link validation removed the refused-text
test goes red, with the de-duplication removed the several-links test does, with the handoff to
the chat or the modal's close button removed the attachment, refusal and close tests do, with a
refused page left in the composer the refusal test does, and with the permission check removed
from the menu entry the no-permission test does; in a build of dev 22102e4a2 that ignores the
`load-url` parameter its test goes red. In the pre-fix build of 0.12.4 (bb8d49ff3 reverted) the
attached page vanishes from the composer without file upload.

A link typed after `#` and picked from the list is attached the same way. Every test except the
`load-url` one (an address parameter of the page) runs in the main chat and in a side chat
(`utils/chat_surface.py`), and the attach-and-send and `#` tests in every way a side chat starts
and every state it can be in. In a side chat neither way attached anything
(open-webui/open-webui#34019, fix PR open-webui/open-webui#34064); for those the dev c15f28649
build is the mutant.
"""

from __future__ import annotations

import dataclasses
from urllib.parse import quote

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.file_context import file_context_ticked
from harness.listener import listening, text_answer
from harness.web_retrieval import LOCAL_WEB_FETCH
from utils.chat_surface import EVERYWHERE, MAIN_CHAT, ChatOpener, ChatSurface
from utils.chat_ui import chat_input, expect_reply, link_dialog, open_add_context, send

pytestmark = [
    pytest.mark.journey,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

REFUSED = "Could not read content from"
NO_WEB_UPLOAD = {**LOCAL_WEB_FETCH, "USER_PERMISSIONS_CHAT_WEB_UPLOAD": "False"}


@pytest.fixture(scope="module")
def pages():
    with listening() as service:
        service.route("GET", "/tides", text_answer("<p>High tide at the harbour is at noon</p>"))
        service.route("GET", "/ferries", text_answer("<p>Ferries leave from the north pier</p>"))
        yield service


@pytest.fixture
def fetching_instance(instance_with):
    """An instance that may fetch local pages, with File Context ticked so a page's text is sent."""
    launched = instance_with(LOCAL_WEB_FETCH)
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    with file_context_ticked(admin_of(launched)):
        yield launched


@pytest.fixture
def restricted_instance(instance_with):
    launched = instance_with(NO_WEB_UPLOAD)
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    return launched


def _on(chat_for: ChatOpener, launched, preserve) -> ChatOpener:
    """`chat_for` on `launched`: its provider answers and side chats are allowed there."""
    admin = admin_of(launched)
    if chat_for.place != MAIN_CHAT:
        preserve("permissions", on=launched)
        with admin.client() as client:
            permissions = client.get("/api/v1/users/default/permissions").json()
            permissions["chat"].update(side_chat=True)
            saved = client.post("/api/v1/users/default/permissions", json=permissions)
        assert saved.status_code == 200, saved.text
    return dataclasses.replace(chat_for, upstream=launched.upstream, admin=admin)


@pytest.fixture
def fetching_chat_for(chat_for, fetching_instance, preserve) -> ChatOpener:
    return _on(chat_for, fetching_instance, preserve)


@pytest.fixture
def restricted_chat_for(chat_for, restricted_instance, preserve) -> ChatOpener:
    return _on(chat_for, restricted_instance, preserve)


def link(pages, path: str) -> str:
    return f"http://localhost:{pages.port}{path}"


def open_modal(chat: ChatSurface):
    expect(chat_input(chat)).to_be_visible(timeout=30_000)
    open_add_context(chat).get_by_role("menuitem", name="Attach Webpage").click()
    modal = chat.page.get_by_role("textbox", name="Webpage URLs")
    expect(modal).to_be_visible()
    return modal


def attach(chat: ChatSurface, *links: str) -> None:
    open_modal(chat).fill("\n".join(links))
    chat.page.get_by_role("button", name="Add", exact=True).click()


def open_chat(chat_for: ChatOpener, launched) -> ChatSurface:
    return chat_for(create_user(launched))


def composer(chat: ChatSurface) -> Locator:
    """The chat input box with its attachments; the refusal toast also names the link."""
    # the main chat's box is #message-input-container, a side tab's has an id of its own
    return chat.panel.locator("[data-composer-id]")


@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_a_link_becomes_an_attachment_and_its_text_reaches_the_model(
    fetching_chat_for, fetching_instance, pages
):
    """In a side chat: open-webui/open-webui#34019."""
    chat = open_chat(fetching_chat_for, fetching_instance)
    prompt = "when is high tide?"
    fetching_instance.upstream.queue(reply.text("at noon", match=reply.answering(prompt)))

    attach(chat, link(pages, "/tides"))

    expect(chat.panel.get_by_text(link(pages, "/tides"))).to_be_visible(timeout=30_000)
    send(chat, prompt)
    expect_reply(chat, "at noon")
    sent = str(fetching_instance.upstream.chat_requests()[-1]["messages"])
    assert "High tide at the harbour is at noon" in sent


@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_a_link_typed_after_a_hash_is_attached_and_its_text_reaches_the_model(
    fetching_chat_for, fetching_instance, pages
):
    """In a side chat: open-webui/open-webui#34019."""
    chat = open_chat(fetching_chat_for, fetching_instance)
    tides = link(pages, "/tides")
    prompt = "when is high tide, says the page?"
    fetching_instance.upstream.queue(reply.text("at noon", match=reply.answering(prompt)))
    expect(chat_input(chat)).to_be_visible(timeout=30_000)
    chat_input(chat).click()

    chat.page.keyboard.type(f"#{tides}")
    chat.page.get_by_role("button", name=tides).click()

    expect(
        composer(chat).get_by_text(tides), f"the link typed after # in the {chat.place} is missing"
    ).to_be_visible(timeout=30_000)
    expect(chat_input(chat)).not_to_contain_text(tides)
    send(chat, prompt)
    expect_reply(chat, "at noon")
    request = next(filter(reply.answering(prompt), fetching_instance.upstream.chat_requests()))
    assert "High tide at the harbour is at noon" in str(request["messages"]), (
        f"the page's text did not reach the model from the {chat.place}"
    )


def test_the_attachment_stays_on_the_message_once_it_is_sent(
    fetching_chat_for, fetching_instance, pages
):
    chat = open_chat(fetching_chat_for, fetching_instance)
    prompt = "what does the page say?"
    fetching_instance.upstream.queue(reply.text("about tides", match=reply.answering(prompt)))
    attach(chat, link(pages, "/tides"))
    expect(chat.panel.get_by_text(link(pages, "/tides"))).to_be_visible(timeout=30_000)

    send(chat, prompt)

    expect_reply(chat, "about tides")
    expect(chat.panel.get_by_text(link(pages, "/tides"))).to_be_visible()


def test_a_link_in_the_load_url_parameter_is_attached_once_confirmed(
    page_for, fetching_instance, pages
):
    page = page_for(create_user(fetching_instance))
    prompt = "what does the linked page say?"
    fetching_instance.upstream.queue(reply.text("about tides", match=reply.answering(prompt)))
    fetched_before = len(pages.requests_to("/tides"))

    page.goto(f"/?load-url={quote(link(pages, '/tides'))}")

    expect(link_dialog(page)).to_contain_text(link(pages, "/tides"))
    assert len(pages.requests_to("/tides")) == fetched_before, "fetched before it was confirmed"
    link_dialog(page).get_by_role("button", name="Confirm").click()
    expect(link_dialog(page)).to_be_hidden()
    expect(page.get_by_text(link(pages, "/tides"))).to_be_visible(timeout=30_000)
    send(page, prompt)
    expect_reply(page, "about tides")
    sent = str(fetching_instance.upstream.chat_requests()[-1]["messages"])
    assert "High tide at the harbour is at noon" in sent


def test_several_links_are_attached_one_per_line_and_repeats_once(
    fetching_chat_for, fetching_instance, pages
):
    chat = open_chat(fetching_chat_for, fetching_instance)
    tides, ferries = link(pages, "/tides"), link(pages, "/ferries")

    attach(chat, tides, ferries, tides)

    expect(chat.panel.get_by_text(ferries)).to_be_visible(timeout=30_000)
    expect(chat.panel.get_by_text(tides)).to_have_count(1)
    prompt = "summarise both"
    fetching_instance.upstream.queue(reply.text("done", match=reply.answering(prompt)))
    send(chat, prompt)
    expect_reply(chat, "done")
    sent = str(fetching_instance.upstream.chat_requests()[-1]["messages"])
    assert "High tide at the harbour is at noon" in sent
    assert "Ferries leave from the north pier" in sent


def test_text_that_is_not_a_link_is_refused_and_the_modal_stays_open(
    fetching_chat_for, fetching_instance
):
    chat = open_chat(fetching_chat_for, fetching_instance)
    page = chat.page

    attach(chat, "not a link", "ftp://example.com/file")

    expect(page.get_by_text("Please enter a valid URL.")).to_be_visible()
    expect(page.get_by_role("textbox", name="Webpage URLs")).to_be_visible()


def test_a_page_the_server_cannot_read_is_refused_and_not_attached(
    fetching_chat_for, fetching_instance, pages
):
    chat = open_chat(fetching_chat_for, fetching_instance)
    missing = link(pages, "/nowhere")

    attach(chat, missing)

    expect(chat.page.get_by_text(REFUSED)).to_be_visible(timeout=30_000)
    expect(composer(chat).get_by_text(missing)).to_have_count(0)


def test_the_modal_closes_without_attaching_anything(fetching_chat_for, fetching_instance, pages):
    chat = open_chat(fetching_chat_for, fetching_instance)
    page = chat.page
    open_modal(chat).fill(link(pages, "/tides"))

    page.get_by_role("button", name="Close modal").click()

    expect(page.get_by_role("textbox", name="Webpage URLs")).to_have_count(0)
    expect(page.get_by_text(link(pages, "/tides"))).to_have_count(0)


def test_an_attached_page_stays_without_the_file_upload_permission(
    fetching_chat_for, fetching_instance, pages, preserve
):
    """#32701 (fix bb8d49ff3): without file upload the composer dropped an attached page."""
    preserve("permissions", on=fetching_instance)
    with admin_of(fetching_instance).client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"]["file_upload"] = False
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text
    chat = open_chat(fetching_chat_for, fetching_instance)
    prompt = "when is high tide, going by the page?"
    fetching_instance.upstream.queue(reply.text("at noon", match=reply.answering(prompt)))

    attach(chat, link(pages, "/tides"))

    expect(chat.panel.get_by_text(link(pages, "/tides"))).to_be_visible(timeout=30_000)
    send(chat, prompt)
    expect_reply(chat, "at noon")
    request = next(filter(reply.answering(prompt), fetching_instance.upstream.chat_requests()))
    assert "High tide at the harbour is at noon" in str(request["messages"])


def test_a_user_without_the_web_upload_permission_cannot_open_the_modal(
    restricted_chat_for, restricted_instance
):
    chat = open_chat(restricted_chat_for, restricted_instance)
    page = chat.page
    expect(chat_input(chat)).to_be_visible(timeout=30_000)
    open_add_context(chat)

    page.get_by_role("menuitem", name="Attach Webpage").hover()

    expect(page.get_by_text("You do not have permission to upload web content.")).to_be_visible()
    page.get_by_role("menuitem", name="Attach Webpage").click()
    expect(page.get_by_role("textbox", name="Webpage URLs")).to_have_count(0)


def test_an_admin_can_open_the_modal_where_web_upload_is_off_by_default(
    restricted_chat_for, restricted_instance
):
    open_modal(restricted_chat_for(admin_of(restricted_instance)))
