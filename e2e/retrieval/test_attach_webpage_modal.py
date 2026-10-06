"""The Attach Webpage modal in the composer: links go in, pages come out as attachments.

The plus menu offers Attach Webpage; the modal takes one link per line, refuses text that is not
an http(s) link, and the page behind each link is fetched by the server and shown in the composer
as an attachment named by its link. The text of the page reaches the model with the next message,
and a page the server cannot read is refused with a toast and leaves no attachment. A link in the
`load-url` parameter of a chat link is attached the same way when the chat opens. Without the web
upload permission the menu entry does nothing for a user and still opens for an admin. The pages
are a local service, on an instance that may fetch loopback addresses.

Discriminates: passes on the 176d31d1d build; with the link validation removed the refused-text
test goes red, with the de-duplication removed the several-links test does, with the handoff to
the chat or the modal's close button removed the attachment, refusal and close tests do, and with
the permission check removed from the menu entry the no-permission test does; in a build that
ignores the `load-url` parameter its test goes red.
"""

from __future__ import annotations

from urllib.parse import quote

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.listener import listening, text_answer
from harness.web_retrieval import LOCAL_WEB_FETCH
from utils.chat_ui import chat_input, expect_reply, send

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
    launched = instance_with(LOCAL_WEB_FETCH)
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    return launched


@pytest.fixture
def restricted_instance(instance_with):
    launched = instance_with(NO_WEB_UPLOAD)
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    return launched


def link(pages, path: str) -> str:
    return f"http://localhost:{pages.port}{path}"


def open_modal(page: Page):
    expect(chat_input(page)).to_be_visible(timeout=30_000)
    page.get_by_role("button", name="More", exact=True).last.click()
    page.get_by_role("button", name="Attach Webpage").click()
    modal = page.get_by_role("textbox", name="Webpage URLs")
    expect(modal).to_be_visible()
    return modal


def attach(page: Page, *links: str) -> None:
    open_modal(page).fill("\n".join(links))
    page.get_by_role("button", name="Add", exact=True).click()


def open_chat(page_for, launched) -> Page:
    return page_for(create_user(launched))


def test_a_link_becomes_an_attachment_and_its_text_reaches_the_model(
    page_for, fetching_instance, pages
):
    page = open_chat(page_for, fetching_instance)
    prompt = "when is high tide?"
    fetching_instance.upstream.queue(reply.text("at noon", match=reply.answering(prompt)))

    attach(page, link(pages, "/tides"))

    expect(page.get_by_text(link(pages, "/tides"))).to_be_visible(timeout=30_000)
    send(page, prompt)
    expect_reply(page, "at noon")
    sent = str(fetching_instance.upstream.chat_requests()[-1]["messages"])
    assert "High tide at the harbour is at noon" in sent


def test_the_attachment_stays_on_the_message_once_it_is_sent(page_for, fetching_instance, pages):
    page = open_chat(page_for, fetching_instance)
    prompt = "what does the page say?"
    fetching_instance.upstream.queue(reply.text("about tides", match=reply.answering(prompt)))
    attach(page, link(pages, "/tides"))
    expect(page.get_by_text(link(pages, "/tides"))).to_be_visible(timeout=30_000)

    send(page, prompt)

    expect_reply(page, "about tides")
    expect(page.get_by_text(link(pages, "/tides"))).to_be_visible()


def test_a_link_in_the_load_url_parameter_is_attached_when_the_chat_opens(
    page_for, fetching_instance, pages
):
    page = page_for(create_user(fetching_instance))
    prompt = "what does the linked page say?"
    fetching_instance.upstream.queue(reply.text("about tides", match=reply.answering(prompt)))

    page.goto(f"/?load-url={quote(link(pages, '/tides'))}")

    expect(page.get_by_text(link(pages, "/tides"))).to_be_visible(timeout=30_000)
    send(page, prompt)
    expect_reply(page, "about tides")
    sent = str(fetching_instance.upstream.chat_requests()[-1]["messages"])
    assert "High tide at the harbour is at noon" in sent


def test_several_links_are_attached_one_per_line_and_repeats_once(
    page_for, fetching_instance, pages
):
    page = open_chat(page_for, fetching_instance)
    tides, ferries = link(pages, "/tides"), link(pages, "/ferries")

    attach(page, tides, ferries, tides)

    expect(page.get_by_text(ferries)).to_be_visible(timeout=30_000)
    expect(page.get_by_text(tides)).to_have_count(1)
    prompt = "summarise both"
    fetching_instance.upstream.queue(reply.text("done", match=reply.answering(prompt)))
    send(page, prompt)
    expect_reply(page, "done")
    sent = str(fetching_instance.upstream.chat_requests()[-1]["messages"])
    assert "High tide at the harbour is at noon" in sent
    assert "Ferries leave from the north pier" in sent


def test_text_that_is_not_a_link_is_refused_and_the_modal_stays_open(page_for, fetching_instance):
    page = open_chat(page_for, fetching_instance)

    attach(page, "not a link", "ftp://example.com/file")

    expect(page.get_by_text("Please enter a valid URL.")).to_be_visible()
    expect(page.get_by_role("textbox", name="Webpage URLs")).to_be_visible()


def test_a_page_the_server_cannot_read_is_refused_and_not_attached(
    page_for, fetching_instance, pages
):
    page = open_chat(page_for, fetching_instance)
    missing = link(pages, "/nowhere")

    attach(page, missing)

    expect(page.get_by_text(REFUSED)).to_be_visible(timeout=30_000)
    expect(page.get_by_text(missing)).to_have_count(0)


def test_the_modal_closes_without_attaching_anything(page_for, fetching_instance, pages):
    page = open_chat(page_for, fetching_instance)
    open_modal(page).fill(link(pages, "/tides"))

    page.get_by_role("button", name="Close modal").click()

    expect(page.get_by_role("textbox", name="Webpage URLs")).to_have_count(0)
    expect(page.get_by_text(link(pages, "/tides"))).to_have_count(0)


def test_a_user_without_the_web_upload_permission_cannot_open_the_modal(
    page_for, restricted_instance
):
    page = open_chat(page_for, restricted_instance)
    expect(chat_input(page)).to_be_visible(timeout=30_000)
    page.get_by_role("button", name="More", exact=True).last.click()

    page.get_by_role("button", name="Attach Webpage").hover()

    expect(page.get_by_text("You do not have permission to upload web content.")).to_be_visible()
    page.get_by_role("button", name="Attach Webpage").click()
    expect(page.get_by_role("textbox", name="Webpage URLs")).to_have_count(0)


def test_an_admin_can_open_the_modal_where_web_upload_is_off_by_default(
    page_for, restricted_instance
):
    open_modal(page_for(admin_of(restricted_instance)))
