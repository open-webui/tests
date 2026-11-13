"""Journey: an HTML page on a terminal opens as a rendered page inside a frame.

Two kinds of terminal show it two ways. A terminal the admin connected: the frame loads the
page through Open WebUI, and in a saved chat the address names the chat. A terminal of a
Terminals orchestrator: the page first asks for a link signed for this person, and the frame
loads the page and the picture beside it from that link, which answers nobody once it is
altered. In each the frame is sandboxed by the person's Interface settings: a page's script runs
unless they switched scripts off, and it reaches neither the app's storage nor, on a connected
terminal, the files beside it unless they allowed the same origin (the docs: "same-origin
requests back to Open WebUI fail"). Source shows the page's text, where an edit is saved and then
shows in the preview. A terminal a person added in their own settings, read by the browser, was
removed from Open WebUI in 44d55bd0f and its tests with it.

Discriminates: all pass on dev c15f28649 with Terminals 0.4.1 and Open Terminal 0.17.7. On one
frontend build of it with these mutations each named test fails: a connected terminal's page put
into the frame as text instead of loaded from Open WebUI, and the chat left out of its address
(connected terminal); the frame's sandbox always allowing scripts and the same origin (frame
settings); the signed link never asked for (orchestrator terminal); Save missing from an HTML
page's source (source).
"""

from __future__ import annotations

import re
import shutil
import uuid

import httpx
import pytest
from playwright.sync_api import Locator, expect

from harness import sample_documents as samples
from harness import upstream as reply
from harness.backends import wait_until
from harness.terminal_server import TERMINAL_SERVERS_CONFIG
from utils.cached_chat import choose_terminal, open_files_tab
from utils.chat_ui import expect_reply, send, split_screen
from utils.file_browser import file_browser
from utils.file_views import connect_for, files_in, files_of, open_file

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

HOME = "/home/user"
REPORT = (
    "<html><head><title>Report</title></head><body><h1>Cider report</h1>"
    '<img src="art/label.png" alt="the label">'
    "<script>"
    "let seen; try { seen = localStorage.getItem('token') ? 'the token' : 'no token' }"
    " catch (refused) { seen = 'refused' }"
    "document.body.append('script ran, storage: ' + seen)"
    "</script></body></html>"
)
REWRITTEN = "<html><body><h1>Perry report</h1></body></html>"


@pytest.fixture
def person(make_user):
    return make_user()


@pytest.fixture
def site(open_terminal):
    """A folder of this test's own in the terminal's home with the page and its picture."""
    path = open_terminal.home / f"site-{uuid.uuid4().hex[:6]}"
    (path / "art").mkdir(parents=True)
    (path / "report.html").write_text(REPORT)
    (path / "art" / "label.png").write_bytes(samples.png(64, 48))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _shown_page(browser: Locator):
    frame = browser.get_by_title("HTML Preview")
    expect(frame).to_be_visible()
    return browser.frame_locator('iframe[title="HTML Preview"]')


def _save_settings(account, **ui) -> None:
    """Set Interface settings of the account before its page opens; they are patched in."""
    with account.client() as client:
        saved = client.post("/api/v1/users/user/settings/update", json={"ui": ui})
    assert saved.status_code == 200, f"saving the settings failed: {saved.text}"


def test_an_html_page_on_a_connected_terminal_runs_its_script_away_from_the_apps_storage(
    site, person, page_for, admin, preserve, open_terminal
):
    preserve(TERMINAL_SERVERS_CONFIG)
    browser = files_of(page_for, admin, person, open_terminal, site.name)

    open_file(browser, "report.html")

    shown = _shown_page(browser)
    expect(shown.get_by_role("heading", name="Cider report")).to_be_visible()
    expect(shown.get_by_text("script ran, storage: refused")).to_be_visible()
    # the frame was given the page's address on Open WebUI, not its text
    frame = browser.get_by_title("HTML Preview")
    expect(frame).to_have_attribute("src", re.compile(rf"/files/serve/.*{site.name}/report\.html$"))


def test_an_html_page_opened_in_a_saved_chat_is_fetched_under_that_chat(
    site, person, page_for, admin, preserve, open_terminal, upstream
):
    preserve(TERMINAL_SERVERS_CONFIG)
    browser = files_of(page_for, admin, person, open_terminal, site.name)
    page = browser.page
    split_screen(page)
    prompt = f"is the report ready {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("It is.", match=reply.answering(prompt)))
    send(page, prompt)
    expect_reply(page, "It is.")
    expect(page).to_have_url(re.compile(r"/c/[0-9a-f-]{36}"))
    chat_id = page.url.rsplit("/c/", 1)[1].split("?")[0]

    with page.expect_request(lambda sent: sent.url.endswith("/report.html")) as fetched:
        open_file(file_browser(page), "report.html")

    assert f"/chats/{chat_id}/files/serve/" in fetched.value.url, (
        f"the page of a saved chat was fetched without naming the chat: {fetched.value.url}"
    )
    shown = _shown_page(file_browser(page))
    expect(shown.get_by_role("heading", name="Cider report")).to_be_visible()


def test_with_the_same_origin_allowed_a_page_reaches_the_picture_beside_it_and_the_storage(
    site, person, page_for, admin, preserve, open_terminal
):
    preserve(TERMINAL_SERVERS_CONFIG)
    name = connect_for(admin, person, open_terminal)
    _save_settings(person, iframeSandboxAllowSameOrigin=True)
    browser = files_in(page_for(person), name, site.name)

    open_file(browser, "report.html")

    shown = _shown_page(browser)
    expect(shown.get_by_text("script ran, storage: the token")).to_be_visible()
    expect(shown.get_by_role("img", name="the label")).to_have_js_property("naturalWidth", 64)


def test_with_scripts_switched_off_a_pages_script_does_not_run(
    site, person, page_for, admin, preserve, open_terminal
):
    preserve(TERMINAL_SERVERS_CONFIG)
    name = connect_for(admin, person, open_terminal)
    _save_settings(person, iframeSandboxAllowScripts=False)
    browser = files_in(page_for(person), name, site.name)

    open_file(browser, "report.html")

    shown = _shown_page(browser)
    expect(shown.get_by_role("heading", name="Cider report")).to_be_visible()
    expect(shown.get_by_text(re.compile("script ran"))).to_have_count(0)


def test_an_html_page_on_an_orchestrator_terminal_loads_from_a_link_signed_for_the_person(
    person, shared_terminal, page_for
):
    name = f"Workbench {uuid.uuid4().hex[:6]}"
    terminal_id = shared_terminal(person, name=name)
    with person.client() as client:
        for directory, file_name, content in (
            (HOME, "report.html", REPORT.encode()),
            (f"{HOME}/art", "label.png", samples.png(64, 48)),
        ):
            made = client.post(
                f"/api/v1/terminals/{terminal_id}/files/mkdir", json={"path": directory}
            )
            assert made.status_code == 200, f"making {directory}: {made.text}"
            uploaded = client.post(
                f"/api/v1/terminals/{terminal_id}/files/upload",
                params={"directory": directory},
                files={"file": (file_name, content, "application/octet-stream")},
            )
            assert uploaded.status_code == 200, f"uploading {file_name}: {uploaded.text}"
    page = page_for(person)
    choose_terminal(page, name)
    browser = open_files_tab(page)

    with page.expect_response(lambda got: got.url.endswith("/files/preview")) as link:
        open_file(browser, "report.html")

    assert link.value.status == 200, f"no signed link for the page: {link.value.text()}"
    shown = _shown_page(browser)
    expect(shown.get_by_role("heading", name="Cider report")).to_be_visible()
    # the link itself says who asked, so the sandboxed page gets the picture beside it as well
    expect(shown.get_by_role("img", name="the label")).to_have_js_property("naturalWidth", 64)
    expect(shown.get_by_text("script ran, storage: refused")).to_be_visible()
    signed = browser.get_by_title("HTML Preview").get_attribute("src")
    forged = re.sub(r"/browser/view/files/[^/]+/", "/browser/view/files/forged/", signed)
    with httpx.Client(base_url=person.base_url) as signed_out:
        fetched = signed_out.get(signed)
        refused = signed_out.get(forged)
    assert fetched.status_code == 200 and "Cider report" in fetched.text, (
        f"the signed link answered {fetched.status_code} to a client without a session"
    )
    assert refused.status_code == 401, f"an altered link was answered with {refused.status_code}"


def test_the_source_of_an_html_page_is_edited_saved_and_then_shown(
    site, person, page_for, admin, preserve, open_terminal
):
    preserve(TERMINAL_SERVERS_CONFIG)
    browser = files_of(page_for, admin, person, open_terminal, site.name)
    page = browser.page
    open_file(browser, "report.html")
    expect(_shown_page(browser).get_by_role("heading", name="Cider report")).to_be_visible()
    # a rendered page is only looked at
    expect(browser.get_by_role("button", name="Save", exact=True)).to_have_count(0)

    browser.get_by_role("button", name="Source", exact=True).click()
    editor = browser.get_by_role("textbox")
    expect(editor).to_contain_text("<h1>Cider report</h1>")
    editor.click()
    page.keyboard.press("Control+a")
    # typed in one piece: key by key the editor would close each tag itself
    page.keyboard.insert_text(REWRITTEN)
    browser.get_by_role("button", name="Save", exact=True).click()

    saved = site / "report.html"
    assert wait_until(lambda: saved.read_text() == REWRITTEN, 10), (
        f"the edit was not saved to the terminal: {saved.read_text()!r}"
    )
    browser.get_by_role("button", name="Preview", exact=True).click()
    expect(_shown_page(browser).get_by_role("heading", name="Perry report")).to_be_visible()
