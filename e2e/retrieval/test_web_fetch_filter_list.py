"""Regression: a quoted `WEB_FETCH_FILTER_LIST` entry blocked every link a user attached.

open-webui 0.11.0 fix `18719fef9` (#26910, issue #26908): Docker Compose list syntax passes
surrounding quotes through, so `"localhost"` became an allow entry with the quotes in it, which
matches no host, and a non-empty allow list refuses everything else: every link a user attached
to a chat was refused. The chat page attaches the link named in its `load-url` query parameter,
the way a "chat about this page" link does, once the person confirms it (since 0ffd86967).

Twin of integration/retrieval/test_web_fetch_filter_list.py.

Discriminates: passes on dev ef67cc3fa; with the pre-fix parser (entries only stripped of
whitespace) the allowed link is refused with a "could not read" toast; the unlisted host is refused
on both.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness.actors import create_user
from harness.listener import listening, text_answer
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import link_dialog

pytestmark = [
    pytest.mark.regression,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

QUOTED_ALLOW_LIST = {"ENABLE_LOCAL_WEB_FETCH": "true", "WEB_FETCH_FILTER_LIST": '"localhost"'}
REFUSED = "Could not read content from"


@pytest.fixture(scope="module")
def page_host():
    with listening() as service:
        service.route("GET", "/notes", text_answer("<p>Local notes behind the allow list</p>"))
        yield service


@pytest.fixture
def filtering_instance(instance_with):
    launched = instance_with(QUOTED_ALLOW_LIST)
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    return launched


def is_attachment_answer(response) -> bool:
    return "/api/v1/retrieval/process/url" in response.url


def open_chat_about(page_for, launched, link: str):
    """The chat page attaching `link`, once confirmed and answered by the server."""
    page = page_for(create_user(launched))
    page.goto(f"/?models={MOCK_MODEL_ID}&load-url={link}")
    with page.expect_response(is_attachment_answer, timeout=30_000):
        link_dialog(page).get_by_role("button", name="Confirm").click()
    return page


def test_a_link_on_a_quoted_allow_list_is_attached(page_for, filtering_instance, page_host):
    link = f"http://localhost:{page_host.port}/notes"

    page = open_chat_about(page_for, filtering_instance, link)

    expect(page.get_by_text(REFUSED)).to_have_count(0)
    expect(page.get_by_text(link)).to_be_visible()


def test_a_link_missing_from_the_allow_list_is_refused(page_for, filtering_instance, page_host):
    link = f"http://127.0.0.1:{page_host.port}/notes"

    page = open_chat_about(page_for, filtering_instance, link)

    expect(page.get_by_text(REFUSED)).to_be_visible(timeout=30_000)
