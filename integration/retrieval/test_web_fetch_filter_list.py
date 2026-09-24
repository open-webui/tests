"""Regression: a quoted `WEB_FETCH_FILTER_LIST` entry must not block every fetch.

open-webui 0.11.0 fix `18719fef9` (#26910, issue #26908): `get_allow_block_lists` kept each entry
verbatim. Docker Compose list syntax passes surrounding quotes through, so `"localhost"` became
the allow entry `"localhost"`, quotes included, which matches no host; and since a non-empty
allow list refuses every host it does not match, one stray quote blocked every web address.
The fix strips the quotes and whitespace and drops entries left empty.

The instance boots with the entry quoted the way Compose hands it over, and local fetching on.

Twin of unit/retrieval/test_web_fetch_filter_list.py, which keeps the other parsing cases
(quoted block entries, quotes inside the `!`, empty entries): each needs a boot of its own here.

Discriminates: passes on dev bbfa876af; keeping entries verbatim fails the allowed host (the
quoted entry matches nothing, so the page is refused); the unlisted host is refused on both.
"""

from __future__ import annotations

import pytest

from harness.listener import text_answer

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

QUOTED_ALLOW_LIST = {"ENABLE_LOCAL_WEB_FETCH": "true", "WEB_FETCH_FILTER_LIST": '"localhost"'}
PAGE_TEXT = "Local notes behind the allow list"


@pytest.fixture(scope="module")
def filtering_instance(instance_with):
    return instance_with(QUOTED_ALLOW_LIST)


@pytest.fixture
def preview(filtering_instance, listener):
    """`preview(host)` attaches the listener's page under that host name, without storing it."""
    listener.route("GET", "/notes", text_answer(f"<p>{PAGE_TEXT}</p>"))
    with filtering_instance.client() as client:

        def attach(host: str):
            url = f"http://{host}:{listener.port}/notes"
            return client.post("/api/v1/retrieval/process/web?process=false", json={"url": url})

        yield attach


def test_a_host_on_a_quoted_allow_list_is_fetched(preview):
    previewed = preview("localhost")

    assert previewed.status_code == 200, f"the allow-listed host was refused: {previewed.text}"
    assert PAGE_TEXT in previewed.json()["content"]


def test_a_host_missing_from_the_allow_list_is_refused(preview, listener):
    refused = preview("127.0.0.1")

    assert refused.status_code == 400, refused.text
    assert listener.requests_to("/notes") == []
