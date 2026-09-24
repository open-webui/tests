"""Regression: the community stats window handed a user's chat statistics to any site.

open-webui PR #29918 (commit 08578557d): Open WebUI opened with `?sync=true` shows the window
that shares usage statistics with the openwebui.com community. It answered `verify:chat`
requests from whichever page had opened it, without checking that page's origin, and posted
every message, the statistics of a synced account included, with a `'*'` target. Any site that
opened the window received them. The fix answers only the community origins and addresses every
message to a community origin.

Each test serves a stand-in opener page on an origin of its choosing through Playwright routing
and opens the stats window from it as a fresh account with one chat.

Twin of unit/frontend/test_stats_window_origin.py.

Discriminates: passes on the bbfa876af build; with the origin check removed and the `'*'`
targets restored in SyncStatsModal.svelte, the foreign opener receives the chat's statistics
from both `verify:chat` and the Sync button.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.chat import ask

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

COMMUNITY_ORIGIN = "http://localhost:9999"
FOREIGN_ORIGIN = "http://stats-collector.test"

OPENER_PAGE = """<!doctype html><title>opener</title><script>
window.received = [];
window.addEventListener('message', (event) => window.received.push(event.data));
</script>"""

RECORD_DELIVERIES = """() => {
    window.deliveredFrom = [];
    window.addEventListener('message', (event) => window.deliveredFrom.push(event.origin));
}"""

# not statistics: the window announces itself to any opener, and the test's own end marker
NOT_STATISTICS = ("loaded", "end of messages")


@pytest.fixture
def community_sharing(admin, preserve):
    preserve("admin_config")
    with admin.client() as client:
        current = client.get("/api/v1/auths/admin/config").json()
        saved = client.post(
            "/api/v1/auths/admin/config", json={**current, "ENABLE_COMMUNITY_SHARING": True}
        )
    saved.raise_for_status()


@pytest.fixture
def owner(make_user, upstream):
    """A fresh account with one chat, and that chat's id."""
    account = make_user()
    upstream.queue(reply.text("hi"))
    with account.client() as client:
        turn, _ = ask(client, "hello")
    return account, turn.chat_id


@pytest.fixture
def open_stats_window(community_sharing, page_for, instance):
    def open_from(origin: str, account) -> tuple[Page, Page]:
        app_page = page_for(account)  # stores the account's session for the app origin
        context = app_page.context
        context.route(
            f"{origin}/**", lambda route: route.fulfill(content_type="text/html", body=OPENER_PAGE)
        )
        opener = context.new_page()
        opener.goto(f"{origin}/")
        with opener.expect_popup() as popup_info:
            opener.evaluate(
                "url => { window.statsWindow = window.open(url); }",
                f"{instance.base_url}/?sync=true",
            )
        stats_window = popup_info.value
        expect(stats_window.get_by_text("Sync Usage Stats")).to_be_visible()
        opener.wait_for_function("() => window.received.includes('loaded')")
        return opener, stats_window

    return open_from


def request_chat_stats(opener: Page, stats_window: Page, chat_id: str) -> None:
    """Post `verify:chat` from the opener and wait until the stats window has received it."""
    stats_window.evaluate(RECORD_DELIVERIES)
    opener.evaluate(
        "chatId => window.statsWindow.postMessage("
        "{type: 'verify:chat', data: {id: chatId}, requestId: 'r1'}, '*')",
        chat_id,
    )
    stats_window.wait_for_function(
        "origin => window.deliveredFrom.includes(origin)",
        arg=opener.evaluate("() => location.origin"),
    )


def sync(opener: Page, stats_window: Page) -> None:
    """Press Sync and wait until everything the window sent the opener has arrived."""
    stats_window.get_by_role("button", name="Sync", exact=True).click()
    expect(stats_window.get_by_text("Sync Complete!")).to_be_visible()
    # one window's messages to another arrive in the order they were sent
    stats_window.evaluate("() => window.opener.postMessage('end of messages', '*')")
    opener.wait_for_function("() => window.received.includes('end of messages')")


def statistics_received(opener: Page) -> list:
    return [
        message
        for message in opener.evaluate("() => window.received")
        if message not in NOT_STATISTICS
    ]


def test_a_foreign_opener_gets_no_answer_to_verify_chat(open_stats_window, owner):
    account, chat_id = owner
    opener, stats_window = open_stats_window(FOREIGN_ORIGIN, account)

    request_chat_stats(opener, stats_window, chat_id)
    opener.wait_for_timeout(2000)  # a community opener has its answer well within this

    assert statistics_received(opener) == [], "the stats window answered a foreign opener"


def test_the_community_opener_gets_the_answer_to_verify_chat(open_stats_window, owner):
    account, chat_id = owner
    opener, stats_window = open_stats_window(COMMUNITY_ORIGIN, account)

    request_chat_stats(opener, stats_window, chat_id)
    opener.wait_for_function(
        "() => window.received.some((message) => message?.type === 'verify:chat:response')"
    )

    [answer] = statistics_received(opener)
    assert answer["type"] == "verify:chat:response"
    assert answer["chatId"] == answer["data"]["id"] == chat_id
    assert answer["requestId"] == "r1"


def test_a_foreign_opener_gets_nothing_from_a_sync(open_stats_window, owner):
    account, _chat_id = owner
    opener, stats_window = open_stats_window(FOREIGN_ORIGIN, account)

    sync(opener, stats_window)

    assert statistics_received(opener) == [], "the sync sent the statistics to a foreign opener"


def test_the_community_opener_gets_the_synced_statistics(open_stats_window, owner):
    account, chat_id = owner
    opener, stats_window = open_stats_window(COMMUNITY_ORIGIN, account)

    sync(opener, stats_window)

    received = statistics_received(opener)
    kinds = [message["type"] for message in received]
    assert kinds[0] == "sync:start" and kinds[-1] == "sync:complete", kinds
    synced_chats = [
        item["id"]
        for message in received
        if message["type"] == "sync:stats:chats"
        for item in message["data"]["items"]
    ]
    assert synced_chats == [chat_id]
