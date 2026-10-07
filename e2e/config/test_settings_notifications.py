"""Journey: Settings > Notifications, the browser notification for a reply finished elsewhere.

A user who allows notifications switches Browser Notifications on, starts a chat and moves to a new
chat before the reply is done; the finished reply then shows as a browser notification as well as
the in-page toast. The switch is still on after a reload, and a second account that never switched
it on gets only the toast. Clicking the toast opens the chat whose reply finished and its close
button dismisses it without leaving the page. The toast plays the notification sound until the user
switches Notification Sound off. A user whose browser denies the permission is told so and the
setting stays off; on dev the switch itself still flips on, so that test stays red until the switch
follows the refusal.

Discriminates: passes on dev 176d31d1d; in a frontend copy, showing the finished-reply
notification without reading `notificationEnabled` turns the second account's check red; on
ebc6add67 builds, with the toast's click going nowhere the toast test fails, with its dismiss
button doing nothing the dismiss test fails, with the sound setting not read the sound test fails,
and with the denied switch put back to off the denied test turns green.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import conversation, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ANSWER = "The tide turns at noon."
DENIED = "Response notifications cannot be activated as the website permissions have been denied."

# keeps what the page shows, so a test can read the notifications that reached the browser
RECORD_NOTIFICATIONS = """
window.shownNotifications = [];
window.Notification = new Proxy(window.Notification, {
    construct(target, args) {
        window.shownNotifications.push({ title: args[0], body: args[1]?.body });
        return Reflect.construct(target, args);
    }
});
"""


def recording_page(page_for, account, allow: bool) -> Page:
    page = page_for(account)
    if allow:
        page.context.grant_permissions(["notifications"])
    page.add_init_script(RECORD_NOTIFICATIONS)
    page.goto("/")
    return page


def notifications_switch(page: Page) -> Locator:
    page.goto("/?settings=notifications")
    switch = page.get_by_role("switch", name="Browser Notifications")
    expect(switch).to_be_visible()
    return switch


def finish_a_reply_elsewhere(page: Page, upstream, question: str) -> list[dict]:
    upstream.queue(
        reply.text(
            ["The tide ", "turns at noon."], chunk_delay=1.0, match=reply.answering(question)
        )
    )
    page.goto("/")
    send(page, question)
    expect(page).to_have_url(re.compile(r"/c/"))
    page.get_by_role("link", name="New Chat").click()
    expect(page.get_by_text(ANSWER)).to_be_visible()
    return page.evaluate("window.shownNotifications")


def test_a_reply_finished_in_another_chat_shows_a_browser_notification(
    page_for, make_user, upstream
):
    page = recording_page(page_for, make_user(), allow=True)
    switch = notifications_switch(page)
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        switch.click()
    expect(switch).to_have_attribute("aria-checked", "true")

    shown = finish_a_reply_elsewhere(page, upstream, "When does the tide turn?")
    assert [notification["body"] for notification in shown] == [ANSWER]

    page.reload()
    expect(notifications_switch(page)).to_have_attribute("aria-checked", "true")

    other = recording_page(page_for, make_user(), allow=True)
    expect(notifications_switch(other)).to_have_attribute("aria-checked", "false")
    assert finish_a_reply_elsewhere(other, upstream, "When does the tide turn today?") == []


# keeps every sound the page starts, so a test can tell whether the notification sound played
RECORD_SOUNDS = """
window.playedSounds = [];
window.Audio = new Proxy(window.Audio, {
    construct(target, args) {
        window.playedSounds.push(args[0]);
        return Reflect.construct(target, args);
    }
});
"""


def test_a_denied_permission_keeps_browser_notifications_off(page_for, make_user):
    account = make_user()
    page = recording_page(page_for, account, allow=False)
    switch = notifications_switch(page)

    switch.click()

    expect(page.get_by_text(DENIED)).to_be_visible()
    with account.client() as client:
        stored = client.get("/api/v1/users/user/settings").json()["ui"]
    assert stored.get("notificationEnabled") is not True
    # red on dev: the switch flips itself on though the setting stayed off
    expect(switch).to_have_attribute("aria-checked", "false")


def test_clicking_the_toast_opens_the_chat_whose_reply_finished(page_for, make_user, upstream):
    page = recording_page(page_for, make_user(), allow=False)
    finish_a_reply_elsewhere(page, upstream, "When does the tide turn tonight?")
    expect(page).not_to_have_url(re.compile(r"/c/"))

    # the toast's own status sits inside the toaster's live region
    page.get_by_role("status").filter(has_text=ANSWER).last.click()

    expect(page).to_have_url(re.compile(r"/c/"))
    expect(conversation(page)).to_contain_text("When does the tide turn tonight?")
    expect_reply(page, ANSWER)


def test_dismissing_the_toast_closes_it_and_stays_in_the_new_chat(page_for, make_user, upstream):
    page = recording_page(page_for, make_user(), allow=False)
    finish_a_reply_elsewhere(page, upstream, "When does the tide turn at night?")
    toast = page.get_by_role("status").filter(has_text=ANSWER).last

    toast.hover()
    toast.get_by_role("button", name="Dismiss notification").click()

    expect(page.get_by_text(ANSWER)).to_have_count(0)
    expect(page).not_to_have_url(re.compile(r"/c/"))


def sounds_played(page: Page) -> list[str]:
    return page.evaluate("window.playedSounds")


def test_the_toast_plays_the_notification_sound_until_it_is_switched_off(
    page_for, make_user, upstream
):
    page = recording_page(page_for, make_user(), allow=False)
    page.add_init_script(RECORD_SOUNDS)
    page.goto("/")

    finish_a_reply_elsewhere(page, upstream, "When does the tide turn at dawn?")
    assert sounds_played(page) == ["/audio/notification.mp3"]

    page.goto("/?settings=notifications")
    sound = page.get_by_role("switch", name="Notification Sound")
    expect(sound).to_have_attribute("aria-checked", "true")
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        sound.click()
    expect(sound).to_have_attribute("aria-checked", "false")
    page.reload()
    finish_a_reply_elsewhere(page, upstream, "When does the tide turn at dusk?")
    assert sounds_played(page) == []
