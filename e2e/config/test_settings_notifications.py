"""Journey: Settings > Notifications, the browser notification for a reply finished elsewhere.

A user who allows notifications switches Browser Notifications on, starts a chat and moves to a new
chat before the reply is done; the finished reply then shows as a browser notification as well as
the in-page toast. The switch is still on after a reload, and a second account that never switched
it on gets only the toast. Clicking the toast opens the chat whose reply finished and its close
button dismisses it without leaving the page. The toast plays the notification sound until the user
switches Notification Sound off. A user whose browser denies the permission is told so and the
setting stays off; on dev the switch itself still flips on, so that test stays red until the switch
follows the refusal (open-webui/open-webui#32024, fixed by 2cdf01521).

Discriminates: passes on dev 176d31d1d; in a frontend copy, showing the finished-reply
notification without reading `notificationEnabled` turns the second account's check red; on
ebc6add67 builds, with the toast's click going nowhere the toast test fails, with its dismiss
button doing nothing the dismiss test fails, with the sound setting not read the sound test fails,
and with the denied switch put back to off the denied test turns green.

Since 13e0fd2c5 clicking a browser notification opens what it is about: the chat whose reply
finished, the channel of a new message (titled with the writer and the channel) and the calendar
for an event reminder. Since b6a77a0f1 the notification sound is a short tune the page plays itself
in the style picked under Notification sound style (Chime by default, Soft, Bell or Tap); the play
button previews it, the choice is kept over a reload and a finished reply plays it. The style row
hides while the sound is off.

Discriminates: on dev f659c0d12, in a frontend copy whose notification click opens nothing the
reply, channel and calendar tests fail at the click, one whose toast plays no sound fails the
sound test, and one that plays the Chime for every style fails the style test at the preview.

A reply that finishes while the window is in the background raises the browser notification and
the toast in the main chat and in a chat in a side tab alike, and a reply read in view raises
none. A side chat raised nothing at all (open-webui/open-webui#34025).

Discriminates: the side chat variants fail on dev 262a8c6f7 and pass on the fix branch
fix/side-chat-background-notification; the main chat variant passes on both.
"""

from __future__ import annotations

import re
import threading
import time
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from conftest import AppConfig
from e2e.conftest import APP_LOADED
from harness import upstream as reply
from harness.backends import wait_until
from harness.calendar_api import MINUTE_NS, create_event, default_calendar_id
from harness.channel_quotes import enable_channels, group_channel, post_message
from utils.chat_surface import (
    ASKED_ASIDE,
    DRAGGED_TAB,
    MAIN_CHAT,
    REPLY_TIMEOUT_MS,
    SIDE_CHAT,
    TEMPORARY_SIDE,
)
from utils.chat_ui import chat_input, conversation, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ANSWER = "The tide turns at noon."
DENIED = "Response notifications cannot be activated as the website permissions have been denied."

# keeps what the page shows, so a test can read the notifications that reached the browser;
# the headless shell reports `permission` as denied even after a grant, so it reports the grant
RECORD_NOTIFICATIONS = """
window.shownNotifications = [];
window.browserNotifications = [];
window.Notification = new Proxy(window.Notification, {
    construct(target, args) {
        window.shownNotifications.push({ title: args[0], body: args[1]?.body });
        const notification = Reflect.construct(target, args);
        window.browserNotifications.push(notification);
        return notification;
    },
    get(target, name) {
        return name === 'permission' && ALLOWED ? 'granted' : Reflect.get(target, name);
    }
});
"""


def load(page: Page, path: str) -> None:
    """Open `path` afresh and wait, as `page_for` does, until the app has loaded."""
    page.goto(path)
    page.wait_for_function(APP_LOADED, timeout=AppConfig().navigation_timeout)


def recording_page(page_for, account, allow: bool) -> Page:
    page = page_for(account)
    if allow:
        page.context.grant_permissions(["notifications"])
    page.add_init_script(f"const ALLOWED = {str(allow).lower()};" + RECORD_NOTIFICATIONS)
    load(page, "/")
    return page


def click_the_browser_notification(page: Page) -> None:
    """Click the last browser notification shown, as the person does in their system tray."""
    page.evaluate("window.browserNotifications.at(-1).dispatchEvent(new Event('click'))")


def switch_browser_notifications_on(page: Page) -> None:
    switch = notifications_switch(page)
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        switch.click()
    expect(switch).to_have_attribute("aria-checked", "true")
    load(page, "/")


def notifications_switch(page: Page) -> Locator:
    load(page, "/?settings=notifications")
    switch = page.get_by_role("switch", name="Browser Notifications")
    expect(switch).to_be_visible()
    return switch


def finish_a_reply_elsewhere(page: Page, upstream, question: str) -> list[dict]:
    left_the_chat = threading.Event()
    upstream.queue(reply.text(ANSWER, hold=left_the_chat, match=reply.answering(question)))
    load(page, "/")
    send(page, question)
    expect(page).to_have_url(re.compile(r"/c/"))
    expect(conversation(page)).to_contain_text(question)
    page.get_by_role("link", name="New Chat").click()
    expect(conversation(page)).to_have_count(0)
    left_the_chat.set()
    expect(page.get_by_text(ANSWER)).to_be_visible()
    return page.evaluate("window.shownNotifications")


def test_a_reply_finished_in_another_chat_shows_a_browser_notification_that_opens_it(
    page_for, make_user, upstream
):
    page = recording_page(page_for, make_user(), allow=True)
    switch_browser_notifications_on(page)

    shown = finish_a_reply_elsewhere(page, upstream, "When does the tide turn?")
    assert [notification["body"] for notification in shown] == [ANSWER]

    # since 13e0fd2c5 clicking the notification opens the chat whose reply finished
    click_the_browser_notification(page)
    expect(page).to_have_url(re.compile(r"/c/"))
    expect(conversation(page)).to_contain_text("When does the tide turn?")

    expect(notifications_switch(page)).to_have_attribute("aria-checked", "true")

    other = recording_page(page_for, make_user(), allow=True)
    expect(notifications_switch(other)).to_have_attribute("aria-checked", "false")
    assert finish_a_reply_elsewhere(other, upstream, "When does the tide turn today?") == []


# a headless page is never hidden, so it is told what a page in a window in the background is told
SHOW_THE_WINDOW_AS = """state => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => state });
    document.dispatchEvent(new Event('visibilitychange'));
}"""


@pytest.mark.chat_surfaces(MAIN_CHAT, SIDE_CHAT, ASKED_ASIDE, DRAGGED_TAB, TEMPORARY_SIDE)
def test_a_reply_that_finishes_while_the_window_is_in_the_background_shows_a_notification(
    chat_for, make_user, upstream
):
    """In a side chat: open-webui/open-webui#34025, no notification, toast or sound at all."""
    account = make_user()
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"notificationEnabled": True}}
        )
    assert saved.status_code == 200, saved.text
    chat = chat_for(
        account,
        init_script="const ALLOWED = true;" + RECORD_NOTIFICATIONS,
        permissions=["notifications"],
    )
    page = chat.page
    in_view = f"When does the ferry leave? ({uuid.uuid4().hex[:6]})"
    upstream.queue(reply.text("At nine.", match=reply.answering(in_view)))
    chat.send(in_view)
    chat.expect_reply("At nine.")
    expect(chat.stop_button).to_be_hidden(timeout=REPLY_TIMEOUT_MS)
    assert page.evaluate("window.shownNotifications") == [], "a reply read in view was notified"

    window_hidden = threading.Event()
    away = f"When does the tide turn? ({uuid.uuid4().hex[:6]})"
    upstream.queue(reply.text(ANSWER, hold=window_hidden, match=reply.answering(away)))
    chat.send(away)
    expect(chat.stop_button).to_be_visible()
    page.evaluate(SHOW_THE_WINDOW_AS, "hidden")
    window_hidden.set()

    page.wait_for_function("window.shownNotifications.length > 0", timeout=REPLY_TIMEOUT_MS)
    shown = page.evaluate("window.shownNotifications")
    assert [notification["body"] for notification in shown] == [ANSWER], (
        f"the {chat.place} notified {shown}"
    )
    expect(page.get_by_role("status").filter(has_text=ANSWER).last).to_be_visible()


# keeps the pitch of every tone the page plays; since b6a77a0f1 the notification sound is a few
# tones made in the page, one oscillator each, in place of an audio file
RECORD_SOUNDS = """
window.playedTones = [];
const createOscillator = AudioContext.prototype.createOscillator;
AudioContext.prototype.createOscillator = function () {
    const oscillator = createOscillator.call(this);
    const setValueAtTime = oscillator.frequency.setValueAtTime.bind(oscillator.frequency);
    let pitched = false;
    oscillator.frequency.setValueAtTime = (value, time) => {
        if (!pitched) window.playedTones.push(value);
        pitched = true;
        return setValueAtTime(value, time);
    };
    return oscillator;
};
"""
# the tones of the Chime and Bell styles, in Hz
CHIME = [659.25, 880]
BELL = [1046.5, 2093, 2900]


def test_a_denied_permission_keeps_browser_notifications_off(page_for, make_user):
    account = make_user()
    page = recording_page(page_for, account, allow=False)
    switch = notifications_switch(page)

    switch.click()

    expect(page.get_by_text(DENIED)).to_be_visible()
    with account.client() as client:
        stored = client.get("/api/v1/users/user/settings").json()["ui"]
    assert stored.get("notificationEnabled") is not True
    flipped_on = "the switch flipped on though the setting stayed off (open-webui/open-webui#32024)"
    expect(switch, flipped_on).to_have_attribute("aria-checked", "false")


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


def tones_played(page: Page) -> list[float]:
    return page.evaluate("window.playedTones")


def sound_page(page_for, account) -> Page:
    page = recording_page(page_for, account, allow=False)
    page.add_init_script(RECORD_SOUNDS)
    load(page, "/")
    return page


def test_the_toast_plays_the_notification_sound_until_it_is_switched_off(
    page_for, make_user, upstream
):
    page = sound_page(page_for, make_user())

    finish_a_reply_elsewhere(page, upstream, "When does the tide turn at dawn?")
    expect(page.get_by_text(ANSWER)).to_be_visible()
    page.wait_for_function("window.playedTones.length > 0")
    assert tones_played(page) == CHIME, "the finished reply did not play the default Chime"

    load(page, "/?settings=notifications")
    sound = page.get_by_role("switch", name="Notification Sound")
    expect(sound).to_have_attribute("aria-checked", "true")
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        sound.click()
    expect(sound).to_have_attribute("aria-checked", "false")
    expect(page.get_by_label("Notification sound style")).to_have_count(0)
    finish_a_reply_elsewhere(page, upstream, "When does the tide turn at dusk?")
    # the toast starts its sound as it shows, so a second is ample for one to begin
    page.wait_for_timeout(1000)
    played = tones_played(page)
    assert played == [], f"the toast played tones {played} with Notification Sound off"


def test_the_chosen_sound_style_is_previewed_kept_and_played(page_for, make_user, upstream):
    account = make_user()
    page = sound_page(page_for, account)
    load(page, "/?settings=notifications")
    style = page.get_by_label("Notification sound style")
    expect(style).to_have_value("chime")

    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        style.select_option("bell")
    page.get_by_role("button", name="Preview notification sound").click()
    page.wait_for_function("window.playedTones.length > 0")
    assert tones_played(page) == BELL, "the preview did not play the Bell style picked"
    with account.client() as client:
        stored = client.get("/api/v1/users/user/settings").json()["ui"]
    assert stored.get("notificationSoundType") == "bell", stored

    load(page, "/?settings=notifications")
    expect(page.get_by_label("Notification sound style")).to_have_value("bell")
    finish_a_reply_elsewhere(page, upstream, "When does the tide turn by the bell?")
    page.wait_for_function("window.playedTones.length > 0")
    assert tones_played(page) == BELL, "the finished reply did not play the Bell style picked"


def test_a_channel_message_shows_a_browser_notification_that_opens_the_channel(
    page_for, make_user, admin, preserve
):
    preserve("admin_config")
    enable_channels(admin)
    reader, writer = make_user(), make_user()
    channel_id = group_channel(writer, reader)
    page = recording_page(page_for, reader, allow=True)
    switch_browser_notifications_on(page)
    expect(chat_input(page)).to_be_visible()

    message = f"The ferry leaves at four {uuid.uuid4().hex[:6]}"
    post_message(writer, channel_id, message)
    page.wait_for_function("window.shownNotifications.length > 0")
    shown = page.evaluate("window.shownNotifications")
    with reader.client() as client:
        name = client.get(f"/api/v1/channels/{channel_id}").json()["name"]
    assert shown == [{"title": f"{writer.name} (#{name}) / Open WebUI", "body": message}]

    click_the_browser_notification(page)
    expect(page).to_have_url(re.compile(rf"/channels/{channel_id}$"))
    expect(page.get_by_role("main").get_by_text(message)).to_be_visible()


def test_a_calendar_reminder_shows_a_browser_notification_that_opens_the_calendar(
    page_for, make_user
):
    account = make_user()
    page = recording_page(page_for, account, allow=True)
    switch_browser_notifications_on(page)
    expect(chat_input(page)).to_be_visible()

    title = f"Harbour pilot {uuid.uuid4().hex[:6]}"
    with account.client() as client:
        start = time.time_ns() + 5 * MINUTE_NS
        created = create_event(
            client, default_calendar_id(client), title=title, start_at=start, alert_minutes=10
        )
        assert created.status_code == 200, created.text
        event_id = created.json()["id"]

    def alerted() -> bool:
        with account.client() as client:
            stored = client.get(f"/api/v1/calendars/events/{event_id}").json()
        return bool((stored.get("meta") or {}).get("alerted_at"))

    # the scheduler polls every ten seconds or so
    assert wait_until(alerted, 30), "the reminder was never sent"
    page.wait_for_function("window.shownNotifications.length > 0")
    shown = page.evaluate("window.shownNotifications")
    assert [notification["title"] for notification in shown] == [f"{title} / Open WebUI"]
    assert re.fullmatch(r"Starting in [3-5] minutes", shown[0]["body"]), shown

    click_the_browser_notification(page)
    expect(page).to_have_url(re.compile(r"/calendar$"))


def test_the_browser_notification_of_a_thread_reply_opens_the_thread_as_the_toast_does(
    page_for, make_user, admin, preserve
):
    """Red: the browser notification of a thread reply opens the channel, not the thread
    (open-webui/open-webui#34232); passes on the fix branch
    fix/thread-reply-notification-opens-channel and fails on dev 2547992b8.
    """
    preserve("admin_config")
    enable_channels(admin)
    reader, writer = make_user(), make_user()
    channel_id = group_channel(writer, reader)
    page = recording_page(page_for, reader, allow=True)
    switch_browser_notifications_on(page)
    expect(chat_input(page)).to_be_visible()

    parent_id = post_message(
        writer, channel_id, f"Who takes the night ferry? {uuid.uuid4().hex[:6]}"
    )
    page.wait_for_function("window.shownNotifications.length === 1")
    post_message(writer, channel_id, "I will, at eleven.", parent_id=parent_id)
    page.wait_for_function("window.shownNotifications.length === 2")

    click_the_browser_notification(page)
    expect(page).to_have_url(re.compile(rf"/channels/{channel_id}\?thread={parent_id}$"))
