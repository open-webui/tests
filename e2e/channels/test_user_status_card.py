"""Journey: a person's profile card in a channel shows their local time and last seen.

The card opens from an author's picture or from an @mention of the person in a message. It shows
the person's status, a "Local time" row with the time now in the timezone stored on their account
(the zone name beside it, the timezone as the row's title) and a "Last seen" row with when they
were last active as a medium date and short time in the viewer's browser zone. A card opened a
second time shows the status the person has by then and not the one it showed before.

Discriminates: passes on dev 538f9c909; in a frontend copy, dropping the local time row turns the
local time test red, dropping the last seen row turns the last seen test red, the mention not
handing its open state to the card turns the mention test red and removing the reset of the
requested person when the card closes turns the reopened card test red.
"""

from __future__ import annotations

import datetime as dt
import re
import time
from zoneinfo import ZoneInfo

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.calendar_api import set_timezone
from harness.channel_quotes import enable_channels, group_channel, post_message

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

OLD_STATUS = "Sailing until noon"
NEW_STATUS = "Back on shore"
PERSON_ZONE = "Asia/Tokyo"
PERSON_ZONE_LABEL = "GMT+9"
VIEWER_ZONE = "Pacific/Auckland"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@pytest.fixture
def people(admin, preserve, make_user):
    """The person with a timezone and a status, the viewer, and a channel they share."""
    preserve("admin_config")
    enable_channels(admin)
    person, viewer = make_user(), make_user()
    with person.client() as client:
        set_timezone(client, PERSON_ZONE)
    _set_status(person, OLD_STATUS)
    return person, viewer, group_channel(person, viewer)


def _set_status(person, message: str) -> None:
    with person.client() as client:
        saved = client.post(
            "/api/v1/users/user/status/update",
            json={"status_emoji": "rocket", "status_message": message},
        )
    assert saved.status_code == 200, saved.text


def _last_active_at(viewer, person) -> int:
    deadline = time.time() + 10
    while time.time() < deadline:
        with viewer.client() as client:
            seen = client.get(f"/api/v1/users/{person.id}/info").json()["last_active_at"]
        if seen:
            return seen
        time.sleep(0.2)
    raise AssertionError("the person's last activity was never recorded")


def _author_picture(page: Page, channel_id: str, author, text: str) -> Locator:
    page.goto(f"/channels/{channel_id}")
    message = page.locator("[id^='message-']").filter(has_text=text).first
    # the author's picture has no alt text, so it has no role to find it by
    return message.locator(f"img[src$='/users/{author.id}/profile/image']")


def _clock_text(zone: str, moment: dt.datetime) -> str:
    local = moment.astimezone(ZoneInfo(zone))
    return f"{local.hour % 12 or 12}:{local.minute:02d}\\s[{'AP'[local.hour // 12]}]M"


def test_the_card_shows_the_local_time_in_the_persons_timezone(people, page_for):
    person, viewer, channel_id = people
    post_message(person, channel_id, "anchoring at the bay")
    page = page_for(viewer, timezone_id=VIEWER_ZONE)
    picture = _author_picture(page, channel_id, person, "anchoring at the bay")

    before = dt.datetime.now(dt.timezone.utc)
    picture.click()

    local_time = page.get_by_title(PERSON_ZONE)
    expect(local_time).to_be_visible()
    after = dt.datetime.now(dt.timezone.utc)
    # a minute may roll over between the two clock reads
    wanted = "|".join(sorted({_clock_text(PERSON_ZONE, before), _clock_text(PERSON_ZONE, after)}))
    expect(local_time).to_have_text(re.compile(rf"^({wanted})\s{re.escape(PERSON_ZONE_LABEL)}$"))
    expect(page.get_by_text("Local time", exact=True).first).to_be_attached()


def test_the_card_shows_when_the_person_was_last_seen_in_the_viewers_zone(people, page_for):
    person, viewer, channel_id = people
    post_message(person, channel_id, "anchoring at the bay")
    page = page_for(viewer, timezone_id=VIEWER_ZONE)
    picture = _author_picture(page, channel_id, person, "anchoring at the bay")

    picture.click()

    expect(page.get_by_text("Last seen", exact=True).first).to_be_attached()
    browser_zone = page.evaluate("Intl.DateTimeFormat().resolvedOptions().timeZone")
    seen = dt.datetime.fromtimestamp(_last_active_at(viewer, person), ZoneInfo(browser_zone))
    date = f"{MONTHS[seen.month - 1]} {seen.day}, {seen.year}"
    last_seen = page.get_by_role("definition").filter(has_text=date)
    expect(last_seen).to_have_text(re.compile(rf"^{date}, \d{{1,2}}:\d{{2}}\s[AP]M$"))


def test_hovering_a_mention_of_a_person_shows_their_card(people, page_for):
    person, viewer, channel_id = people
    post_message(viewer, channel_id, f"<@U:{person.id}|{person.name}> are you at the bay?")
    page = page_for(viewer)
    page.goto(f"/channels/{channel_id}")

    page.get_by_text(f"@{person.name}", exact=True).hover()

    expect(page.get_by_text(OLD_STATUS)).to_be_visible()
    expect(page.get_by_title(PERSON_ZONE)).to_be_visible()


def test_a_reopened_card_shows_the_status_the_person_has_by_then(people, page_for):
    person, viewer, channel_id = people
    post_message(person, channel_id, "anchoring at the bay")
    page = page_for(viewer)
    picture = _author_picture(page, channel_id, person, "anchoring at the bay")
    picture.hover()
    expect(page.get_by_text(OLD_STATUS)).to_be_visible()
    page.mouse.move(0, 0)
    expect(page.get_by_text(OLD_STATUS)).to_have_count(0)

    _set_status(person, NEW_STATUS)
    picture.hover()

    expect(page.get_by_text(NEW_STATUS)).to_be_visible()
    expect(page.get_by_text(OLD_STATUS)).to_have_count(0)
