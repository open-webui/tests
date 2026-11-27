"""Journey: a person's profile card in a channel shows their local time and last seen.

The card opens when pointing at an author's picture or at an @mention of the person in a message.
It shows the person's status, a "Local time" row with the time now in the timezone stored on their
account (the zone name beside it, the timezone as the row's title) and a "Last seen" row with how
long ago they were last active, whose tooltip gives the medium date and long time in the viewer's
browser zone. A card opened a second time shows the status the person has by then and not the one
it showed before. Clicking the picture opens the person's full profile in a dialog with their
name, status, a Message button and the local time in their timezone.

Discriminates: passes on dev 538f9c909; in a frontend copy, dropping the local time row turns the
local time test red, dropping the last seen row turns the last seen test red, the mention not
handing its open state to the card turns the mention test red and removing the reset of the
requested person when the card closes turns the reopened card test red. The full profile test
passes on dev 4dc058bbb and fails on 9c9a3ca0d, where a click opened no profile dialog.

Retargeted for dev b1da0e8e8: the last seen row shows a relative time and moved the date and time
into its tooltip (a70e96525). Retargeted for dev 4dc058bbb: a click on a picture opens the full
profile dialog in place of the card (4b648ec4e), so the card tests point at the picture.
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
    picture.hover()

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

    picture.hover()

    expect(page.get_by_text("Last seen", exact=True).first).to_be_attached()
    last_seen = page.get_by_role("definition").filter(has_text="ago")
    expect(last_seen).to_have_text("a few seconds ago")

    last_seen.get_by_text("a few seconds ago").hover()

    browser_zone = page.evaluate("Intl.DateTimeFormat().resolvedOptions().timeZone")
    seen = dt.datetime.fromtimestamp(_last_active_at(viewer, person), ZoneInfo(browser_zone))
    date = f"{MONTHS[seen.month - 1]} {seen.day}, {seen.year}"
    clock = f"{seen.hour % 12 or 12}:{seen.minute:02d}:{seen.second:02d}\\s{'AP'[seen.hour // 12]}M"
    # the viewer's zone is a whole number of hours off UTC, which the browser names GMT+N
    zone = f"GMT{seen.utcoffset() // dt.timedelta(hours=1):+d}"
    expect(page.get_by_role("tooltip")).to_have_text(
        re.compile(rf"^{date}, {clock} {re.escape(zone)}$")
    )


def test_clicking_a_persons_picture_opens_their_full_profile(people, page_for):
    person, viewer, channel_id = people
    post_message(person, channel_id, "anchoring at the bay")
    page = page_for(viewer, timezone_id=VIEWER_ZONE)
    picture = _author_picture(page, channel_id, person, "anchoring at the bay")

    before = dt.datetime.now(dt.timezone.utc)
    picture.click()

    profile = page.get_by_role("dialog", name="Profile")
    expect(profile.get_by_text(person.name, exact=True)).to_be_visible()
    expect(profile.get_by_text(OLD_STATUS)).to_be_visible()
    expect(profile.get_by_role("button", name="Message")).to_be_visible()
    local_time = profile.get_by_role("definition").filter(has_text=PERSON_ZONE_LABEL)
    expect(local_time).to_be_visible()
    after = dt.datetime.now(dt.timezone.utc)
    wanted = "|".join(sorted({_clock_text(PERSON_ZONE, before), _clock_text(PERSON_ZONE, after)}))
    expect(local_time).to_have_text(re.compile(rf"^({wanted})\s{re.escape(PERSON_ZONE_LABEL)}$"))


def test_hovering_a_mention_of_a_person_shows_their_card(people, page_for):
    person, viewer, channel_id = people
    post_message(viewer, channel_id, f"<@U:{person.id}|{person.name}> are you at the bay?")
    page = page_for(viewer, path=f"/channels/{channel_id}")

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
