"""Scheduling and calendar regressions fixed in 0.11.0.

* `CalendarEventAttendeeTable.set_attendees` took each attendee's RSVP status straight
  from the caller-supplied dict and rewrote the whole attendee set, so whoever created or
  edited an event decided whether YOU had accepted the invitation, and any edit reset an
  RSVP already given. It now reads the existing rows first and reuses their status, so a
  caller-supplied status is ignored and RSVP is only ever set through `update_rsvp`
  (`9b635d8f3`, #27007).
* The attendee lookup in `get_events_by_range` selected every event the user was an
  attendee of, so an invitation the user had declined stayed on their calendar. It now
  excludes `status == 'declined'` (same commit).
* `_parse_rule` scraped FREQ/INTERVAL out of the raw rule text with a case-sensitive
  mini-parser: lower-case keys were never recognised, `SECONDLY` was not handled, an
  embedded `DTSTART` was silently replaced by a fixed year-2000 epoch, and unsupportable
  rules (`EXRULE`, two `RRULE:` lines, `INTERVAL=0`) were accepted and behaved
  unpredictably. It now upper-cases the keys, handles `SECONDLY`, refuses the
  unsupportable rules by name, and keeps an embedded `DTSTART` only while the occurrence
  budget between it and now stays at or below 100k (`c4ae8c8`, `2d928df`, #27470).
* `validate_rrule` parsed the rule BEFORE resolving the user's timezone, so a sub-daily
  rule was anchored on the fixed year-2000 epoch while the emptiness check used the user's
  local clock. All four rrule entry points now compute the timezone-resolved `now` first
  and pass it into `_parse_rule` (`b3aead2`, issue #26954).

Every rule here is bounded by construction. The sub-daily cases either assert on the
`_dtstart` of the returned rule without ever walking it, or carry a `COUNT` that caps the
walk at a handful of occurrences. Nothing lets the pre-fix year-2000 anchor enumerate its
way to the present.

v0.11.0 upper-cases the rule's KEYS only, so a lower-case FREQ VALUE still slips past the
alignment branch. That successor defect belongs to 0.11.1 and is covered by
unit/security/test_recurrence_rule_parsing.py, so nothing here asserts on it.

v0.11.1 changed `validate_rrule` on purpose: a COUNT rule carrying no DTSTART is now refused
up front (`ERROR_MESSAGES.AUTOMATION_COUNT_REQUIRES_DTSTART`), because the synthesised anchor
makes the counted window meaningless. The clock test below therefore pins the thing the bug
was about, that the rule is never reported as having no future runs, and accepts the 0.11.1
refusal by name.

Discriminates: passes on v0.11.0 and v0.11.1, fails on v0.10.2 (an organiser sets and resets other
people's RSVP, a declined invitation still shows up, lower-case-key and SECONDLY rules skip
clock alignment while a far-past DTSTART survives, EXRULE / duplicate RRULE / INTERVAL=0
are accepted, and a COUNT-bounded sub-daily rule is rejected as having no future runs).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

pytestmark = pytest.mark.regression

HOUR_NS = 60 * 60 * 1_000_000_000
DAY_NS = 24 * HOUR_NS

# The scheduler snaps sub-daily rules to this epoch, so it is also the worst possible
# hand-written DTSTART: ~26 years of one-minute occurrences to walk.
FAR_PAST_DTSTART = 'DTSTART:20000101T000000'

# Frozen server clock. Naive = server wall clock (UTC here), aware = the same instant.
NOW_NAIVE = datetime(2026, 5, 5, 13, 7, 23)
NOW_UTC = NOW_NAIVE.replace(tzinfo=timezone.utc)

# A zone whose local clock is hours BEHIND the frozen server clock.
BEHIND_ZONE = 'America/Los_Angeles'

# COUNT caps every walk; 8 quarter-hours is well under an hour of occurrences.
QUARTER_HOURLY = 'RRULE:FREQ=MINUTELY;INTERVAL=15;COUNT=8'


class FrozenClock(datetime):
    """`datetime` with a fixed `now()`, so alignment is asserted against a known instant."""

    @classmethod
    def now(cls, tz=None):
        return NOW_UTC.astimezone(tz) if tz else NOW_NAIVE


def floor_to(moment: datetime, step: timedelta) -> datetime:
    epoch = datetime(2000, 1, 1)
    return epoch + ((moment - epoch) // step) * step


@pytest.fixture
def frozen_now(automations_module, monkeypatch):
    monkeypatch.setattr(automations_module, 'datetime', FrozenClock)
    return NOW_NAIVE


@pytest.fixture(scope='module')
def db(owui_module):
    """`open_webui.internal.db`, after config has run the migrations."""
    owui_module('open_webui.config')
    return owui_module('open_webui.internal.db')


@pytest.fixture(scope='module')
def calendar_model(db, owui_module):
    return owui_module('open_webui.models.calendar')


@pytest.fixture(scope='module')
def count_needs_dtstart(owui_module):
    """The 0.11.1 refusal message, or None on refs that still accept COUNT without DTSTART."""
    errors = owui_module('open_webui.constants').ERROR_MESSAGES
    return getattr(errors, 'AUTOMATION_COUNT_REQUIRES_DTSTART', None)


@pytest.fixture
def ids():
    """Unique suffix so rows from different tests in the shared scratch db cannot collide."""
    return uuid4().hex[:12]


async def seed_event(calendar_model, ids: str, organiser: str) -> str:
    """An event on the organiser's own calendar, starting an hour from now."""
    calendar = await calendar_model.Calendars.insert_new_calendar(
        organiser, calendar_model.CalendarForm(name=f'cal-{ids}')
    )
    start = int(time.time_ns()) + HOUR_NS
    event = await calendar_model.CalendarEvents.insert_new_event(
        organiser,
        calendar_model.CalendarEventForm(
            calendar_id=calendar.id,
            title=f'event-{ids}',
            start_at=start,
            end_at=start + HOUR_NS,
        ),
    )
    return event.id


async def status_of(calendar_model, event_id: str, user_id: str) -> str | None:
    rows = await calendar_model.CalendarEventAttendees.get_attendees_by_event(event_id)
    return next((row.status for row in rows if row.user_id == user_id), None)


async def visible_event_ids(calendar_model, user_id: str) -> list[str]:
    now = int(time.time_ns())
    events = await calendar_model.CalendarEvents.get_events_by_range(
        user_id, now - DAY_NS, now + DAY_NS
    )
    return [event.id for event in events]


####################
# Narrow: RSVP is the attendee's alone (#27007)
####################


@pytest.mark.asyncio
async def test_organiser_cannot_accept_on_an_attendees_behalf(calendar_model, ids):
    """A caller-supplied 'accepted' must not answer an invitation the attendee has not."""
    organiser, invitee = f'org-{ids}', f'inv-{ids}'
    event_id = await seed_event(calendar_model, ids, organiser)

    await calendar_model.CalendarEventAttendees.set_attendees(event_id, [{'user_id': invitee}])
    await calendar_model.CalendarEventAttendees.set_attendees(
        event_id, [{'user_id': invitee, 'status': 'accepted'}]
    )

    assert await status_of(calendar_model, event_id, invitee) == 'pending', (
        'the organiser RSVPed for the invitee'
    )


@pytest.mark.asyncio
async def test_editing_the_event_keeps_an_rsvp_already_given(calendar_model, ids):
    """Rewriting the attendee list must not throw away the answers people gave."""
    organiser, invitee = f'org-{ids}', f'inv-{ids}'
    event_id = await seed_event(calendar_model, ids, organiser)

    await calendar_model.CalendarEventAttendees.set_attendees(event_id, [{'user_id': invitee}])
    await calendar_model.CalendarEventAttendees.update_rsvp(event_id, invitee, 'accepted')
    await calendar_model.CalendarEventAttendees.set_attendees(
        event_id, [{'user_id': invitee}, {'user_id': f'other-{ids}'}]
    )

    assert await status_of(calendar_model, event_id, invitee) == 'accepted', (
        'the attendee list rewrite reset an RSVP that was already given'
    )


@pytest.mark.asyncio
async def test_declined_invitation_leaves_the_calendar(calendar_model, ids):
    """A declined invite is the only thing linking this user to the event, so it must go."""
    organiser, invitee = f'org-{ids}', f'inv-{ids}'
    event_id = await seed_event(calendar_model, ids, organiser)

    await calendar_model.CalendarEventAttendees.set_attendees(event_id, [{'user_id': invitee}])
    await calendar_model.CalendarEventAttendees.update_rsvp(event_id, invitee, 'declined')

    assert event_id not in await visible_event_ids(calendar_model, invitee), (
        'a declined invitation is still listed on the attendee calendar'
    )


####################
# Broad: no caller-supplied status ever reaches the row
####################


@pytest.mark.asyncio
@pytest.mark.parametrize('supplied', ['accepted', 'declined', 'tentative', 'pending'])
async def test_supplied_status_is_ignored_for_every_value(calendar_model, ids, supplied):
    organiser, invitee = f'org-{ids}', f'inv-{ids}'
    event_id = await seed_event(calendar_model, ids, organiser)

    seeded = 'accepted' if supplied == 'tentative' else 'tentative'

    await calendar_model.CalendarEventAttendees.set_attendees(event_id, [{'user_id': invitee}])
    await calendar_model.CalendarEventAttendees.update_rsvp(event_id, invitee, seeded)
    await calendar_model.CalendarEventAttendees.set_attendees(
        event_id, [{'user_id': invitee, 'status': supplied}]
    )

    assert await status_of(calendar_model, event_id, invitee) == seeded


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['pending', 'accepted', 'tentative'])
async def test_every_status_but_declined_stays_on_the_calendar(calendar_model, ids, status):
    organiser, invitee = f'org-{ids}', f'inv-{ids}'
    event_id = await seed_event(calendar_model, ids, organiser)

    await calendar_model.CalendarEventAttendees.set_attendees(event_id, [{'user_id': invitee}])
    await calendar_model.CalendarEventAttendees.update_rsvp(event_id, invitee, status)

    assert event_id in await visible_event_ids(calendar_model, invitee)


####################
# Nearby: attendee handling that was already right
####################


@pytest.mark.asyncio
async def test_a_newly_added_attendee_starts_pending(calendar_model, ids):
    organiser, invitee = f'org-{ids}', f'inv-{ids}'
    event_id = await seed_event(calendar_model, ids, organiser)

    await calendar_model.CalendarEventAttendees.set_attendees(event_id, [{'user_id': invitee}])

    assert await status_of(calendar_model, event_id, invitee) == 'pending'


@pytest.mark.asyncio
async def test_attendee_meta_still_comes_from_the_caller(calendar_model, ids):
    """Only `status` is protected; `meta` is still the organiser's to write."""
    organiser, invitee = f'org-{ids}', f'inv-{ids}'
    event_id = await seed_event(calendar_model, ids, organiser)

    await calendar_model.CalendarEventAttendees.set_attendees(
        event_id, [{'user_id': invitee, 'meta': {'role': 'chair'}}]
    )

    rows = await calendar_model.CalendarEventAttendees.get_attendees_by_event(event_id)
    assert [row.meta for row in rows] == [{'role': 'chair'}]


@pytest.mark.asyncio
async def test_the_organiser_still_sees_their_own_declined_event(calendar_model, ids):
    """Declining reaches the attendee's view, never the owner's."""
    organiser = f'org-{ids}'
    event_id = await seed_event(calendar_model, ids, organiser)

    await calendar_model.CalendarEventAttendees.set_attendees(event_id, [{'user_id': organiser}])
    await calendar_model.CalendarEventAttendees.update_rsvp(event_id, organiser, 'declined')

    assert event_id in await visible_event_ids(calendar_model, organiser)


####################
# Narrow: hand-written recurrence rules (#27470)
####################


def test_lower_case_keys_are_clock_aligned(automations_module, frozen_now):
    """dateutil is case-insensitive about the keys; the pre-fix scrape was not.

    A lower-case `freq=` scraped to no frequency at all, so alignment was skipped. Asserted
    on the returned rule's anchor, never by walking it.
    """
    rule = automations_module._parse_rule('RRULE:freq=MINUTELY;interval=5')

    assert rule._dtstart == floor_to(frozen_now, timedelta(minutes=5)), (
        f'lower-case-key rule anchored at {rule._dtstart}, not on the 5-minute boundary '
        'before now'
    )


def test_secondly_rule_is_clock_aligned(automations_module, frozen_now):
    """SECONDLY was not in the pre-fix frequency list at all."""
    rule = automations_module._parse_rule('RRULE:FREQ=SECONDLY;INTERVAL=10')

    assert rule._dtstart == floor_to(frozen_now, timedelta(seconds=10)), (
        f'SECONDLY rule anchored at {rule._dtstart}, not on the 10-second boundary before now'
    )


def test_far_past_start_is_dropped_when_the_budget_is_blown(automations_module, frozen_now):
    """A hand-written DTSTART worth millions of occurrences must be re-anchored to now.

    The anchor is checked here precisely so nothing downstream ever walks the rule.
    """
    rule = automations_module._parse_rule(f'{FAR_PAST_DTSTART}\nRRULE:FREQ=MINUTELY;INTERVAL=1')

    assert rule._dtstart == floor_to(frozen_now, timedelta(minutes=1)), (
        f'rule kept its year-2000 start ({rule._dtstart}); walking it would enumerate '
        'every minute since'
    )


def test_exrule_is_refused(automations_module):
    with pytest.raises(ValueError, match='EXRULE is not supported'):
        automations_module._parse_rule('RRULE:FREQ=DAILY;COUNT=3\nEXRULE:FREQ=DAILY;COUNT=1')


def test_a_second_rrule_line_is_refused(automations_module):
    with pytest.raises(ValueError, match='only one RRULE is supported'):
        automations_module._parse_rule('RRULE:FREQ=DAILY;COUNT=3\nRRULE:FREQ=HOURLY;COUNT=2')


def test_non_positive_interval_is_refused(automations_module):
    with pytest.raises(ValueError, match='INTERVAL must be a positive integer'):
        automations_module._parse_rule('RRULE:FREQ=MINUTELY;INTERVAL=0')


####################
# Broad: every sub-daily frequency, in either case, snaps to the clock
####################


@pytest.mark.parametrize(
    'text,step',
    [
        ('RRULE:FREQ=SECONDLY;INTERVAL=30', timedelta(seconds=30)),
        ('RRULE:freq=SECONDLY;interval=30', timedelta(seconds=30)),
        ('RRULE:FREQ=MINUTELY;INTERVAL=20', timedelta(minutes=20)),
        ('RRULE:freq=MINUTELY;interval=20', timedelta(minutes=20)),
        ('RRULE:FREQ=HOURLY;INTERVAL=6', timedelta(hours=6)),
        ('RRULE:freq=HOURLY;interval=6', timedelta(hours=6)),
    ],
)
def test_sub_daily_rules_snap_to_the_clock(automations_module, frozen_now, text, step):
    rule = automations_module._parse_rule(text)

    assert rule._dtstart == floor_to(frozen_now, step)


@pytest.mark.parametrize('freq', ['SECONDLY', 'MINUTELY', 'HOURLY'])
@pytest.mark.parametrize('interval', ['0', '-1'])
def test_every_sub_daily_frequency_refuses_a_non_positive_interval(
    automations_module, freq, interval
):
    with pytest.raises(ValueError, match='INTERVAL must be a positive integer'):
        automations_module._parse_rule(f'RRULE:FREQ={freq};INTERVAL={interval}')


####################
# Nearby: rules the fix deliberately leaves alone
####################


def test_a_daily_rule_is_returned_untouched(automations_module):
    """Daily and coarser rules never went through the alignment branch. COUNT caps the walk."""
    rule = automations_module._parse_rule('RRULE:FREQ=DAILY;COUNT=3')
    occurrences = list(rule)

    assert len(occurrences) == 3
    assert [b - a for a, b in zip(occurrences, occurrences[1:])] == [timedelta(days=1)] * 2


def test_a_recent_start_is_preserved(automations_module, frozen_now):
    """Three hours of hourly occurrences is well inside the budget, so the DTSTART stands."""
    start = frozen_now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=3)
    rule = automations_module._parse_rule(
        f'DTSTART:{start.strftime("%Y%m%dT%H%M%S")}\nRRULE:FREQ=HOURLY;INTERVAL=1'
    )

    assert rule._dtstart == start


####################
# Narrow: the user's clock decides, not the server's (issue #26954)
####################


def test_near_future_rule_validates_in_a_zone_behind_the_server(
    automations_module, frozen_now, count_needs_dtstart
):
    """Pre-fix the rule was anchored on the year-2000 epoch while `now` was local, so its
    eight quarter-hours were all exhausted and validation rejected a live schedule.

    COUNT=8 keeps the occurrence list at eight quarter-hours on every ref. The only refusal
    allowed here is the 0.11.1 COUNT-needs-DTSTART one, which is a different decision made
    before any clock is consulted.
    """
    try:
        automations_module.validate_rrule(QUARTER_HOURLY, tz=BEHIND_ZONE)
    except ValueError as error:
        assert str(error) == count_needs_dtstart, (
            f'a rule with eight future quarter-hours was rejected: {error}'
        )


def test_next_run_is_aligned_to_the_users_clock(automations_module, frozen_now):
    run_ns = automations_module.next_run_ns(QUARTER_HOURLY, tz=BEHIND_ZONE)

    assert run_ns is not None, 'a rule with eight future quarter-hours reported no next run'
    local = datetime.fromtimestamp(run_ns / 1_000_000_000, ZoneInfo(BEHIND_ZONE))
    assert (local.minute % 15, local.second) == (0, 0), f'{local} is not on a quarter-hour'
    assert timedelta(0) < local - NOW_UTC <= timedelta(minutes=15)


####################
# Broad: all four rrule entry points honour the resolved clock
####################


def test_next_n_runs_are_ascending_quarter_hours(automations_module, frozen_now):
    runs = automations_module.next_n_runs_ns(QUARTER_HOURLY, n=3, tz=BEHIND_ZONE)

    assert len(runs) == 3
    assert runs == sorted(runs)
    assert [b - a for a, b in zip(runs, runs[1:])] == [15 * 60 * 1_000_000_000] * 2


def test_interval_seconds_reads_a_count_bounded_sub_daily_rule(automations_module, frozen_now):
    assert automations_module.rrule_interval_seconds(QUARTER_HOURLY) == 900


####################
# Nearby: validation that behaved correctly before the fix
####################


def test_an_exhausted_rule_is_still_rejected(automations_module, frozen_now):
    """UNTIL in the past means no future runs whichever clock is consulted."""
    with pytest.raises(ValueError, match='no future occurrences'):
        automations_module.validate_rrule('RRULE:FREQ=DAILY;UNTIL=20200101T000000Z', tz=BEHIND_ZONE)


def test_a_daily_rule_reports_a_days_interval(automations_module, frozen_now):
    assert automations_module.rrule_interval_seconds('RRULE:FREQ=DAILY') == 86400
