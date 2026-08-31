"""Calendar and automation regressions fixed in 0.11.2.

* Moving an event to another date made it vanish from the calendar. `get_events_by_range`
  required `end_at > range_start` for any event carrying an end, and the move rewrote
  `start_at` only, so an event whose stale `end_at` still sat on the old date matched
  nothing. The range query now also accepts `start_at >= range_start`, which brings rows
  already stuck in that state back as well (`1c5128c4a`, #29085, issue #29067).
* `expand_recurring_event` passed the rule text to `rrulestr` untouched. A DTSTART line
  inside the rule wins over the `dtstart=` argument in dateutil, so occurrences were
  generated from the rule's own anchor and landed on the wrong weekday, day of month or
  hour. DTSTART lines are now stripped so the event's own start anchors the expansion
  (`a93c50803`, `8c0c7b3b6`).
* Same commits added a floor on calendar recurrence: an event may not repeat more often
  than once a day, and both event forms reject a sub-daily rule with
  `ERROR_MESSAGES.CALENDAR_RRULE_TOO_FREQUENT`.
* `rrule_interval_seconds` had the same DTSTART problem, so the interval it reported was
  the spacing of the rule's own anchor series rather than the rule's period
  (`fd679e1da`).

Every expansion here is bounded by a window of a few days plus `max_instances`, and every
rule fed to `rrule_interval_seconds` walks exactly two occurrences.

Discriminates: passes on v0.11.3, fails on v0.11.1 (a moved event drops out of the range
query, occurrences follow the rule's DTSTART instead of the event, a sub-daily rule is
accepted, and the reported interval is the DTSTART series' spacing).
"""

from __future__ import annotations

import datetime as dt
import time
from uuid import uuid4

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.regression

HOUR_NS = 60 * 60 * 1_000_000_000
DAY_NS = 24 * HOUR_NS

# A Tuesday at 09:30 UTC, far from any DST boundary.
EVENT_START = dt.datetime(2026, 3, 10, 9, 30, tzinfo=dt.timezone.utc)
EVENT_START_NS = int(EVENT_START.timestamp() * 1_000_000_000)

# The Thursday before, at 17:00: a different weekday, day of month and hour.
FOREIGN_DTSTART = 'DTSTART:20260305T170000'

SUB_DAILY_RULES = (
    'RRULE:FREQ=HOURLY',
    'RRULE:FREQ=MINUTELY;INTERVAL=30',
    'RRULE:FREQ=HOURLY;INTERVAL=12',
    'RRULE:FREQ=SECONDLY;INTERVAL=45',
)


@pytest.fixture(scope='module')
def calendar_utils(owui_module):
    return owui_module('open_webui.utils.calendar')


@pytest.fixture(scope='module')
def db(owui_module):
    """`open_webui.internal.db`, after config has run the migrations."""
    owui_module('open_webui.config')
    return owui_module('open_webui.internal.db')


@pytest.fixture(scope='module')
def calendar_model(db, owui_module):
    return owui_module('open_webui.models.calendar')


@pytest.fixture(scope='module')
def too_frequent_message(owui_module):
    """The 0.11.2 refusal text, or None on refs without the limit."""
    errors = owui_module('open_webui.constants').ERROR_MESSAGES
    return getattr(errors, 'CALENDAR_RRULE_TOO_FREQUENT', None)


@pytest.fixture
def ids():
    """Unique suffix so rows from different tests in the shared scratch db cannot collide."""
    return uuid4().hex[:12]


def recurring_event(rrule: str) -> dict:
    return {
        'id': 'event-1',
        'title': 'Standup',
        'start_at': EVENT_START_NS,
        'end_at': EVENT_START_NS + HOUR_NS,
        'rrule': rrule,
    }


def local_starts(instances: list[dict]) -> list[dt.datetime]:
    return [
        dt.datetime.fromtimestamp(instance['start_at'] / 1_000_000_000, dt.timezone.utc)
        for instance in instances
    ]


def expand(calendar_utils, rrule: str, days: int = 30, max_instances: int = 4) -> list[dict]:
    return calendar_utils.expand_recurring_event(
        recurring_event(rrule),
        EVENT_START_NS - DAY_NS,
        EVENT_START_NS + days * DAY_NS,
        tz='UTC',
        max_instances=max_instances,
    )


async def seed_calendar(calendar_model, owner: str) -> str:
    calendar = await calendar_model.Calendars.insert_new_calendar(
        owner, calendar_model.CalendarForm(name=f'cal-{owner}')
    )
    return calendar.id


async def seed_event(calendar_model, calendar_id: str, owner: str, start: int, end: int) -> str:
    event = await calendar_model.CalendarEvents.insert_new_event(
        owner,
        calendar_model.CalendarEventForm(
            calendar_id=calendar_id, title=f'event-{owner}', start_at=start, end_at=end
        ),
    )
    return event.id


async def event_ids_in_range(calendar_model, user_id: str, start: int, end: int) -> list[str]:
    events = await calendar_model.CalendarEvents.get_events_by_range(user_id, start, end)
    return [event.id for event in events]


####################
# Narrow: a moved event stays on the calendar (#29067)
####################


@pytest.mark.asyncio
async def test_moved_event_is_still_returned_for_its_new_day(calendar_model, ids):
    """Changing only `start_at` leaves a stale `end_at` behind, and the event must survive it.

    Pre-fix the range query demanded `end_at > range_start`, so the event disappeared from
    the day it had just been moved to.
    """
    owner = f'own-{ids}'
    calendar_id = await seed_calendar(calendar_model, owner)
    original_start = int(time.time_ns()) + DAY_NS
    event_id = await seed_event(
        calendar_model, calendar_id, owner, original_start, original_start + HOUR_NS
    )

    moved_start = original_start + 7 * DAY_NS
    await calendar_model.CalendarEvents.update_event_by_id(
        event_id, calendar_model.CalendarEventUpdateForm(start_at=moved_start)
    )

    visible = await event_ids_in_range(
        calendar_model, owner, moved_start - HOUR_NS, moved_start + DAY_NS
    )
    assert event_id in visible, 'the event vanished from the day it was moved to'


@pytest.mark.asyncio
async def test_event_already_stuck_with_a_stale_end_is_shown_again(calendar_model, ids):
    """A row moved before the fix still has its end on the old date, and must come back."""
    owner = f'own-{ids}'
    calendar_id = await seed_calendar(calendar_model, owner)
    start = int(time.time_ns()) + 30 * DAY_NS
    event_id = await seed_event(calendar_model, calendar_id, owner, start, start - 6 * DAY_NS)

    visible = await event_ids_in_range(calendar_model, owner, start - HOUR_NS, start + DAY_NS)
    assert event_id in visible, 'an event with a stale end date stays hidden'


####################
# Broad: any event starting inside the window is returned, whatever its end says
####################


@pytest.mark.asyncio
@pytest.mark.parametrize('end_offset_days', [-30, -7, -1, 0, 1])
async def test_start_inside_the_window_is_enough(calendar_model, ids, end_offset_days):
    owner = f'own-{ids}'
    calendar_id = await seed_calendar(calendar_model, owner)
    start = int(time.time_ns()) + 60 * DAY_NS
    event_id = await seed_event(
        calendar_model, calendar_id, owner, start, start + end_offset_days * DAY_NS
    )

    visible = await event_ids_in_range(calendar_model, owner, start - HOUR_NS, start + DAY_NS)
    assert event_id in visible


####################
# Nearby: range filtering that was already right
####################


@pytest.mark.asyncio
async def test_an_unmoved_event_is_still_returned(calendar_model, ids):
    owner = f'own-{ids}'
    calendar_id = await seed_calendar(calendar_model, owner)
    start = int(time.time_ns()) + 90 * DAY_NS
    event_id = await seed_event(calendar_model, calendar_id, owner, start, start + HOUR_NS)

    visible = await event_ids_in_range(calendar_model, owner, start - HOUR_NS, start + DAY_NS)
    assert event_id in visible


@pytest.mark.asyncio
async def test_an_event_outside_the_window_is_still_excluded(calendar_model, ids):
    """The looser condition must not drag in events the window does not cover."""
    owner = f'own-{ids}'
    calendar_id = await seed_calendar(calendar_model, owner)
    start = int(time.time_ns()) + 120 * DAY_NS
    event_id = await seed_event(calendar_model, calendar_id, owner, start, start + HOUR_NS)

    before = await event_ids_in_range(
        calendar_model, owner, start + 10 * DAY_NS, start + 20 * DAY_NS
    )
    after = await event_ids_in_range(
        calendar_model, owner, start - 20 * DAY_NS, start - 10 * DAY_NS
    )
    assert event_id not in before
    assert event_id not in after


@pytest.mark.asyncio
async def test_a_long_running_event_overlapping_the_window_is_returned(calendar_model, ids):
    """An event that started before the window but has not ended is still on the calendar."""
    owner = f'own-{ids}'
    calendar_id = await seed_calendar(calendar_model, owner)
    start = int(time.time_ns()) + 150 * DAY_NS
    event_id = await seed_event(calendar_model, calendar_id, owner, start, start + 5 * DAY_NS)

    visible = await event_ids_in_range(
        calendar_model, owner, start + 2 * DAY_NS, start + 3 * DAY_NS
    )
    assert event_id in visible


####################
# Narrow: occurrences come from the event, not from the rule's own DTSTART
####################


def test_expansion_ignores_a_dtstart_on_another_weekday(calendar_utils):
    """dateutil lets a DTSTART inside the rule override the `dtstart=` argument.

    Pre-fix the weekly series ran on the rule's Thursday 17:00 instead of the event's
    Tuesday 09:30.
    """
    instances = expand(calendar_utils, f'{FOREIGN_DTSTART}\nRRULE:FREQ=WEEKLY')

    assert instances, 'no occurrences returned'
    starts = local_starts(instances)
    assert starts[0] == EVENT_START, f'first occurrence is {starts[0]}, not the event start'
    assert all(start.weekday() == EVENT_START.weekday() for start in starts), (
        f'occurrences landed on {[start.strftime("%a") for start in starts]}'
    )
    assert all((start.hour, start.minute) == (9, 30) for start in starts)


def test_expansion_ignores_a_dtstart_on_another_day_of_month(calendar_utils):
    instances = expand(calendar_utils, f'{FOREIGN_DTSTART}\nRRULE:FREQ=MONTHLY', days=70)

    starts = local_starts(instances)
    assert starts[0] == EVENT_START
    assert all(start.day == EVENT_START.day for start in starts), (
        f'occurrences landed on days {[start.day for start in starts]}'
    )


####################
# Broad: every frequency is driven by the event's own start
####################


@pytest.mark.parametrize(
    ('freq', 'days', 'step'),
    [('DAILY', 6, dt.timedelta(days=1)), ('WEEKLY', 30, dt.timedelta(weeks=1))],
)
def test_occurrences_step_from_the_event_start(calendar_utils, freq, days, step):
    instances = expand(calendar_utils, f'{FOREIGN_DTSTART}\nRRULE:FREQ={freq}', days=days)

    starts = local_starts(instances)
    assert starts[0] == EVENT_START
    assert all(later - earlier == step for earlier, later in zip(starts, starts[1:]))


def test_monthly_occurrences_step_from_the_event_start(calendar_utils):
    instances = expand(calendar_utils, f'{FOREIGN_DTSTART}\nRRULE:FREQ=MONTHLY', days=70)

    starts = local_starts(instances)
    assert starts[:3] == [
        EVENT_START,
        EVENT_START.replace(month=4),
        EVENT_START.replace(month=5),
    ]


####################
# Nearby: expansion of rules that never carried a DTSTART
####################


def test_a_rule_without_a_dtstart_is_unchanged(calendar_utils):
    instances = expand(calendar_utils, 'RRULE:FREQ=WEEKLY')

    starts = local_starts(instances)
    assert starts[0] == EVENT_START
    assert all(
        later - earlier == dt.timedelta(weeks=1) for earlier, later in zip(starts, starts[1:])
    )


def test_instance_duration_survives_the_anchor_change(calendar_utils):
    instances = expand(calendar_utils, f'{FOREIGN_DTSTART}\nRRULE:FREQ=WEEKLY')

    assert all(
        instance['end_at'] - instance['start_at'] == HOUR_NS for instance in instances
    )


####################
# Narrow: a calendar event may not repeat more often than daily
####################


def test_an_hourly_event_is_refused(calendar_model, too_frequent_message):
    with pytest.raises(ValidationError) as excinfo:
        calendar_model.CalendarEventForm(
            calendar_id='cal-1', title='Standup', start_at=EVENT_START_NS, rrule='RRULE:FREQ=HOURLY'
        )

    assert too_frequent_message in str(excinfo.value)


def test_an_hourly_update_is_refused(calendar_model, too_frequent_message):
    with pytest.raises(ValidationError) as excinfo:
        calendar_model.CalendarEventUpdateForm(rrule='RRULE:FREQ=HOURLY')

    assert too_frequent_message in str(excinfo.value)


####################
# Broad: every sub-daily frequency is refused by both forms
####################


@pytest.mark.parametrize('rule', SUB_DAILY_RULES)
def test_every_sub_daily_rule_is_refused_on_create(calendar_model, rule, too_frequent_message):
    with pytest.raises(ValidationError) as excinfo:
        calendar_model.CalendarEventForm(
            calendar_id='cal-1', title='Standup', start_at=EVENT_START_NS, rrule=rule
        )

    assert too_frequent_message in str(excinfo.value)


@pytest.mark.parametrize('rule', SUB_DAILY_RULES)
def test_every_sub_daily_rule_is_refused_on_update(calendar_model, rule, too_frequent_message):
    with pytest.raises(ValidationError) as excinfo:
        calendar_model.CalendarEventUpdateForm(rrule=rule)

    assert too_frequent_message in str(excinfo.value)


####################
# Nearby: daily and coarser rules, and no rule at all, still pass validation
####################


@pytest.mark.parametrize(
    'rule',
    [
        'RRULE:FREQ=DAILY',
        'RRULE:FREQ=DAILY;INTERVAL=2',
        'RRULE:FREQ=WEEKLY;BYDAY=TU',
        'RRULE:FREQ=MONTHLY',
        None,
    ],
)
def test_daily_and_coarser_rules_are_accepted(calendar_model, rule):
    """Exactly daily sits on the boundary and is allowed."""
    form = calendar_model.CalendarEventForm(
        calendar_id='cal-1', title='Standup', start_at=EVENT_START_NS, rrule=rule
    )
    assert form.rrule == rule


####################
# Narrow: the reported interval ignores a DTSTART line (fd679e1da)
####################


def test_interval_ignores_a_leap_day_dtstart(automations_module):
    """A 29 February anchor makes a yearly rule step four years at a time.

    The interval is the rule's period, so it must be a single year and must match the same
    rule without the anchor.
    """
    with_anchor = automations_module.rrule_interval_seconds(
        'DTSTART:20240229T090000\nRRULE:FREQ=YEARLY'
    )
    without_anchor = automations_module.rrule_interval_seconds('RRULE:FREQ=YEARLY')

    assert with_anchor in (365 * 86400, 366 * 86400), (
        f'a yearly rule reported {with_anchor / 86400:.0f} days'
    )
    assert with_anchor == without_anchor, (
        f'the DTSTART line changed the reported interval: {with_anchor} vs {without_anchor}'
    )


####################
# Broad: no DTSTART line changes the reported interval
####################


# The leap-day anchors discriminate on any date; the 31 January one only when today is not the 31st.
@pytest.mark.parametrize(
    ('anchor', 'rule'),
    [
        ('DTSTART:20240229T090000', 'RRULE:FREQ=YEARLY'),
        ('DTSTART:20240229T090000', 'RRULE:FREQ=YEARLY;INTERVAL=2'),
        ('DTSTART:20260131T090000', 'RRULE:FREQ=MONTHLY'),
        ('DTSTART:20260305T170000', 'RRULE:FREQ=WEEKLY'),
        ('DTSTART:20260305T170000', 'RRULE:FREQ=DAILY;INTERVAL=3'),
    ],
)
def test_an_anchor_never_changes_the_interval(automations_module, anchor, rule):
    assert automations_module.rrule_interval_seconds(f'{anchor}\n{rule}') == (
        automations_module.rrule_interval_seconds(rule)
    )


####################
# Nearby: intervals that were already reported correctly
####################


def test_an_anchored_multi_day_rule_reports_its_own_period(automations_module):
    anchored = f'{FOREIGN_DTSTART}\nRRULE:FREQ=DAILY;INTERVAL=3'
    assert automations_module.rrule_interval_seconds(anchored) == 3 * 86400


def test_a_one_shot_rule_still_reports_no_interval(automations_module):
    one_shot = f'{FOREIGN_DTSTART}\nRRULE:COUNT=1;FREQ=DAILY'
    assert automations_module.rrule_interval_seconds(one_shot) is None
