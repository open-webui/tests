"""Regressions in the automation, calendar and timer plumbing fixed in 0.11.1.

* Recurring calendar events were expanded in the SERVER's timezone: `expand_recurring_event`
  built its scan window and its rrule anchor with naive `datetime.fromtimestamp`, so every
  instance came back shifted by the gap between the server zone and the user's
  (`d721b0d196`, now resolves the zone through `_resolve_tz` and converts in it).
* An RRULE carrying `COUNT=` but no `DTSTART` was accepted by `validate_rrule`. With no
  anchor the occurrence window never exhausted and the automation ran forever
  (`2ab0311b9`, #27781).
* A timer whose chat completion raised left the row marked `completed` while no reply ever
  arrived, and the exception escaped the scheduler (`f5a5a434b`, #27785, issue #27783).
* `get_upcoming_events` read `meta.alert_minutes` and compared it to a number. `meta` is
  user-writable and the poll is shared by every user, so one event holding a string there
  raised and stopped reminders for everyone (`abc69000b`, #28790).
* Searching a JSON column casts it to text, and whether non-ASCII was stored raw or as
  `\\uXXXX` escapes depends on the codec in force when the row was written, so a single
  LIKE pattern found roughly half the rows. `json_text_variants` ORs both spellings
  (`189c14fc4`, #28399).
* A forked chat copied a timer's `meta` verbatim and became a second claim target, so the
  timer fired twice. Timers now hang off a dedicated `chat.timer_at` column that a fork
  does not carry (`16c2a9eda`, #27663, issues #27622/#27745).

The third `json_text_variants` call site, the prompt tag search, is not covered: on SQLite
and PostgreSQL it takes a `json_each` / `json_array_elements_text` branch, and the LIKE
fallback that the fix touches is only reachable on a dialect this suite cannot run.

Every rrule expansion here is bounded by construction: the calendar rules carry `COUNT=3`
inside a three-day window with a small `max_instances`, and the `validate_rrule` cases only
ever walk one occurrence. Nothing in this file can run away on a checkout without the fixes.

Discriminates: passes on v0.11.1, fails on v0.11.0 (instances come back shifted off the
event's own start, a COUNT rule with no DTSTART validates, the timer exception escapes and
the row stays `completed`, a string `alert_minutes` raises out of the shared poll, the
single-spelling LIKE misses rows stored the other way, and the fork gets claimed too).
"""

from __future__ import annotations

import datetime as dt
import time
from types import SimpleNamespace
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio
from sqlalchemy import select, text

pytestmark = pytest.mark.regression

DAY_NS = 24 * 60 * 60 * 1_000_000_000
MINUTE_NS = 60 * 1_000_000_000

# Fixed-offset zones only: a DST fold would make "same wall clock" ambiguous.
FIXED_OFFSET_ZONES = ('Asia/Tokyo', 'Asia/Kolkata', 'UTC', 'Pacific/Kiritimati', 'Pacific/Honolulu')

# A Tuesday, far from any DST boundary in the zones above.
EVENT_START = dt.datetime(2026, 3, 10, 9, 30, tzinfo=dt.timezone.utc)
EVENT_START_NS = int(EVENT_START.timestamp() * 1_000_000_000)

# COUNT bounds the walk; the three-day window bounds it again.
DAILY_THRICE = 'RRULE:FREQ=DAILY;COUNT=3'

CJK = '天気'  # two chars a JSON encoder either writes raw or escapes to 天気


def foreign_zone() -> str:
    """A zone whose offset differs from this machine's, so the bug can show."""
    local_offset = EVENT_START.astimezone().utcoffset()
    for name in FIXED_OFFSET_ZONES:
        if EVENT_START.astimezone(ZoneInfo(name)).utcoffset() != local_offset:
            return name
    raise AssertionError('no candidate zone differs from the server zone')


def recurring_event(rrule: str = DAILY_THRICE) -> dict:
    return {
        'id': 'event-1',
        'title': 'Standup',
        'start_at': EVENT_START_NS,
        'end_at': EVENT_START_NS + 30 * MINUTE_NS,
        'rrule': rrule,
    }


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
def automations_model(db, owui_module):
    return owui_module('open_webui.models.automations')


@pytest.fixture(scope='module')
def models_model(db, owui_module):
    return owui_module('open_webui.models.models')


@pytest.fixture(scope='module')
def chats_model(db, owui_module):
    return owui_module('open_webui.models.chats')


@pytest.fixture(scope='module')
def users_model(db, owui_module):
    return owui_module('open_webui.models.users')


@pytest.fixture(scope='module')
def timers_module(db, owui_module):
    return owui_module('open_webui.utils.timers')


@pytest.fixture
def ids():
    """Unique prefix so rows from different tests in the shared scratch db cannot collide."""
    return uuid4().hex[:12]


def well_past_due() -> int:
    """A poll instant far enough ahead that every timer seeded here is due."""
    return int(time.time_ns()) + 10 * 60 * 1_000_000_000


async def execute_sql(db, statement: str, params: dict) -> None:
    async with db.get_async_db_context() as session:
        await session.execute(text(statement), params)
        await session.commit()


####################
# 33 — recurring events expanded in the server's timezone
####################


def test_recurring_instance_lands_on_the_events_own_start(calendar_utils):
    """The first expanded instance is the event itself, whatever zone it is read in.

    Pre-fix the anchor was the server-zone wall clock of `start_at`, then stamped back as
    the user's wall clock, so every instance moved by the offset between the two zones.
    """
    zone = foreign_zone()
    instances = calendar_utils.expand_recurring_event(
        recurring_event(),
        EVENT_START_NS - DAY_NS,
        EVENT_START_NS + 3 * DAY_NS,
        tz=zone,
        max_instances=5,
    )

    assert instances, f'no instances returned for {zone}'
    drift_hours = (instances[0]['start_at'] - EVENT_START_NS) / 3_600_000_000_000
    assert instances[0]['start_at'] == EVENT_START_NS, (
        f'first instance is {drift_hours:+.2f}h off the event start when read in {zone}'
    )


def test_instance_end_follows_the_shifted_start(calendar_utils):
    """Duration is preserved, so a drifting start drags the end along with it."""
    zone = foreign_zone()
    instances = calendar_utils.expand_recurring_event(
        recurring_event(),
        EVENT_START_NS - DAY_NS,
        EVENT_START_NS + 3 * DAY_NS,
        tz=zone,
        max_instances=5,
    )

    assert instances[0]['end_at'] == EVENT_START_NS + 30 * MINUTE_NS


@pytest.mark.parametrize('zone', FIXED_OFFSET_ZONES)
def test_expansion_is_anchored_in_the_requested_zone(calendar_utils, zone):
    """Broad: for every zone the run of instances starts at the event and steps a whole day."""
    instances = calendar_utils.expand_recurring_event(
        recurring_event(),
        EVENT_START_NS - DAY_NS,
        EVENT_START_NS + 3 * DAY_NS,
        tz=zone,
        max_instances=5,
    )

    starts = [instance['start_at'] for instance in instances]
    assert starts[0] == EVENT_START_NS, f'expansion in {zone} does not start at the event'
    assert all(later - earlier == DAY_NS for earlier, later in zip(starts, starts[1:]))


def test_expansion_without_a_zone_keeps_the_event_start(calendar_utils):
    """Nearby: with no zone the server clock is both sides of the conversion, so nothing moves."""
    instances = calendar_utils.expand_recurring_event(
        recurring_event(),
        EVENT_START_NS - DAY_NS,
        EVENT_START_NS + 3 * DAY_NS,
        tz=None,
        max_instances=5,
    )

    assert instances[0]['start_at'] == EVENT_START_NS


def test_non_recurring_event_is_returned_untouched(calendar_utils):
    event = {'id': 'event-2', 'start_at': EVENT_START_NS, 'end_at': None}
    assert calendar_utils.expand_recurring_event(
        event, EVENT_START_NS - DAY_NS, EVENT_START_NS + DAY_NS, tz='Asia/Tokyo'
    ) == [event]


def test_unparseable_rrule_falls_back_to_the_single_event(calendar_utils):
    event = recurring_event('RRULE:FREQ=NOPE')
    assert calendar_utils.expand_recurring_event(
        event, EVENT_START_NS - DAY_NS, EVENT_START_NS + DAY_NS, tz='Asia/Tokyo'
    ) == [event]


def test_max_instances_caps_the_expansion(calendar_utils):
    instances = calendar_utils.expand_recurring_event(
        recurring_event('RRULE:FREQ=DAILY;COUNT=3'),
        EVENT_START_NS - DAY_NS,
        EVENT_START_NS + 3 * DAY_NS,
        tz='Asia/Tokyo',
        max_instances=2,
    )
    assert len(instances) == 2


####################
# 81 — COUNT without DTSTART never exhausts
####################


def test_count_rule_without_dtstart_is_rejected(automations_module):
    """The rule has a limit but no anchor, so the limit can never be reached."""
    with pytest.raises(ValueError) as excinfo:
        automations_module.validate_rrule('RRULE:FREQ=DAILY;COUNT=5')

    assert 'DTSTART' in str(excinfo.value)


@pytest.mark.parametrize(
    'rule',
    [
        'RRULE:FREQ=DAILY;COUNT=5',
        'RRULE:FREQ=WEEKLY;COUNT=2;BYDAY=MO',
        'RRULE:FREQ=MONTHLY;COUNT=12',
        'RRULE:FREQ=HOURLY;INTERVAL=6;COUNT=4',
        'rrule:freq=daily;count=5',
        'RRULE:FREQ=DAILY;INTERVAL=2;COUNT=3;WKST=MO',
    ],
)
def test_every_count_rule_needs_an_anchor(automations_module, rule):
    """Broad: the check is on COUNT itself, in any casing and at any frequency."""
    with pytest.raises(ValueError, match='DTSTART'):
        automations_module.validate_rrule(rule)


@pytest.mark.parametrize(
    'rule',
    [
        'DTSTART:20260101T090000\nRRULE:FREQ=DAILY;COUNT=5000',
        'DTSTART:20260101T090000\nRRULE:FREQ=WEEKLY;COUNT=500;BYDAY=MO',
        'RRULE:FREQ=DAILY',
        'RRULE:FREQ=WEEKLY;BYDAY=MO,WE',
        f'RRULE:FREQ=DAILY;UNTIL={(dt.datetime.now() + dt.timedelta(days=30)):%Y%m%dT%H%M%S}',
    ],
)
def test_anchored_and_unlimited_rules_still_validate(automations_module, rule):
    """Nearby: a COUNT rule that carries a DTSTART, and rules with no COUNT at all."""
    automations_module.validate_rrule(rule)


def test_exhausted_rule_is_still_rejected_for_having_no_future_runs(automations_module):
    """Nearby: the new check must not shadow the existing exhaustion check."""
    with pytest.raises(ValueError, match='future'):
        automations_module.validate_rrule('DTSTART:20200101T090000\nRRULE:FREQ=DAILY;UNTIL=20200201T090000')


####################
# 115 — non-ASCII search across both JSON text spellings
####################


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('weather', ['weather']),
        ('', ['']),
        ('a b', ['a b']),
        (CJK, [CJK, '\\u5929\\u6c17']),
        ('café', ['café', 'caf\\u00e9']),
    ],
)
def test_json_text_variants_covers_both_spellings(misc_module, value, expected):
    """ASCII collapses to one variant; non-ASCII yields the raw and the escaped spelling."""
    assert misc_module.json_text_variants(value) == expected


def test_json_text_variants_escapes_the_quote_the_same_way_both_times(misc_module):
    """The variants are unquoted fragments, so an embedded quote stays JSON-escaped."""
    assert misc_module.json_text_variants('say "hi"') == ['say \\"hi\\"']


@pytest.mark.asyncio
async def test_automation_search_finds_an_escaped_prompt(db, automations_model, ids):
    """A row whose JSON was written with `\\uXXXX` escapes must still match the raw query.

    Pre-fix the search LIKEd the query text verbatim against the serialised column, so it
    only ever found rows written by an encoder that agreed with it.
    """
    now = int(time.time())
    stored_prompts = (
        ('raw', f'{{"prompt": "{CJK}"}}'),
        ('escaped', '{"prompt": "\\u5929\\u6c17"}'),
    )
    for suffix, stored in stored_prompts:
        await execute_sql(
            db,
            'INSERT INTO automation (id, user_id, name, data, is_active, created_at, updated_at) '
            'VALUES (:id, :user_id, :name, :data, 1, :now, :now)',
            {
                'id': f'{ids}-{suffix}',
                'user_id': ids,
                'name': f'automation {suffix}',
                'data': stored,
                'now': now,
            },
        )

    result = await automations_model.Automations.search_automations(ids, query=CJK)

    found = {automation.id for automation in result.items}
    assert f'{ids}-escaped' in found, 'escaped spelling was not matched'
    assert f'{ids}-raw' in found, 'raw spelling was not matched'


@pytest.mark.asyncio
async def test_model_tag_search_finds_a_raw_tag(db, models_model, ids):
    """A workspace model whose meta was stored as raw UTF-8 must match its own tag.

    Pre-fix SQLite always got the escaped pattern, so a row written by the orjson codec
    (the default since ENABLE_ORJSON) could not be found by tag at all.
    """
    now = int(time.time())
    for suffix, stored in (
        ('raw', f'{{"tags": [{{"name": "{CJK}"}}]}}'),
        ('escaped', '{"tags": [{"name": "\\u5929\\u6c17"}]}'),
    ):
        await execute_sql(
            db,
            'INSERT INTO model (id, user_id, base_model_id, name, params, meta, is_active, '
            'created_at, updated_at) '
            "VALUES (:id, :user_id, 'base-model', :name, '{}', :meta, 1, :now, :now)",
            {
                'id': f'{ids}-{suffix}',
                'user_id': ids,
                'name': f'model {suffix}',
                'meta': stored,
                'now': now,
            },
        )

    result = await models_model.Models.search_models(ids, filter={'tag': CJK, 'user_id': ids})

    found = {model.id for model in result.items}
    assert f'{ids}-raw' in found, 'raw spelling was not matched'
    assert f'{ids}-escaped' in found, 'escaped spelling was not matched'


@pytest.mark.asyncio
async def test_ascii_tag_search_is_unchanged(db, models_model, ids):
    """Nearby: an ASCII tag collapses to one variant and keeps matching case-insensitively."""
    now = int(time.time())
    await execute_sql(
        db,
        'INSERT INTO model (id, user_id, base_model_id, name, params, meta, is_active, '
        'created_at, updated_at) '
        "VALUES (:id, :user_id, 'base-model', :name, '{}', :meta, 1, :now, :now)",
        {
            'id': f'{ids}-ascii',
            'user_id': ids,
            'name': 'model ascii',
            'meta': '{"tags": [{"name": "Weather"}]}',
            'now': now,
        },
    )

    result = await models_model.Models.search_models(ids, filter={'tag': 'weather', 'user_id': ids})
    assert {model.id for model in result.items} == {f'{ids}-ascii'}

    unrelated = await models_model.Models.search_models(
        ids, filter={'tag': 'weathervane', 'user_id': ids}
    )
    assert unrelated.items == []


####################
# 134 — a user-writable alert_minutes stopped the shared reminder poll
####################


@pytest_asyncio.fixture
async def insert_event(db):
    """Seed calendar rows and drop them again: the poll is global, so a row left behind
    would decide the next test's result."""
    seeded: list[str] = []

    async def _insert(event_id: str, user_id: str, start_at: int, meta: str | None) -> None:
        now = int(time.time_ns())
        seeded.append(event_id)
        await execute_sql(
            db,
            'INSERT INTO calendar_event (id, calendar_id, user_id, title, start_at, all_day, meta, '
            'is_cancelled, created_at, updated_at) '
            'VALUES (:id, :calendar_id, :user_id, :title, :start_at, 0, :meta, 0, :now, :now)',
            {
                'id': event_id,
                'calendar_id': f'cal-{user_id}',
                'user_id': user_id,
                'title': f'Event {event_id}',
                'start_at': start_at,
                'meta': meta,
                'now': now,
            },
        )

    yield _insert

    for event_id in seeded:
        await execute_sql(db, 'DELETE FROM calendar_event WHERE id = :id', {'id': event_id})


@pytest.mark.parametrize('poison', ['"10"', '["10"]', '{"minutes": 10}', '"not a number"'])
@pytest.mark.asyncio
async def test_non_numeric_alert_minutes_does_not_break_the_poll(
    insert_event, calendar_model, ids, poison
):
    """One event with a junk alert window must not take the whole reminder sweep down.

    Pre-fix the value went straight into `alert_minutes < 0`, which raises for anything
    that is not a number, so every user's reminders stopped until the row was fixed.
    """
    now_ns = int(time.time_ns())
    poison_meta = f'{{"alert_minutes": {poison}}}'
    await insert_event(f'{ids}-poison', f'{ids}-a', now_ns + 5 * MINUTE_NS, poison_meta)
    await insert_event(f'{ids}-plain', f'{ids}-b', now_ns + 5 * MINUTE_NS, None)

    events = await calendar_model.CalendarEvents.get_upcoming_events(now_ns, 30 * MINUTE_NS)

    found = {event.id for event, _ in events}
    assert f'{ids}-plain' in found, "a second user's event was lost to the junk value"
    assert f'{ids}-poison' in found, 'the junk value should fall back to the default window'


@pytest.mark.asyncio
async def test_numeric_alert_minutes_still_narrows_the_window(insert_event, calendar_model, ids):
    """Nearby: a real number is honoured, so an event outside its own window is withheld."""
    now_ns = int(time.time_ns())
    await insert_event(f'{ids}-soon', f'{ids}-a', now_ns + 5 * MINUTE_NS, '{"alert_minutes": 10}')
    await insert_event(f'{ids}-later', f'{ids}-a', now_ns + 20 * MINUTE_NS, '{"alert_minutes": 10}')

    events = await calendar_model.CalendarEvents.get_upcoming_events(now_ns, 30 * MINUTE_NS)

    found = {event.id for event, _ in events}
    assert f'{ids}-soon' in found
    assert f'{ids}-later' not in found


@pytest.mark.asyncio
async def test_negative_alert_minutes_still_means_no_alert(insert_event, calendar_model, ids):
    """Nearby: the negative sentinel keeps suppressing the event."""
    now_ns = int(time.time_ns())
    await insert_event(f'{ids}-muted', f'{ids}-a', now_ns + 5 * MINUTE_NS, '{"alert_minutes": -1}')

    events = await calendar_model.CalendarEvents.get_upcoming_events(now_ns, 30 * MINUTE_NS)

    assert f'{ids}-muted' not in {event.id for event, _ in events}


####################
# 113 / 159 — timer execution and forked timer chats
####################


class RecordingHandler:
    """Stands in for app.state.CHAT_COMPLETION_HANDLER, the one I/O boundary here."""

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = 0

    async def __call__(self, request, form_data, user=None):
        self.calls += 1
        if self.error:
            raise self.error
        return {'choices': []}


def fake_app(handler) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(redis=None, CHAT_COMPLETION_HANDLER=handler))


async def seed_user(users_model, user_id: str):
    return await users_model.Users.insert_new_user(
        user_id, f'User {user_id}', f'{user_id}@example.test', role='user'
    )


async def seed_parent_chat(chats_model, chat_id: str, user_id: str):
    return await chats_model.Chats.insert_new_chat(
        chat_id,
        user_id,
        chats_model.ChatForm(chat={'id': chat_id, 'title': 'Parent', 'history': {'messages': {}}}),
    )


async def seed_timer(timers_module, user, parent_chat_id: str, due_in_seconds: int = 60) -> None:
    due_at = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=due_in_seconds)
    result = await timers_module.create_timer(
        prompt='remind me',
        at=due_at.isoformat(),
        cancel_on=[],
        request=None,
        user_data=user.model_dump(),
        metadata={'model_id': 'test-model'},
        parent_chat_id=parent_chat_id,
        parent_message_id=None,
    )
    assert not result.startswith('Error'), result


async def timer_row_for_user(db, chats_model, user_id: str):
    async with db.get_async_db_context() as session:
        result = await session.execute(
            select(chats_model.Chat).where(chats_model.Chat.user_id == user_id)
        )
        rows = result.scalars().all()
        timers = [row for row in rows if (row.meta or {}).get('type') == 'timer']
        assert len(timers) == 1, f'expected exactly one timer chat, got {len(timers)}'
        return timers[0].id, dict(timers[0].meta or {})


@pytest.mark.asyncio
async def test_failed_completion_marks_the_timer_as_errored(
    db, chats_model, users_model, timers_module, ids
):
    """A timer whose completion raises records the error instead of looking delivered.

    Pre-fix the exception escaped `execute_due_timer` and the row kept the `completed`
    status written just before the handler ran, so the user saw a finished timer and no reply.
    """
    user = await seed_user(users_model, ids)
    await seed_parent_chat(chats_model, f'{ids}-parent', user.id)
    await seed_timer(timers_module, user, f'{ids}-parent')
    timer_id, _ = await timer_row_for_user(db, chats_model, user.id)

    claimed = await timers_module.claim_due_timers(well_past_due(), limit=100)
    claim_id = dict(claimed)[timer_id]

    handler = RecordingHandler(RuntimeError('model not found'))
    await timers_module.execute_due_timer(fake_app(handler), timer_id, claim_id)

    assert handler.calls == 1
    timer = await chats_model.Chats.get_chat_by_id(timer_id)
    assert timer.meta['status'] == 'error', 'a failed completion was still reported as delivered'
    assert 'model not found' in timer.meta.get('timer_error', '')


@pytest.mark.asyncio
async def test_successful_completion_stays_completed(
    db, chats_model, users_model, timers_module, ids
):
    """Nearby: the happy path is untouched, no error state and no error text."""
    user = await seed_user(users_model, ids)
    await seed_parent_chat(chats_model, f'{ids}-parent', user.id)
    await seed_timer(timers_module, user, f'{ids}-parent')
    timer_id, _ = await timer_row_for_user(db, chats_model, user.id)

    claimed = await timers_module.claim_due_timers(well_past_due(), limit=100)
    claim_id = dict(claimed)[timer_id]

    handler = RecordingHandler()
    await timers_module.execute_due_timer(fake_app(handler), timer_id, claim_id)

    timer = await chats_model.Chats.get_chat_by_id(timer_id)
    assert timer.meta['status'] == 'completed'
    assert 'timer_error' not in timer.meta


@pytest.mark.asyncio
async def test_forked_timer_chat_is_not_claimed(db, chats_model, users_model, timers_module, ids):
    """Forking a timer chat copies its meta, and the copy must not become a second claim.

    The fork route re-inserts the source meta verbatim, which pre-fix was the whole
    definition of a claimable timer, so the scheduler fired the same timer from both rows.
    """
    user = await seed_user(users_model, ids)
    await seed_parent_chat(chats_model, f'{ids}-parent', user.id)
    await seed_timer(timers_module, user, f'{ids}-parent')
    timer_id, timer_meta = await timer_row_for_user(db, chats_model, user.id)

    fork_id = f'{ids}-fork'
    fork = await chats_model.Chats.insert_new_chat(
        fork_id,
        user.id,
        chats_model.ChatForm(
            chat={'id': fork_id, 'title': 'Timer (fork)', 'history': {'messages': {}}}
        ),
        internal_meta={**timer_meta, 'forked_from': timer_id},
    )
    assert fork is not None

    claimed = dict(await timers_module.claim_due_timers(well_past_due(), limit=100))

    assert timer_id in claimed, 'the real timer was not claimed'
    assert fork_id not in claimed, 'the forked copy was claimed as a second timer'


@pytest.mark.asyncio
async def test_timer_not_yet_due_is_left_alone(db, chats_model, users_model, timers_module, ids):
    """Nearby: claiming still respects the due time."""
    user = await seed_user(users_model, ids)
    await seed_parent_chat(chats_model, f'{ids}-parent', user.id)
    await seed_timer(timers_module, user, f'{ids}-parent', due_in_seconds=3600)
    timer_id, _ = await timer_row_for_user(db, chats_model, user.id)

    claimed = dict(await timers_module.claim_due_timers(int(time.time_ns()), limit=100))

    assert timer_id not in claimed
