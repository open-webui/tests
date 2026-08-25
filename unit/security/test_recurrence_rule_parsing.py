"""Recurrence rule parsing must derive its repeat step from the rule dateutil
actually PARSED, never from the raw text of the rule.

The pre-fix `_parse_rule` scraped FREQ/INTERVAL out of the RRULE string with its
own mini-parser and only applied the clock-alignment + occurrence guard when that
text scrape said the rule was SECONDLY/MINUTELY/HOURLY. dateutil is far more
tolerant than the scrape (lowercase keywords, a leading space on the content
line), so a crafted rule could parse as a one-second recurrence while the text
scrape saw nothing at all. The guard was then skipped, the rule kept its
attacker-supplied DTSTART years in the past, and the first `.after(now)` walked
the server through hundreds of millions of occurrences one second at a time.

Every test here asserts on the DTSTART of the returned rule BEFORE walking it.
That ordering is deliberate: on a checkout without the fix the assertion fails
immediately and no occurrence walk is ever started, so this file can never wedge
a run. Walks that do happen are capped by count.

Discriminates: passes on v0.11.1, fails on v0.11.0 (the pre-fix text scrape misses
the crafted rule, so the far-past DTSTART survives and the returned rule is still
anchored in the year 2000).
"""

from datetime import datetime, timedelta

import pytest

pytestmark = pytest.mark.regression

# 2000-01-01 is the epoch the scheduler snaps sub-daily rules to. Using it as a
# crafted DTSTART is the worst case: ~26 years of one-second occurrences.
FAR_PAST_DTSTART = "DTSTART:20000101T000000"

# dateutil uppercases every RRULE token before parsing; the pre-fix text scrape
# did not, so these two lines parse as a real rule while scraping to nothing.
LOWERCASE_SECONDLY = "RRULE:freq=secondly;interval=1"
SPACED_HOURLY = " RRULE:FREQ=MINUTELY;INTERVAL=60"


def bounded_occurrences(rule, start: datetime, count: int) -> list[datetime]:
    """Walk at most *count* occurrences.

    Only ever called after the caller has asserted the rule is anchored near the
    present, so the walk is short by construction.
    """
    occurrences = []
    cursor = start
    for _ in range(count):
        cursor = rule.after(cursor)
        if cursor is None:
            break
        occurrences.append(cursor)
    return occurrences


####################
# Narrow: the fix itself
####################


def test_alignment_step_comes_from_the_parsed_rule(automations_module):
    """A rule whose text scrapes to nothing still gets the parsed 60-minute step.

    The leading space hides the content line from any text scrape. dateutil reads
    it as MINUTELY/INTERVAL=60, so the schedule must be snapped to a whole hour
    near *now*. Pre-fix the scrape saw no FREQ, skipped alignment entirely, and
    handed back the rule still anchored in the year 2000.
    """
    now = datetime(2026, 8, 25, 14, 37, 21)
    rule = automations_module._parse_rule(f"{FAR_PAST_DTSTART}\n{SPACED_HOURLY}", now)

    assert rule._interval == 60
    assert now - timedelta(hours=1) < rule._dtstart <= now, (
        f"rule still anchored at {rule._dtstart}; the far-past DTSTART was not replaced"
    )
    assert (rule._dtstart.minute, rule._dtstart.second) == (0, 0), (
        "start was not snapped to a whole hour, so the step did not come from the parsed rule"
    )

    occurrences = bounded_occurrences(rule, now, 3)
    assert len(occurrences) == 3
    assert [later - earlier for earlier, later in zip(occurrences, occurrences[1:])] == [
        timedelta(hours=1),
        timedelta(hours=1),
    ]


def test_crafted_rule_cannot_keep_a_far_past_start(automations_module):
    """Text and parse disagreeing must not buy the rule an unbounded walk.

    Lowercase `freq=secondly` parses as a one-second recurrence. With a DTSTART
    in 2000 the pre-fix code returned it untouched, and the very first
    `.after(now)` iterated ~8x10^8 times before answering. The rule handed back
    must be anchored at the present instead.
    """
    now = datetime(2026, 8, 25, 14, 37, 21)
    rule = automations_module._parse_rule(f"{FAR_PAST_DTSTART}\n{LOWERCASE_SECONDLY}", now)

    assert rule._dtstart > now - timedelta(minutes=1), (
        f"rule anchored at {rule._dtstart}; walking it from now would enumerate "
        "every second since then"
    )

    occurrences = bounded_occurrences(rule, now, 3)
    assert occurrences == [now + timedelta(seconds=n) for n in (1, 2, 3)]


def test_start_carrying_a_timezone_still_schedules(automations_module):
    """A DTSTART with a TZID must produce a next run, not an error.

    dateutil keeps the tzinfo on DTSTART even under ignoretz, so the pre-fix code
    returned a timezone-aware rule that blew up the moment the scheduler compared
    it against its naive "now".
    """
    rule_text = "DTSTART;TZID=America/New_York:20200101T090000\nRRULE:FREQ=DAILY"

    assert automations_module.next_run_ns(rule_text) is not None
    automations_module.validate_rrule(rule_text)


def test_sub_daily_start_carrying_a_timezone_still_schedules(automations_module):
    """Same for a sub-daily rule recent enough to keep its own DTSTART."""
    recent = datetime.now() - timedelta(minutes=30)
    rule_text = (
        f"DTSTART;TZID=America/New_York:{recent.strftime('%Y%m%dT%H%M%S')}\n"
        "RRULE:FREQ=MINUTELY;INTERVAL=5"
    )

    interval = automations_module.rrule_interval_seconds(rule_text)
    assert interval == 300


####################
# Broad: the guard the fix protects
####################


def test_far_past_start_is_realigned_for_every_sub_daily_frequency(automations_module):
    """No sub-daily rule may keep a start that is >100k occurrences behind now."""
    now = datetime(2026, 8, 25, 14, 37, 21)
    for freq, step in (
        ("SECONDLY", timedelta(seconds=1)),
        ("MINUTELY", timedelta(minutes=1)),
        ("HOURLY", timedelta(hours=1)),
    ):
        rule = automations_module._parse_rule(f"{FAR_PAST_DTSTART}\nRRULE:FREQ={freq}", now)
        assert now - step < rule._dtstart <= now, f"{freq} rule left anchored at {rule._dtstart}"


def test_recent_start_is_honoured(automations_module):
    """A start that is only a few occurrences back is the user's intent: keep it."""
    now = datetime(2026, 8, 25, 14, 37, 21)
    start = now - timedelta(minutes=10)
    rule = automations_module._parse_rule(
        f"DTSTART:{start.strftime('%Y%m%dT%H%M%S')}\nRRULE:FREQ=MINUTELY;INTERVAL=5", now
    )

    assert rule._dtstart == start
    assert bounded_occurrences(rule, now, 1) == [start + timedelta(minutes=15)]


def test_rule_without_a_start_snaps_to_clock_boundaries(automations_module):
    """Every 5 minutes means the clock boundaries :00, :05, :10."""
    now = datetime(2026, 8, 25, 14, 37, 21)
    rule = automations_module._parse_rule("RRULE:FREQ=MINUTELY;INTERVAL=5", now)

    assert bounded_occurrences(rule, now, 2) == [
        datetime(2026, 8, 25, 14, 40),
        datetime(2026, 8, 25, 14, 45),
    ]


####################
# Nearby: rules that must still be refused or still work
####################


@pytest.mark.parametrize(
    "rule_text",
    [
        "RRULE:FREQ=DAILY\nEXRULE:FREQ=WEEKLY",
        "RRULE:FREQ=DAILY\nRRULE:FREQ=WEEKLY",
        "RRULE:FREQ=MINUTELY;INTERVAL=0",
        "RRULE:FREQ=NEVERLY",
    ],
)
def test_unsupported_rules_are_refused(automations_module, rule_text):
    with pytest.raises(ValueError):
        automations_module._parse_rule(rule_text, datetime(2026, 8, 25, 14, 37, 21))


def test_exhausted_rule_is_refused(automations_module):
    with pytest.raises(ValueError):
        automations_module.validate_rrule("DTSTART:20200101T090000\nRRULE:FREQ=DAILY;COUNT=3")


def test_daily_interval_and_preview(automations_module):
    rule_text = "RRULE:FREQ=DAILY"
    assert automations_module.rrule_interval_seconds(rule_text) == 86400

    preview = automations_module.next_n_runs_ns(rule_text, n=4)
    assert len(preview) == 4
    assert preview == sorted(preview)


def test_one_shot_rule_has_no_interval(automations_module):
    start = (datetime.now() + timedelta(days=1)).strftime("%Y%m%dT%H%M%S")
    rule_text = f"DTSTART:{start}\nRRULE:FREQ=DAILY;COUNT=1"
    assert automations_module.rrule_interval_seconds(rule_text) is None
