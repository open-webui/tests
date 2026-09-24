"""Regression: an automation's repeat step must come from the rule dateutil parsed.

open-webui 0.11.1 fix `067114c28`: `_parse_rule` scraped FREQ and INTERVAL out of the RRULE
text and only moved a far-past DTSTART up to the present when that scrape said SECONDLY,
MINUTELY or HOURLY. dateutil accepts more than the scrape (lowercase keywords, a leading
space), so a crafted rule parsed as a one-second recurrence from the year 2000 and the first
`.after(now)` walked about 8x10^8 occurrences. The same rewrite dropped the timezone dateutil
keeps on a `DTSTART;TZID=...` line, which had made such a rule fail against the naive "now".

Every rule here goes through POST /api/v1/automations/create, which validates it and returns
the next runs. The instance is this module's own and the client timeout is short: on a
checkout without the fix (or without the 2 s evaluation budget) the crafted rule may wedge
the server.

Twin of unit/security/test_recurrence_rule_parsing.py.

Discriminates: passes on dev bbfa876af, fails with the pre-`067114c28` text scrape restored
(the TZID rules give 400, both crafted rules run out the 2 s budget and give 400).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from harness.actors import create_user
from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

SECOND_NS = 1_000_000_000
FAR_PAST_START = "DTSTART:20000101T000000"
MIN_INTERVAL_SECONDS = 600


@pytest.fixture(scope="module")
def automations_instance(instance_with):
    return instance_with(
        {
            "USER_PERMISSIONS_FEATURES_AUTOMATIONS": "true",
            "AUTOMATION_MIN_INTERVAL": str(MIN_INTERVAL_SECONDS),
        }
    )


@pytest.fixture
def create(automations_instance):
    """`create(rule, actor=None)` posts an inactive automation; the admin bypasses limits."""

    def post(rule: str, actor=None):
        token = actor.token if actor else None
        with automations_instance.client(token) as client:
            client.timeout = 30.0
            return client.post(
                "/api/v1/automations/create",
                json={
                    "name": "schedule",
                    "is_active": False,
                    "data": {"prompt": "ping", "model_id": MOCK_MODEL_ID, "rrule": rule},
                },
            )

    return post


def _runs(response) -> list[datetime]:
    assert response.status_code == 200, response.text
    return [datetime.fromtimestamp(run / SECOND_NS) for run in response.json()["next_runs"]]


def _steps(runs: list[datetime]) -> set[timedelta]:
    return {later - earlier for earlier, later in zip(runs, runs[1:])}


def _local(moment: datetime) -> str:
    return moment.strftime("%Y%m%dT%H%M%S")


# dateutil reads both as one-second rules; a text scrape of FREQ sees neither.
@pytest.mark.parametrize(
    "crafted_rule",
    ["RRULE:freq=secondly;interval=1", " RRULE:FREQ=SECONDLY;INTERVAL=1"],
    ids=["lowercase", "leading-space"],
)
def test_a_crafted_rule_cannot_keep_a_far_past_start(create, crafted_rule):
    started = datetime.now()

    runs = _runs(create(f"{FAR_PAST_START}\n{crafted_rule}"))

    assert started - timedelta(seconds=5) < runs[0] < started + timedelta(seconds=30), (
        f"first run {runs[0]}; a lowercase rule kept its year-2000 start and the server walked "
        "every second since then to find the next run"
    )
    assert _steps(runs) == {timedelta(seconds=1)}


def test_a_start_carrying_a_timezone_still_schedules(create):
    runs = _runs(create("DTSTART;TZID=America/New_York:20200101T090000\nRRULE:FREQ=DAILY"))

    assert _steps(runs) == {timedelta(days=1)}


def test_a_recent_sub_daily_start_carrying_a_timezone_still_schedules(create):
    recent = datetime.now() - timedelta(minutes=30)

    runs = _runs(
        create(f"DTSTART;TZID=America/New_York:{_local(recent)}\nRRULE:FREQ=MINUTELY;INTERVAL=5")
    )

    assert _steps(runs) == {timedelta(minutes=5)}


@pytest.mark.parametrize(
    "frequency, step",
    [
        ("SECONDLY", timedelta(seconds=1)),
        ("MINUTELY", timedelta(minutes=1)),
        ("HOURLY", timedelta(hours=1)),
    ],
)
def test_a_far_past_start_is_realigned_for_every_sub_daily_frequency(create, frequency, step):
    started = datetime.now()

    runs = _runs(create(f"{FAR_PAST_START}\nRRULE:FREQ={frequency}"))

    assert started - step < runs[0] <= datetime.now() + step
    assert _steps(runs) == {step}


def test_a_recent_start_is_honoured(create):
    start = datetime.now().replace(microsecond=0) - timedelta(minutes=10)

    runs = _runs(create(f"DTSTART:{_local(start)}\nRRULE:FREQ=MINUTELY;INTERVAL=5"))

    assert runs[0] == start + timedelta(minutes=15)


def test_a_rule_without_a_start_snaps_to_clock_boundaries(create):
    runs = _runs(create("RRULE:FREQ=MINUTELY;INTERVAL=5"))

    assert all(run.minute % 5 == 0 and run.second == 0 for run in runs), runs


def test_a_daily_rule_previews_five_runs(create):
    runs = _runs(create("RRULE:FREQ=DAILY"))

    assert len(runs) == 5
    assert _steps(runs) == {timedelta(days=1)}


@pytest.mark.parametrize(
    "rule",
    [
        "RRULE:FREQ=DAILY\nEXRULE:FREQ=WEEKLY",
        "RRULE:FREQ=DAILY\nRRULE:FREQ=WEEKLY",
        "RRULE:FREQ=MINUTELY;INTERVAL=0",
        "RRULE:FREQ=NEVERLY",
        "DTSTART:20200101T090000\nRRULE:FREQ=DAILY;COUNT=3",
    ],
    ids=["exrule", "two-rrules", "zero-interval", "unknown-frequency", "exhausted"],
)
def test_unsupported_and_exhausted_rules_are_refused(create, rule):
    assert create(rule).status_code == 400


def test_the_minimum_interval_measures_the_parsed_step(automations_instance, create):
    member = create_user(automations_instance)
    tomorrow = _local(datetime.now() + timedelta(days=1))

    assert create("RRULE:FREQ=MINUTELY;INTERVAL=5", member).status_code == 400
    assert create("RRULE:FREQ=DAILY", member).status_code == 200
    assert create(f"DTSTART:{tomorrow}\nRRULE:FREQ=DAILY;COUNT=1", member).status_code == 200
