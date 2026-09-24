"""Regression: one knowledge-search pattern must not be able to stall the worker.

open-webui 0.11.0 fix `3ab202626` (PR #27471): `build_matcher` in
`open_webui/tools/knowledge_fs.py` compiled a caller-supplied pattern with Python's backtracking
`re` and ran it over every line of every reachable file, with no timeout. A catastrophic pattern
such as `(a|aa)+$` against a long non-matching line costs exponential time, the search loop is
synchronous inside an async handler, and the default worker count is 1, so a single
model-issued search froze the whole instance for minutes.

dev matches with RE2, which is linear-time, and keeps a per-tool-call budget
(`MATCH_BUDGET_SECONDS`, entered with `match_budget()`) charged only for the time spent inside
searches. Stays a unit test: over HTTP a regression would wedge the shared instance, so the
catastrophic pattern runs in a child process with a hard timeout, and the budget is driven with
a scripted clock.

Discriminates: passes on dev `bbfa876af`; with RE2 swapped for `re` the catastrophic search
never returns, with a fresh budget per matcher the shared-budget test fails, and with the clock
started when the matcher is built the waiting test fails.
"""

import json
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CATASTROPHIC_PATTERN = "(a|aa)+$"
NON_MATCHING_LINE = "a" * 48 + "!"
SUBPROCESS_CEILING_SECONDS = 25.0
# Each clock reading moves this far, so every search costs a fraction of the budget.
CLOCK_STEP_SECONDS = 0.3

RUNNER = """
import importlib, json, sys, time

sys.path.insert(0, sys.argv[1])
knowledge_fs = importlib.import_module("open_webui.tools.knowledge_fs")

with knowledge_fs.match_budget():
    matcher, error = knowledge_fs.build_matcher(sys.argv[2], use_regex=True)
    started = time.monotonic()
    try:
        outcome = "returned %r" % matcher(sys.argv[3])
    except Exception as e:
        outcome = type(e).__name__
    elapsed = time.monotonic() - started

print(json.dumps({"outcome": outcome, "elapsed": elapsed, "error": error}))
"""

LINES = ["alpha beta", "gamma delta", "ALPHA omega", "epsilon"]


class SteppingClock:
    """Stands in for the `time` module: every reading is a step later than the last."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        self.now += CLOCK_STEP_SECONDS
        return self.now


@pytest.fixture(scope="session")
def knowledge_fs(owui_module):
    return owui_module("open_webui.tools.knowledge_fs")


@pytest.fixture
def clock(knowledge_fs, monkeypatch) -> SteppingClock:
    stepping = SteppingClock()
    monkeypatch.setattr(knowledge_fs, "time", stepping)
    return stepping


def _searches_until_refused(knowledge_fs, matcher, limit: int = 50) -> int:
    for search in range(1, limit + 1):
        try:
            matcher("a line")
        except knowledge_fs.MatchBudgetExceeded:
            return search
    raise AssertionError(f"{limit} searches never exhausted the budget")


def _matching_lines(matcher, lines):
    return [line for line in lines if matcher(line)]


# ── narrow: the catastrophic pattern answers instead of running forever ──


def test_catastrophic_pattern_answers_without_stalling(knowledge_fs, open_webui_backend, tmp_path):
    """Out of process: an unbounded matcher never returns, so this cannot hang the suite."""
    script = tmp_path / "run_catastrophic_match.py"
    script.write_text(RUNNER, encoding="utf-8")
    command = [
        sys.executable,
        str(script),
        str(open_webui_backend),
        CATASTROPHIC_PATTERN,
        NON_MATCHING_LINE,
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=SUBPROCESS_CEILING_SECONDS
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"matching {CATASTROPHIC_PATTERN!r} against a {len(NON_MATCHING_LINE)}-character "
            f"line was still running after {SUBPROCESS_CEILING_SECONDS:g}s, so a single "
            "model-issued knowledge search holds the worker and every other user waits (#27471)"
        )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["outcome"] in ("MatchBudgetExceeded", "returned False"), result
    assert result["elapsed"] < 3 * knowledge_fs.MATCH_BUDGET_SECONDS, (
        f"the matcher took {result['elapsed']:.1f}s, far past the "
        f"{knowledge_fs.MATCH_BUDGET_SECONDS:g}s budget it is supposed to enforce (#27471)"
    )


# ── broad: one budget per tool call, charged only while searching ────────


def test_searches_in_one_call_share_a_single_budget(knowledge_fs, clock):
    """A pipeline builds one matcher per segment; they must all draw on one budget."""
    with knowledge_fs.match_budget():
        first_matcher, _ = knowledge_fs.build_matcher("a", use_regex=True)
        searches_alone = _searches_until_refused(knowledge_fs, first_matcher)

    with knowledge_fs.match_budget():
        first_matcher, _ = knowledge_fs.build_matcher("a", use_regex=True)
        second_matcher, _ = knowledge_fs.build_matcher("line", use_regex=True)
        for _ in range(searches_alone - 1):
            first_matcher("a line")
        with pytest.raises(knowledge_fs.MatchBudgetExceeded):
            second_matcher("a line")


def test_each_call_starts_with_a_full_budget(knowledge_fs, clock):
    """A spent budget must not leak into the next tool call and refuse everything."""
    with knowledge_fs.match_budget():
        matcher, _ = knowledge_fs.build_matcher("a", use_regex=True)
        _searches_until_refused(knowledge_fs, matcher)

    with knowledge_fs.match_budget():
        matcher, _ = knowledge_fs.build_matcher("a", use_regex=True)
        try:
            matched = matcher("a line")
        except knowledge_fs.MatchBudgetExceeded:
            pytest.fail("a later tool call inherited the previous call's spent budget (#27471)")
    assert matched is True


def test_waiting_between_searches_does_not_drain_the_budget(knowledge_fs, clock):
    """Database round-trips and other coroutines run between searches and cost nothing."""
    with knowledge_fs.match_budget():
        matcher, _ = knowledge_fs.build_matcher("a", use_regex=True)
        clock.now += 10 * knowledge_fs.MATCH_BUDGET_SECONDS

        assert matcher("a line") is True, (
            "time spent outside search() was charged to the budget, so a slow database "
            "makes legitimate searches fail (#27471)"
        )


# ── nearby: ordinary searching still behaves ─────────────────────────────


@pytest.mark.parametrize(
    "pattern, options, expected",
    [
        ("alpha|gamma", {"use_regex": True}, ["alpha beta", "gamma delta"]),
        ("alpha", {"use_regex": True, "case_insensitive": True}, ["alpha beta", "ALPHA omega"]),
        ("beta", {"use_regex": False}, ["alpha beta"]),
        ("OMEGA", {"use_regex": False, "case_insensitive": True}, ["ALPHA omega"]),
        (r"alpha\|epsilon", {"use_regex": True}, ["alpha beta", "epsilon"]),
    ],
)
def test_ordinary_searches_return_the_right_lines(knowledge_fs, pattern, options, expected):
    with knowledge_fs.match_budget():
        matcher, error = knowledge_fs.build_matcher(pattern, **options)
        assert error is None
        assert _matching_lines(matcher, LINES) == expected


@pytest.mark.parametrize("pattern", ["(unclosed", "*nothing-to-repeat", "[a-"])
def test_invalid_regex_reports_an_error_instead_of_raising(knowledge_fs, pattern):
    matcher, error = knowledge_fs.build_matcher(pattern, use_regex=True)

    assert matcher is None
    assert error.startswith("Invalid"), (
        f"{pattern!r} has to come back as a readable error for the model, not as an "
        "exception out of the tool call"
    )
