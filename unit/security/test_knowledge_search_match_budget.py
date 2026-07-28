"""Regression: one knowledge-search pattern must not be able to stall the worker.

open-webui 0.11.0 fix `3ab202626` (PR #27471): `build_matcher` in
`open_webui/tools/knowledge_fs.py` compiled a caller-supplied pattern with
Python's backtracking `re` and ran it over every line of every reachable file,
with no timeout. A catastrophic pattern such as `(a|aa)+$` against a long
non-matching line costs exponential time, the search loop is synchronous inside
an async handler, and the default worker count is 1, so a single model-issued
search froze the whole instance for minutes.

The fix moves matching onto the `regex` module, which takes a per-search
timeout, and spends a single `MATCH_BUDGET_SECONDS = 2.0` budget across a whole
tool call. The budget lives in a `contextvars` variable scoped by the
`match_budget()` context manager, is charged only for time spent inside
`search()`, and raises `MatchBudgetExceeded` once it runs out.

Discriminates: passes on v0.11.0, fails on v0.10.2 (the matcher there has no
timeout at all and no budget to scope, so the subprocess running the pattern
either dies for want of `match_budget` or has to be killed).
"""

import json
import subprocess
import sys
import time

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.slow]

# Not resolved by the regex module's optimiser, so the timeout is what stops it.
CATASTROPHIC_PATTERN = "(a|aa)+$"
# Cost doubles per added character, so this is minutes of work for an unbounded matcher.
NON_MATCHING_LINE = "a" * 48 + "!"

# Generous enough that a bounded matcher never trips it, short enough that an
# unbounded one does not hold the suite.
SUBPROCESS_CEILING_SECONDS = 25.0

RUNNER = '''
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
'''


@pytest.fixture(scope="session")
def knowledge_fs(owui_module):
    return owui_module("open_webui.tools.knowledge_fs")


@pytest.fixture
def require_match_budget(knowledge_fs):
    """Skip instead of hanging: without a budget the catastrophic pattern never returns."""
    if not hasattr(knowledge_fs, "match_budget"):
        pytest.skip("checkout has no match_budget(); the pattern would never return in-process")


def _exhaust_budget(knowledge_fs):
    """Burn the active budget on the catastrophic pattern, returning the seconds it took."""
    matcher, _ = knowledge_fs.build_matcher(CATASTROPHIC_PATTERN, use_regex=True)
    started = time.monotonic()
    with pytest.raises(knowledge_fs.MatchBudgetExceeded):
        matcher(NON_MATCHING_LINE)
    return time.monotonic() - started


def _matching_lines(matcher, lines):
    return [line for line in lines if matcher(line)]


# ── narrow: the catastrophic pattern gives up instead of running forever ──


def test_catastrophic_pattern_gives_up_within_the_budget(
    knowledge_fs, open_webui_backend, tmp_path
):
    """Run it out of process: an unbounded matcher never returns, so this is the
    only way to observe the bug without hanging the suite."""
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
            "model-issued knowledge search holds the worker and every other user waits "
            "for it (#27471)"
        )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])

    assert result["outcome"] == "MatchBudgetExceeded", (
        f"the matcher answered {result['outcome']} after {result['elapsed']:.1f}s instead of "
        "abandoning the search, so pattern cost is still unbounded (#27471)"
    )
    assert result["elapsed"] < 3 * knowledge_fs.MATCH_BUDGET_SECONDS, (
        f"the matcher took {result['elapsed']:.1f}s to give up, far past the "
        f"{knowledge_fs.MATCH_BUDGET_SECONDS:g}s budget it is supposed to enforce (#27471)"
    )


# ── broad: one budget per tool call, reset on the way out ────────────────


def test_searches_in_one_call_share_a_single_budget(knowledge_fs, require_match_budget):
    """A pipeline builds one matcher per segment; a per-search budget would
    multiply by segment count."""
    started = time.monotonic()
    with knowledge_fs.match_budget():
        _exhaust_budget(knowledge_fs)
        second_search_seconds = _exhaust_budget(knowledge_fs)
    total_seconds = time.monotonic() - started

    assert second_search_seconds < 0.5, (
        f"the second search in the same tool call got {second_search_seconds:.1f}s of "
        "fresh matching time, so a multi-segment command multiplies the stall by its "
        "segment count (#27471)"
    )
    assert total_seconds < 2 * knowledge_fs.MATCH_BUDGET_SECONDS, (
        f"one tool call spent {total_seconds:.1f}s matching, more than the single "
        f"{knowledge_fs.MATCH_BUDGET_SECONDS:g}s budget it is allowed (#27471)"
    )


def test_leaving_the_context_starts_the_next_call_clean(knowledge_fs, require_match_budget):
    """An exhausted budget must not leak into the next tool call and reject
    every pattern it sees."""
    with knowledge_fs.match_budget():
        _exhaust_budget(knowledge_fs)

    with knowledge_fs.match_budget():
        matcher, _ = knowledge_fs.build_matcher("needle", use_regex=True)
        assert matcher("a needle here") is True, (
            "a later tool call inherited the previous call's spent budget, so ordinary "
            "searches fail until the process restarts (#27471)"
        )
        replenished_seconds = _exhaust_budget(knowledge_fs)

    assert replenished_seconds > knowledge_fs.MATCH_BUDGET_SECONDS / 2, (
        f"the next tool call only got {replenished_seconds:.1f}s of matching time "
        "instead of a full budget (#27471)"
    )


def test_time_outside_matching_does_not_drain_the_budget(knowledge_fs, require_match_budget):
    """The budget is charged inside search() only, so database round-trips and
    other coroutines cannot spend it."""
    with knowledge_fs.match_budget():
        time.sleep(0.4)
        available_seconds = _exhaust_budget(knowledge_fs)

    assert available_seconds > knowledge_fs.MATCH_BUDGET_SECONDS / 2, (
        f"only {available_seconds:.1f}s of the budget survived a wait that did no "
        "matching, so a slow database makes legitimate searches fail (#27471)"
    )


# ── nearby: ordinary searching still behaves ─────────────────────────────


LINES = ["alpha beta", "gamma delta", "ALPHA omega", "epsilon"]


def test_ordinary_regex_returns_the_right_lines(knowledge_fs):
    with knowledge_fs.match_budget():
        matcher, error = knowledge_fs.build_matcher("alpha|gamma", use_regex=True)
        assert error is None
        assert _matching_lines(matcher, LINES) == ["alpha beta", "gamma delta"]


def test_case_insensitive_regex_still_matches(knowledge_fs):
    with knowledge_fs.match_budget():
        matcher, error = knowledge_fs.build_matcher("alpha", case_insensitive=True, use_regex=True)
        assert error is None
        assert _matching_lines(matcher, LINES) == ["alpha beta", "ALPHA omega"]


def test_literal_search_is_unaffected(knowledge_fs):
    with knowledge_fs.match_budget():
        matcher, error = knowledge_fs.build_matcher("beta", use_regex=False)
        assert error is None
        assert _matching_lines(matcher, LINES) == ["alpha beta"]


def test_case_insensitive_literal_search_is_unaffected(knowledge_fs):
    with knowledge_fs.match_budget():
        matcher, error = knowledge_fs.build_matcher("OMEGA", case_insensitive=True, use_regex=False)
        assert error is None
        assert _matching_lines(matcher, LINES) == ["ALPHA omega"]


@pytest.mark.parametrize("pattern", ["(unclosed", "*nothing-to-repeat", "[a-"])
def test_invalid_regex_reports_an_error_instead_of_raising(knowledge_fs, pattern):
    with knowledge_fs.match_budget():
        matcher, error = knowledge_fs.build_matcher(pattern, use_regex=True)
    assert matcher is None
    assert error.startswith("Invalid regex:"), (
        f"{pattern!r} has to come back as a readable error for the model, not as an "
        "exception out of the tool call"
    )


def test_escaped_pipe_is_normalized_to_alternation(knowledge_fs):
    with knowledge_fs.match_budget():
        matcher, error = knowledge_fs.build_matcher(r"alpha\|epsilon", use_regex=True)
        assert error is None
        assert _matching_lines(matcher, LINES) == ["alpha beta", "epsilon"]
