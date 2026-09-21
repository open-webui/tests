"""Regression: one knowledge-search pattern must not be able to stall the worker.

open-webui 0.11.0 fix `3ab202626` (PR #27471): `build_matcher` in
`open_webui/tools/knowledge_fs.py` compiled a caller-supplied pattern with
Python's backtracking `re` and ran it over every line of every reachable file,
with no timeout. A catastrophic pattern such as `(a|aa)+$` against a long
non-matching line costs exponential time, the search loop is synchronous inside
an async handler, and the default worker count is 1, so a single model-issued
search froze the whole instance for minutes.

The original fix moved matching onto the `regex` module with a per-search
timeout and a `MATCH_BUDGET_SECONDS = 2.0` budget scoped by `match_budget()`.
dev replaced the engine with Google's RE2 (`re2`), whose matching is
guaranteed linear-time, so an engine-level blow-up is no longer possible; the
budget stays as a secondary cap and is still enforced through the same
`match_budget()` / `MatchBudgetExceeded` surface, charged per `search()` call.

Discriminates: passes on dev; a checkout without any bound would never return
from the catastrophic pattern and has to be killed (that is also the v0.10.2
behaviour).
"""

import json
import subprocess
import sys
import time

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.slow]

# RE2 refuses to program-nest this one deeply enough to blow up, but the
# matcher must still come back with an answer or a budget error, never hang.
CATASTROPHIC_PATTERN = "(a|aa)+$"
NON_MATCHING_LINE = "a" * 48 + "!"

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
    """Spend the active budget directly and confirm the matcher refuses to run.

    RE2 answers the old catastrophic pattern instantly (that is the fix), so
    the budget can no longer be burnt by a real pattern on this hardware. The
    charging path a genuinely slow search takes ends at
    `budget.remaining <= 0`, so drive that state itself.
    """
    budget = knowledge_fs._active_budget.get()
    assert budget is not None, "no active match budget"
    budget.remaining = 0.0
    matcher, _ = knowledge_fs.build_matcher("a", use_regex=True)
    with pytest.raises(knowledge_fs.MatchBudgetExceeded):
        matcher("line")


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

    assert result["outcome"] in ("MatchBudgetExceeded", "returned False"), (
        f"the matcher answered {result['outcome']} after {result['elapsed']:.1f}s instead of "
        "finishing or abandoning the search, so pattern cost is still unbounded (#27471)"
    )
    assert result["elapsed"] < 3 * knowledge_fs.MATCH_BUDGET_SECONDS, (
        f"the matcher took {result['elapsed']:.1f}s to answer, far past the "
        f"{knowledge_fs.MATCH_BUDGET_SECONDS:g}s budget it is supposed to enforce (#27471)"
    )


# ── broad: one budget per tool call, reset on the way out ────────────────


def test_searches_in_one_call_share_a_single_budget(knowledge_fs, require_match_budget):
    """A pipeline builds one matcher per segment; every matcher reads the same
    contextvar budget, so one spent budget covers the whole tool call rather
    than refreshing per search."""
    with knowledge_fs.match_budget():
        first_matcher, _ = knowledge_fs.build_matcher("a", use_regex=True)
        knowledge_fs._active_budget.get().remaining = 0.0
        second_matcher, _ = knowledge_fs.build_matcher("b", use_regex=True)

        for segment_matcher in (first_matcher, second_matcher):
            with pytest.raises(knowledge_fs.MatchBudgetExceeded):
                segment_matcher("line")


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
        _exhaust_budget(knowledge_fs)


def test_time_outside_matching_does_not_drain_the_budget(knowledge_fs, require_match_budget):
    """The budget is charged inside search() only, so database round-trips and
    other coroutines cannot spend it."""
    with knowledge_fs.match_budget():
        time.sleep(0.4)
        matcher, _ = knowledge_fs.build_matcher("a", use_regex=True)

        assert knowledge_fs._active_budget.get().remaining > (
            knowledge_fs.MATCH_BUDGET_SECONDS - 0.1
        ), (
            "a wait outside search() spent the budget, so a slow database makes "
            "legitimate searches fail (#27471)"
        )
        assert matcher("a line") is True


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
    assert error.startswith("Invalid"), (
        f"{pattern!r} has to come back as a readable error for the model, not as an "
        "exception out of the tool call"
    )


def test_escaped_pipe_is_normalized_to_alternation(knowledge_fs):
    with knowledge_fs.match_budget():
        matcher, error = knowledge_fs.build_matcher(r"alpha\|epsilon", use_regex=True)
        assert error is None
        assert _matching_lines(matcher, LINES) == ["alpha beta", "epsilon"]
