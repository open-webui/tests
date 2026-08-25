"""Regression: a knowledge-search pattern must not be able to blow up before it runs.

open-webui 0.11.1 fix `d5b66533e` (PR #28284): `build_matcher` in
`open_webui/tools/knowledge_fs.py` handed any caller-supplied pattern straight
to `regex.compile`. The `regex` module materialises counted quantifiers at
compile time, so cost grows with the product of the counts, not with the length
of the pattern. Measured on this checkout, `(a{4000}){4000}` (16M expansion)
takes ~2.3s and ~4.1 GB of resident memory to compile, and nesting one more
level is enough to take the box down. A model-issued search is all it takes.

The fix adds `validate_regex_quantifiers`, called before compilation: a single
count over `MAX_REGEX_QUANTIFIER_COUNT = 2_000` is refused, and the running
product of every count in the pattern is refused once it passes
`MAX_REGEX_QUANTIFIER_EXPANSION = 100_000`. Refusal comes back as the ordinary
`(None, error)` pair the model already understands.

This is a different bound from the one guarded by
`test_knowledge_search_match_budget.py`. That one is `MATCH_BUDGET_SECONDS`,
charged while `search()` backtracks over a line; it can only help once a
pattern is compiled and running. This one is spent inside `regex.compile`,
before a single line is read, so no match-time budget can ever see it.

Every pattern used here is bounded by construction, so on the pre-fix ref the
tests pay the compile they are refusing rather than wedging: the priciest,
`a{1234567}`, measures 0.16s and ~340 MB, and the rest stay under 60 MB. The
tests observe the refusal itself and never run the real bomb.

Discriminates: passes on v0.11.1, fails on v0.11.0 (which compiles the
expansion bomb and returns a working matcher with no error).
"""

import time

import pytest

pytestmark = [pytest.mark.regression]

# 500 * 500 = 250,000, over the 100,000 expansion limit, while each individual
# count stays under the 2,000 per-count limit: only the product catches it.
EXPANSION_BOMB = "(a{500}){500}"

LINES = ["aaa found", "aa short", "42 answers", "nothing here"]


@pytest.fixture(scope="session")
def knowledge_fs(owui_module):
    return owui_module("open_webui.tools.knowledge_fs")


def _matching_lines(matcher, lines):
    return [line for line in lines if matcher(line)]


# ── narrow: the expansion bomb is refused before anything compiles it ────


def test_expansion_bomb_is_refused_instead_of_compiled(knowledge_fs):
    started = time.monotonic()
    matcher, error = knowledge_fs.build_matcher(EXPANSION_BOMB, use_regex=True)
    elapsed = time.monotonic() - started

    assert matcher is None and error, (
        f"{EXPANSION_BOMB!r} was accepted and compiled in {elapsed:.2f}s, so a model can "
        "nest the counts a little further and spend gigabytes of the worker's memory "
        "before a single line is searched (#28284)"
    )
    assert "quantifier" in error.lower(), (
        f"the refusal reads {error!r}, which does not tell the model the counts are the "
        "problem, so it cannot fix its own pattern (#28284)"
    )
    # ordering is the fix: a compiler-shaped error means the cost was already paid
    assert not error.startswith("Invalid regex:"), (
        f"the pattern was only rejected by the compiler ({error!r}) rather than up front, "
        "so the expansion is materialised before anything inspects the counts (#28284)"
    )


def test_a_single_oversized_count_is_refused(knowledge_fs):
    pattern = "a{50000}"
    matcher, error = knowledge_fs.build_matcher(pattern, use_regex=True)

    assert matcher is None and error, (
        f"{pattern!r} was accepted, so one quantifier is enough to make the compiler "
        f"materialise 50,000 copies of a branch (#28284)"
    )


# ── broad: where the two limits sit, and what they do not count ──────────


@pytest.mark.parametrize(
    "pattern",
    [
        "a{2001}",  # one count just over MAX_REGEX_QUANTIFIER_COUNT
        "a{1234567}",  # more digits than the validator will parse
        "(a{1000}){101}",  # product just over MAX_REGEX_QUANTIFIER_EXPANSION
        "((a{50}){50}){50}",  # three nested levels, 125,000
        "a{1000}b{1000}",  # side by side, not nested, still 1,000,000
    ],
)
def test_costly_quantifier_combinations_are_rejected(knowledge_fs, pattern):
    matcher, error = knowledge_fs.build_matcher(pattern, use_regex=True)

    assert matcher is None and error, (
        f"{pattern!r} was accepted, so its expansion is paid in full at compile time (#28284)"
    )


@pytest.mark.parametrize(
    "pattern",
    [
        "a{2000}",  # exactly at the per-count limit
        "(a{1000}){100}",  # exactly at the expansion limit
        r"\d{2,4}-\w{1,8}",  # the shape real searches use
        r"version \{3000\}",  # escaped braces are literal text, not a quantifier
        "no quantifiers at all",
    ],
)
def test_ordinary_quantifier_use_is_accepted(knowledge_fs, pattern):
    matcher, error = knowledge_fs.build_matcher(pattern, use_regex=True)

    assert error is None and matcher is not None, (
        f"{pattern!r} was rejected ({error!r}), so the bound is tight enough to break "
        "searches that cost nothing to compile (#28284)"
    )


def test_literal_search_is_not_subject_to_the_limit(knowledge_fs):
    """Braces in a literal search are just characters; refusing them would break
    searching for text that happens to contain them."""
    matcher, error = knowledge_fs.build_matcher("retries{50000}", use_regex=False)

    assert error is None and matcher is not None
    assert matcher("config retries{50000} here") is True


# ── nearby: quantified patterns that should still search normally ────────


def test_counted_quantifier_still_matches(knowledge_fs):
    with knowledge_fs.match_budget():
        matcher, error = knowledge_fs.build_matcher("a{3}", use_regex=True)
        assert error is None
        assert _matching_lines(matcher, LINES) == ["aaa found"]


def test_bounded_range_quantifier_still_matches(knowledge_fs):
    with knowledge_fs.match_budget():
        matcher, error = knowledge_fs.build_matcher(r"\d{2}", use_regex=True)
        assert error is None
        assert _matching_lines(matcher, LINES) == ["42 answers"]


def test_quantified_pattern_honours_case_insensitivity(knowledge_fs):
    with knowledge_fs.match_budget():
        matcher, error = knowledge_fs.build_matcher("A{2}", case_insensitive=True, use_regex=True)
        assert error is None
        assert _matching_lines(matcher, LINES) == ["aaa found", "aa short"]
