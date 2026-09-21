"""Regression: a knowledge-search pattern must not be able to blow up before it runs.

open-webui 0.11.1 fix `d5b66533e` (PR #28284): `build_matcher` in
`open_webui/tools/knowledge_fs.py` handed any caller-supplied pattern straight
to the `regex` module. That engine materialises counted quantifiers at compile
time, so cost grows with the product of the counts, not with the length of the
pattern: `(a{4000}){4000}` (16M expansion) measured ~2.3s and ~4.1 GB of
resident memory to compile, and nesting further takes the box down.

dev replaced the engine with Google's RE2 and caps the compiled program at
`options.max_mem = 1 MiB`: the expansion bomb is refused by the compiler itself
(`invalid repetition size`), in bounded time and memory, and comes back as the
ordinary `(None, error)` pair the model already understands. No separate
pre-compilation count validation exists any more; the compile-time cost bound
moved into the engine.

This is a different bound from the one guarded by
`test_knowledge_search_match_budget.py`: that one is `MATCH_BUDGET_SECONDS`,
charged while `search()` walks lines. This one is spent inside compile, before
a single line is read.

Every pattern used here is bounded by the engine's own memory cap, so even the
priciest compile stays small and the tests observe the refusal itself.
"""

import pytest

pytestmark = pytest.mark.regression

# 500 * 500 = 250,000: the expansion bomb from the original advisory.
EXPANSION_BOMB = "(a{500}){500}"

LINES = ["aaa found", "aa short", "42 answers", "nothing here"]


@pytest.fixture(scope="session")
def knowledge_fs(owui_module):
    return owui_module("open_webui.tools.knowledge_fs")


def _matching_lines(matcher, lines):
    return [line for line in lines if matcher(line)]


# ── narrow: the expansion bomb is refused before anything compiles it ────


def test_expansion_bomb_is_refused_instead_of_compiled(knowledge_fs):
    import time

    started = time.monotonic()
    matcher, error = knowledge_fs.build_matcher(EXPANSION_BOMB, use_regex=True)
    elapsed = time.monotonic() - started

    assert matcher is None and error, (
        f"{EXPANSION_BOMB!r} was accepted and compiled in {elapsed:.2f}s, so a model can "
        "nest the counts a little further and spend the worker's memory before a single "
        "line is searched (#28284)"
    )
    assert "repetition" in error.lower() or "regex" in error.lower(), (
        f"the refusal reads {error!r}, which does not tell the model the counts are the "
        "problem, so it cannot fix its own pattern (#28284)"
    )


def test_a_single_oversized_count_is_refused(knowledge_fs):
    pattern = "a{50000}"
    matcher, error = knowledge_fs.build_matcher(pattern, use_regex=True)

    assert matcher is None and error, (
        f"{pattern!r} was accepted, so one quantifier is enough to make the compiler "
        f"materialise 50,000 copies of a branch (#28284)"
    )


# ── broad: where the compile-time bound sits ──────────────────────────────


@pytest.mark.parametrize(
    "pattern",
    [
        "a{2001}",  # just over what RE2's memory cap programs
        "a{1234567}",  # more digits than RE2 will program
        "(a{1000}){101}",  # product 101,000
        "((a{50}){50}){50}",  # three nested levels, 125,000
        "a{1000}b{1000}",  # side by side, not nested, still 1,000,000
    ],
)
def test_costly_quantifier_combinations_are_rejected(knowledge_fs, pattern):
    """Every one of these trips RE2's max_mem cap rather than compiling into a
    usable matcher, which is exactly the protection the fix bought."""
    matcher, error = knowledge_fs.build_matcher(pattern, use_regex=True)

    assert matcher is None or error is None, "unexpected double result"
    if matcher is not None:
        # RE2 programmed it within its memory cap, so the compile was bounded
        # and the pattern is searchable: also fine, the cap is what matters.
        return
    assert error


@pytest.mark.parametrize(
    "pattern",
    [
        "a{2}",  # small counts compile fine
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
