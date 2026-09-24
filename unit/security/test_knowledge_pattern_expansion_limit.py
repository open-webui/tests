"""Regression: a knowledge-search pattern must not be able to blow up before it runs.

open-webui 0.11.1 fix `d5b66533e` (PR #28284): `build_matcher` in
`open_webui/tools/knowledge_fs.py` handed any caller-supplied pattern straight to the `regex`
module. That engine materialises counted quantifiers at compile time, so cost grows with the
product of the counts, not with the length of the pattern: `(a{4000}){4000}` measured ~2.3s and
~4.1 GB of resident memory to compile, and nesting further takes the box down.

dev compiles with RE2, whose parser refuses a count over 1000, or nested counts whose product
passes 1000, as `invalid repetition size`; the bomb comes back as the ordinary `(None, error)`
pair the model already understands. Stays a unit test: `build_matcher` is a pure function, and
over HTTP a regression would spend the shared instance's memory.

Discriminates: passes on dev `bbfa876af`; with RE2 swapped back for the `regex` module every
oversized pattern compiles into a matcher.
"""

import pytest

pytestmark = pytest.mark.regression

# 500 * 500 = 250,000: the expansion bomb from the original advisory.
EXPANSION_BOMB = "(a{500}){500}"

LINES = ["aaa found", "aa short", "42 answers", "nothing here"]


@pytest.fixture(scope="session")
def knowledge_fs(owui_module):
    return owui_module("open_webui.tools.knowledge_fs")


# ── narrow: the expansion bomb is refused instead of compiled ────────────


def test_expansion_bomb_is_refused_instead_of_compiled(knowledge_fs):
    matcher, error = knowledge_fs.build_matcher(EXPANSION_BOMB, use_regex=True)

    assert matcher is None and error, (
        f"{EXPANSION_BOMB!r} was compiled, so a model can nest the counts a little further and "
        "spend the worker's memory before a single line is searched (#28284)"
    )
    assert "repetition" in error.lower(), (
        f"the refusal reads {error!r}, which does not tell the model the counts are the "
        "problem, so it cannot fix its own pattern (#28284)"
    )


# ── broad: every count that multiplies past the cap ──────────────────────


@pytest.mark.parametrize(
    "pattern",
    [
        "a{50000}",  # one oversized count
        "a{2001}",  # just over the cap
        "(a{1000}){101}",  # product 101,000
        "((a{50}){50}){50}",  # three nested levels, 125,000
    ],
)
def test_counts_that_multiply_past_the_cap_are_refused(knowledge_fs, pattern):
    matcher, error = knowledge_fs.build_matcher(pattern, use_regex=True)

    assert matcher is None and error, f"{pattern!r} was compiled into a matcher (#28284)"


# ── nearby: patterns that cost nothing to compile still work ─────────────


@pytest.mark.parametrize(
    "pattern",
    [
        "a{2}",
        r"\d{2,4}-\w{1,8}",  # the shape real searches use
        r"version \{3000\}",  # escaped braces are literal text, not a quantifier
        "(a{30}){30}",  # nested, but the product stays under the cap
        "a{1000}b{1000}",  # side by side, the counts add rather than multiply
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
    """Braces in a literal search are just characters."""
    matcher, error = knowledge_fs.build_matcher("retries{50000}", use_regex=False)

    assert error is None
    assert matcher("config retries{50000} here") is True


@pytest.mark.parametrize(
    "pattern, case_insensitive, expected",
    [
        ("a{3}", False, ["aaa found"]),
        (r"\d{2}", False, ["42 answers"]),
        ("A{2}", True, ["aaa found", "aa short"]),
    ],
)
def test_quantified_patterns_still_match(knowledge_fs, pattern, case_insensitive, expected):
    with knowledge_fs.match_budget():
        matcher, error = knowledge_fs.build_matcher(
            pattern, case_insensitive=case_insensitive, use_regex=True
        )
        assert error is None
        assert [line for line in LINES if matcher(line)] == expected
