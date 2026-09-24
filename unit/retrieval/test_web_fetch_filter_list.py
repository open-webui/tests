"""Regression: malformed `WEB_FETCH_FILTER_LIST` entries must not turn into rules.

open-webui 0.11.0 fix `18719fef9` (#26910, issue #26908): `get_allow_block_lists` appended each
entry verbatim. Docker Compose list syntax passes surrounding quotes through, so a quoted entry
matched no host, and an empty or whitespace-only entry became the allow entry `''`. A non-empty
allow list refuses every host it does not match, so either one blocked every web address. The
fix strips quotes and whitespace, strips again after a leading `!` and drops empty entries.

The reported case, a quoted allow entry, is driven through a booted instance in
integration/retrieval/test_web_fetch_filter_list.py. The parsing cases below would each need a
boot of their own there, so they call the parser the fetch guard uses.

Discriminates: passes on dev bbfa876af; the pre-fix parser (entries only stripped of whitespace,
empty ones kept) fails the quoted and empty-entry cases, the other two pass on both.
"""

import pytest

pytestmark = pytest.mark.regression


def test_a_quoted_block_entry_is_unquoted(misc_module):
    """`!` marks a block entry; the quotes sit outside it."""
    assert misc_module.get_allow_block_lists(filter_list=["'!evil.com'"]) == ([], ["evil.com"])


def test_quotes_inside_the_bang_are_stripped_too(misc_module):
    assert misc_module.get_allow_block_lists(filter_list=['!"evil.com"']) == ([], ["evil.com"])


def test_empty_entries_are_dropped_not_turned_into_an_allow_rule(misc_module):
    """A blank entry would make an allow list out of nothing: the same total blackout."""
    entries = ["", "   ", '""', "!", '!""']

    assert misc_module.get_allow_block_lists(filter_list=entries) == ([], [])


def test_wellformed_entries_are_unchanged(misc_module):
    entries = ["example.com", "!evil.com", " spaced.com "]

    allow, block = misc_module.get_allow_block_lists(filter_list=entries)

    assert (allow, block) == (["example.com", "spaced.com"], ["evil.com"])


@pytest.mark.parametrize("entries", [None, []])
def test_no_entries_yield_no_rules(misc_module, entries):
    assert misc_module.get_allow_block_lists(filter_list=entries) == ([], [])
