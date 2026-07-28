"""Regression: a malformed WEB_FETCH_FILTER_LIST entry must not block every fetch.

open-webui 0.11.0 fix `18719fef9` (#26910, issue #26908): `get_allow_block_lists`
appended each entry verbatim. Docker Compose list syntax passes surrounding
quotes through literally, so `- '"example.com"'` produced the allow entry
`"example.com"` (quotes included). An allow list is all-or-nothing: once it is
non-empty every host that doesn't match is refused, so one stray quote silently
blocked every web address. An empty or whitespace-only entry did the same thing
by producing an allow entry of `''`.

The fix strips surrounding quotes and whitespace, strips again after removing a
leading `!`, and drops entries that end up empty.

Discriminates: passes on v0.11.0, fails on v0.10.2 (entries kept verbatim).
"""

import pytest

pytestmark = pytest.mark.regression


def test_quoted_allow_entry_is_unquoted(misc_module):
    """The reported case: Compose hands the entry over with its quotes."""
    allow, block = misc_module.get_allow_block_lists(['"example.com"'])
    assert allow == ["example.com"], (
        "a quoted entry stayed quoted, so it matches no host, and with a non-empty "
        "allow list that blocks every web address (#26908)"
    )
    assert block == []


def test_quoted_block_entry_is_unquoted(misc_module):
    """`!` marks a block entry; the quotes sit outside it."""
    allow, block = misc_module.get_allow_block_lists(["'!evil.com'"])
    assert block == ["evil.com"]
    assert allow == []


def test_quotes_inside_the_bang_are_stripped_too(misc_module):
    """`!"evil.com"`: quoting applied after the marker must strip as well."""
    _, block = misc_module.get_allow_block_lists(['!"evil.com"'])
    assert block == ["evil.com"]


def test_empty_entries_are_dropped_not_turned_into_an_allow_rule(misc_module):
    """A blank entry must not create an allow list out of nothing; that is the
    same total-blackout failure as the quoting bug."""
    allow, block = misc_module.get_allow_block_lists(["", "   ", '""', "!", '!""'])
    assert allow == []
    assert block == []


def test_wellformed_entries_are_unchanged(misc_module):
    """Sanity: the ordinary configuration still parses the same way."""
    allow, block = misc_module.get_allow_block_lists(["example.com", "!evil.com", " spaced.com "])
    assert allow == ["example.com", "spaced.com"]
    assert block == ["evil.com"]


def test_none_and_empty_list_yield_no_rules(misc_module):
    assert misc_module.get_allow_block_lists(None) == ([], [])
    assert misc_module.get_allow_block_lists([]) == ([], [])
