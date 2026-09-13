"""Regression test for 0.11.2's skill-mention parser eating ordinary user text.

Commit `0afe69e1a` (PR #29051, issue #29041). `SKILL_MENTION_RE` and
`SKILL_MENTION_STRIP_RE` in `utils/middleware.py` accepted `[^|>]+` as the skill id, so any
`<$...>` run in the user's own text (shell variables, prices, inline math) parsed as a mention
and `strip_skill_mentions` silently deleted it from the message before it reached the model.
Both patterns now require `[a-z0-9_-]+`, the id shape skills are validated against on create.
Commit `0edd731c7` additionally requires resolved skill IDs and preserves whitespace.

Discriminates: passes on dev `05484aa05`; widening the ID patterns extracts ordinary text
as mentions. Removing resolved-ID filtering or restoring whitespace stripping also fails.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.regression


# Shapes from the issue: none of these is a skill mention, all were being deleted.
NOT_MENTIONS = [
    'echo "<$PATH>" >> log',
    '```bash\nprintf "<$HOME>"\n```',
    'if a <$x + y$> b then stop',
    'Total is <$5.99> per unit',
    'set <$user name> here',
    'compare a <$b$ and c> d',
    'open </home/user|the file> now',
]


@pytest.fixture(scope='session')
def middleware_module(owui_module):
    """`open_webui.utils.middleware` (SKILL_MENTION_RE, strip_skill_mentions)."""
    return owui_module('open_webui.utils.middleware')


def _user_message(text: str) -> dict:
    return {'role': 'user', 'content': text}


def _user_message_parts(text: str) -> dict:
    return {'role': 'user', 'content': [{'type': 'text', 'text': text}]}


def _strip_mentions(module, messages, skill_ids=None):
    if 'skill_ids' in inspect.signature(module.strip_skill_mentions).parameters:
        module.strip_skill_mentions(messages, {'my_skill'} if skill_ids is None else skill_ids)
    else:
        module.strip_skill_mentions(messages)


# -----------------------------------------------------------------------------
# Narrow: the user's own text survives byte for byte
# -----------------------------------------------------------------------------


@pytest.mark.parametrize('text', NOT_MENTIONS)
def test_text_that_looks_like_a_mention_is_kept_intact(middleware_module, text):
    messages = [_user_message(text)]
    _strip_mentions(middleware_module, messages)
    assert messages[0]['content'] == text


@pytest.mark.parametrize('text', NOT_MENTIONS)
def test_text_parts_that_look_like_a_mention_are_kept_intact(middleware_module, text):
    messages = [_user_message_parts(text)]
    _strip_mentions(middleware_module, messages)
    assert messages[0]['content'][0]['text'] == text


@pytest.mark.parametrize('text', NOT_MENTIONS)
def test_no_skill_id_is_extracted_from_ordinary_text(middleware_module, text):
    assert middleware_module.extract_skill_ids_from_messages([_user_message(text)]) == set()


def test_real_mention_is_stripped_without_touching_neighbouring_text(middleware_module):
    messages = [_user_message('<$my_skill|My Skill> run echo "<$PATH>" for me')]
    _strip_mentions(middleware_module, messages)
    assert messages[0]['content'] == 'My Skill run echo "<$PATH>" for me'


# -----------------------------------------------------------------------------
# Broad: real mentions are still recognised, so the fix did not disable the feature
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    'text, skill_id',
    [
        ('<$my_skill|My Skill> summarise this', 'my_skill'),
        ('</my_skill|My Skill> summarise this', 'my_skill'),
        ('<$skill-2|My Skill> summarise this', 'skill-2'),
        ('<$my_skill> summarise this', 'my_skill'),
    ],
)
def test_real_mention_is_extracted(middleware_module, text, skill_id):
    assert middleware_module.extract_skill_ids_from_messages([_user_message(text)]) == {skill_id}


@pytest.mark.parametrize(
    'text, expected',
    [
        ('<$my_skill|My Skill> summarise this', 'My Skill summarise this'),
        ('</my_skill|My Skill> summarise this', 'My Skill summarise this'),
        ('<$my_skill> summarise this', ' summarise this'),
    ],
)
def test_real_mention_is_replaced_by_its_label(middleware_module, text, expected):
    messages = [_user_message(text)]
    _strip_mentions(middleware_module, messages)
    if 'skill_ids' not in inspect.signature(middleware_module.strip_skill_mentions).parameters:
        expected = expected.strip()
    assert messages[0]['content'] == expected


def test_real_mention_is_extracted_from_text_parts(middleware_module):
    messages = [_user_message_parts('<$my_skill|My Skill> go')]
    assert middleware_module.extract_skill_ids_from_messages(messages) == {'my_skill'}
    _strip_mentions(middleware_module, messages)
    assert messages[0]['content'][0]['text'] == 'My Skill go'


def test_mentions_across_several_messages_are_all_extracted(middleware_module):
    messages = [
        _user_message('<$alpha|Alpha> first'),
        {'role': 'assistant', 'content': 'ok'},
        _user_message('</beta|Beta> second'),
    ]
    assert middleware_module.extract_skill_ids_from_messages(messages) == {'alpha', 'beta'}


# -----------------------------------------------------------------------------
# Nearby: unchanged on both refs
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    'text',
    ['just a plain question', '', 'a < b and c > d', 'price is $5.99', 'x <y> z'],
)
def test_plain_text_is_untouched(middleware_module, text):
    messages = [_user_message(text)]
    _strip_mentions(middleware_module, messages)
    assert messages[0]['content'] == text
    assert middleware_module.extract_skill_ids_from_messages(messages) == set()


def test_message_that_is_only_a_mention_collapses_to_its_label(middleware_module):
    messages = [_user_message('<$my_skill|My Skill>')]
    assert middleware_module.extract_skill_ids_from_messages(messages) == {'my_skill'}
    _strip_mentions(middleware_module, messages)
    assert messages[0]['content'] == 'My Skill'


def test_message_without_content_is_left_alone(middleware_module):
    messages = [{'role': 'user'}, {'role': 'user', 'content': None}]
    _strip_mentions(middleware_module, messages)
    assert messages == [{'role': 'user'}, {'role': 'user', 'content': None}]
    assert middleware_module.extract_skill_ids_from_messages(messages) == set()


@pytest.mark.parametrize('message_factory', [_user_message, _user_message_parts])
@pytest.mark.parametrize('skill_ids', [set(), {'my_skill'}])
def test_unresolved_mentions_and_whitespace_survive(middleware_module, message_factory, skill_ids):
    if 'skill_ids' not in inspect.signature(middleware_module.strip_skill_mentions).parameters:
        pytest.skip('resolved-skill filtering was added in 0edd731c7')
    text = '  <$my_skill|My Skill> while (<$fh>) { </unknown|Unknown> }  '
    messages = [message_factory(text)]
    _strip_mentions(middleware_module, messages, skill_ids)
    expected = text.replace('<$my_skill|My Skill>', 'My Skill') if skill_ids else text
    assert messages == [message_factory(expected)]
