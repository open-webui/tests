"""Text in a user message that looks like a skill mention reaches the model unchanged.

0.11.2 `0afe69e1a` (PR #29051, issue #29041) narrowed the skill-mention patterns to the id shape
skills are created with, because `[^|>]+` turned shell variables, prices and inline math such
as `<$PATH>` into mentions that `strip_skill_mentions` deleted before the model saw them.
`0edd731c7` then strips only mentions of skills that actually resolved, which keeps lowercase
look-alikes such as Perl's `<$fh>`, and stops trimming the message's own whitespace. A resolved
mention still collapses to its label.

Twin of unit/chat/test_skill_mention_text_preservation.py.

Discriminates: passes on dev bbfa876af; reverting `0edd731c7` (strip every match, trim the
message) fails the `<$fh>` and whitespace cases, and reverting `0afe69e1a` on top fails all seven.
"""

from __future__ import annotations

import uuid

import pytest

from harness.chat import ask

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

NOT_MENTIONS = [
    'echo "<$PATH>" >> log',
    "Total is <$5.99> per unit",
    "set <$user name> here",
    "open </home/user|the file> now",
    "while (<$fh>) { print }",
]


@pytest.fixture
def skill_id(admin):
    """A skill the admin can mention, removed again afterwards."""
    new_id = f"skill_{uuid.uuid4().hex[:8]}"
    form = {"id": new_id, "name": "My Skill", "content": "Answer in haiku.", "meta": {}}
    with admin.client() as client:
        created = client.post("/api/v1/skills/create", json=form)
        assert created.status_code == 200, created.text
    yield new_id
    with admin.client() as client:
        client.delete(f"/api/v1/skills/id/{new_id}/delete")


def _sent_to_the_model(admin, upstream, content: str) -> str:
    with admin.client() as client:
        ask(client, content)
    return upstream.chat_requests()[-1]["messages"][-1]["content"]


@pytest.mark.parametrize("text", NOT_MENTIONS)
def test_text_that_looks_like_a_mention_reaches_the_model_unchanged(
    admin, upstream, skill_id, text
):
    assert _sent_to_the_model(admin, upstream, text) == text, (
        "part of the user's own text was stripped as a skill mention (#29041)"
    )


def test_a_real_mention_becomes_its_label_and_leaves_the_rest_alone(admin, upstream, skill_id):
    sent = _sent_to_the_model(admin, upstream, f'<${skill_id}|My Skill> run echo "<$PATH>"')

    assert sent == 'My Skill run echo "<$PATH>"'


def test_unresolved_mentions_and_surrounding_whitespace_survive(admin, upstream, skill_id):
    text = f"  <${skill_id}|My Skill> while (<$fh>) {{ </unknown|Unknown> }}  "

    sent = _sent_to_the_model(admin, upstream, text)

    assert sent == "  My Skill while (<$fh>) { </unknown|Unknown> }  "
