"""A `:::writing{...}` block in a reply is titled from its braced attributes, and only from those.

Models wrap a drafted email as `:::writing{variant="email" subject="..." recipient="..."}`; the
chat shows the subject and recipient as the block's title (eff5c4a2d, PR #28280). Attributes are
read from the braces on the opening line alone: prose after the fence name, text after the
closing brace or an unclosed brace must not put a recipient or subject on the title, or a reply
could pass off an address the model only mentioned as the recipient of the draft.

Browser twin of frontend/marked/colon-fence-extension.test.ts, which drives the tokenizer.

Discriminates: passes on the bbfa876af build; with the tokenizer reading attributes from the
whole opening line, the smuggled recipients and subject appear in the block titles.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from utils.chat_ui import expect_reply, last_reply, send

pytestmark = [pytest.mark.requires_browser, pytest.mark.requires_source]

FENCED_REPLY = """Here are the drafts.

:::writing{variant="email" subject="Short question" recipient="mail@example.com"}
The email body.
:::

:::note prose mentioning recipient="evil@example.com"
The note body.
:::

:::writing{recipient="stray@example.com"
The unclosed body.
:::

:::writing{variant="document"} draft subject="junk"
The trailing body.
:::

:::writing{subject="use {x} here"}
The braces body.
:::
"""


@pytest.fixture
def fenced_reply(page_for, make_user, upstream):
    page = page_for(make_user())
    upstream.queue(reply.text(FENCED_REPLY))
    send(page, "draft it for me")
    expect_reply(page, "The braces body.")
    return last_reply(page)


def test_a_block_is_titled_from_its_braced_attributes(fenced_reply):
    expect(fenced_reply.get_by_title("Short question · mail@example.com")).to_be_visible()
    expect(fenced_reply.get_by_text("The email body.")).to_be_visible()
    expect(fenced_reply.get_by_title("use {x} here", exact=True)).to_be_visible()


def test_attributes_outside_closed_braces_reach_no_title(fenced_reply):
    expect(fenced_reply.get_by_title("Note", exact=True)).to_be_visible()
    expect(fenced_reply.get_by_title("Writing", exact=True)).to_have_count(2)
    for smuggled in ("evil@example.com", "stray@example.com", "junk"):
        expect(fenced_reply).not_to_contain_text(smuggled)
