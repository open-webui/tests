"""Text typed into the chat input that looks like a skill mention is sent and shown as typed.

0.11.2 `0afe69e1a` (PR #29051, issue #29041) and `0edd731c7`: `<$...>` runs in the user's own
text (shell variables, Perl filehandles) parsed as skill mentions and were deleted before the
model saw them. `0edd731c7` also keeps an unresolved `$` mention as raw text when the chat
input and the message renderer meet one.

Twin of unit/chat/test_skill_mention_text_preservation.py.

Discriminates: passes on dev bbfa876af; reverting both fixes in the backend fails both cases (the
model is sent the text with the look-alike stripped), and a build with the `0edd731c7` renderer
change reverted fails both (the message shows a mention chip where `<$HOME>` was typed), while an
unrelated chat journey passes on that build.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from utils.chat_ui import conversation, expect_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


# No quotes or `>>`: the input's typography turns those into curly quotes and guillemets.
@pytest.mark.parametrize("typed", ["print <$HOME> twice", "while (<$fh>) { print }"])
def test_typed_look_alike_mentions_are_sent_and_shown_verbatim(
    page_for, make_user, upstream, typed
):
    page = page_for(make_user())
    upstream.queue(reply.text("noted"))

    send(page, typed)
    expect_reply(page, "noted")

    assert upstream.chat_requests()[-1]["messages"][-1]["content"] == typed
    user_message = conversation(page).locator(".chat-user").last
    expect(user_message).to_contain_text(typed)
