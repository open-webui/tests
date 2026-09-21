"""Regression: deleting a channel message left its quote on every reply.

open-webui 0.11.4 PR #30314 / issue #30313 (commit `b988f06ce`): the channel
and thread components handled `message:delete` by filtering the deleted row
out of the message list and nothing else, so the quote of the deleted message
stayed above every reply, and a reply already being composed against it kept
its reply chip. Clicking the dead quote did nothing, and sending the composed
reply was refused by the server (its target no longer exists) while the
message stayed on screen until a reload.

The fix is frontend-only (Channel.svelte and Thread.svelte); there is no
backend commit in the PR. The backend contract the clearing relies on is that
no read surface hands out a deleted message's content, and that is pinned
behaviourally against a real scratch database in
`unit/chat/test_deleted_message_quotes.py`. This file pins the frontend
clearing itself, reading the shipped component sources, following the
frontend source-pinning style of `unit/frontend/test_workspace_permissions.py`.

Discriminates: passes on 344ea5306 (0.11.4), fails on b988f06ce^ (the
`message:delete` branch filters the row out but never clears the quotes, and
the reply composer keeps the deleted target).
"""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.regression

COMPONENTS = [
    "lib/components/channel/Channel.svelte",
    "lib/components/channel/Thread.svelte",
]


def _component_source(open_webui_backend, relative: str) -> str:
    path = open_webui_backend.parent / "src" / relative
    if not path.is_file():
        pytest.skip(f"frontend source file not found: {path}")
    return path.read_text(encoding="utf-8")


def _delete_branch(source: str) -> str:
    """The `message:delete` handling block inside the channel-events handler."""
    match = re.search(
        r"type === 'message:delete'\) \{(?P<body>.*?)\n\t+\} else", source, re.DOTALL
    )
    assert match, "couldn't locate the message:delete branch in the component"
    return match.group("body")


# --- narrow -------------------------------------------------------------------


@pytest.mark.parametrize("component", COMPONENTS)
def test_the_delete_branch_clears_the_quote_of_the_deleted_message(
    open_webui_backend, component
):
    """Pre-fix the branch filtered the deleted row out and left every reply still quoting it."""
    body = _delete_branch(_component_source(open_webui_backend, component))

    assert "reply_to_message" in body, (
        f"{component} removed the deleted message but left the quote of it sitting on every "
        "reply, with the deleted text and author, doing nothing when clicked (#30313)"
    )


@pytest.mark.parametrize("component", COMPONENTS)
def test_the_delete_branch_clears_the_reply_being_composed_against_it(
    open_webui_backend, component
):
    """Pre-fix the reply chip in the composer kept the deleted target, so sending failed."""
    body = _delete_branch(_component_source(open_webui_backend, component))

    assert "replyToMessage" in body, (
        f"{component} left the reply composer pointing at the deleted message, so a reply "
        "already being composed was still sent addressed to it and was refused (#30313)"
    )


# --- broad ----------------------------------------------------------------------


@pytest.mark.parametrize("component", COMPONENTS)
def test_the_deleted_row_is_still_removed(open_webui_backend, component):
    """The clearing must extend the filtering, not replace it."""
    body = _delete_branch(_component_source(open_webui_backend, component))

    assert "filter" in body and "data.id" in body, (
        f"{component} stopped removing the deleted row itself, so the fix over-corrected (#30314)"
    )


def test_the_thread_panel_shares_the_clearing(open_webui_backend):
    """Broad: both surfaces that show channel messages must clear, or the thread panel
    keeps the quotes the channel dropped."""
    for component in COMPONENTS:
        body = _delete_branch(_component_source(open_webui_backend, component))
        assert "reply_to_message: null" in body, (
            f"{component} does not null the quote reference on the surviving messages (#30314)"
        )
