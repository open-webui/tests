"""Regression: the community statistics window trusted any opener and answered to anywhere.

open-webui 0.11.4 PR #29918 (commit `08578557d`): the SyncStatsModal window
that shares chat statistics with the openwebui.com community read
`message` events from whatever page opened it, with no origin check, and
replied with `postMessage(..., '*')`, so any site that tricked a browser into
opening the modal (or opened it as `window.opener` itself) could request a
user's chat statistics export and receive the reply. The fix checks
`event.origin` against the community origin list before reading the message,
replies to `event.origin` instead of `'*'`, and names the community origins as
the explicit targets of the messages the modal sends. The three origins were
also repeated inline in five window message handlers and now come from the
single `COMMUNITY_ORIGINS` constant in `constants.ts`.

The handler is Svelte component code, so the narrow tests read the shipped
component source and pin what the handler does with the origin, following the
frontend source-pinning style of `unit/frontend/test_workspace_permissions.py`.

Discriminates: passes on 344ea5306 (0.11.4), fails on 08578557d^ (the handler
reads `verify:chat` from any origin and posts its reply with a `'*'` target).
"""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.regression

COMMUNITY_ORIGINS = [
    "https://openwebui.com",
    "https://www.openwebui.com",
    "http://localhost:9999",
]


def _frontend_root(open_webui_backend) -> object:
    return open_webui_backend.parent / "src"


def _read_component(open_webui_backend) -> str:
    path = (
        open_webui_backend.parent
        / "src"
        / "lib"
        / "components"
        / "chat"
        / "Settings"
        / "SyncStatsModal.svelte"
    )
    if not path.is_file():
        pytest.skip(f"frontend source file not found: {path}")
    return path.read_text(encoding="utf-8")


def _handler_body(source: str) -> str:
    match = re.search(r"const handleMessage = .+? => \{(?P<body>.*?)\n\t\};", source, re.DOTALL)
    assert match, "couldn't locate the handleMessage body in SyncStatsModal.svelte"
    return match.group("body")


def _post_to_opener_body(source: str) -> str:
    match = re.search(
        r"const postToOpener = \(message: object\) => \{(?P<body>.*?)\n\t\};", source, re.DOTALL
    )
    assert match, "couldn't locate the postToOpener body in SyncStatsModal.svelte"
    return match.group("body")


# --- narrow -------------------------------------------------------------------


def test_the_handler_ignores_messages_from_a_non_community_origin(open_webui_backend):
    """Pre-fix the handler read `verify:chat` from any page that opened the window."""
    body = _handler_body(_read_component(open_webui_backend))

    assert "COMMUNITY_ORIGINS.includes(event.origin)" in body, (
        "the sync stats handler reads verify:chat messages without checking the sender "
        "origin, so any site that opened the window could request a user's chat "
        "statistics export (#29918)"
    )


def test_the_reply_is_addressed_to_the_message_origin_not_the_wildcard(open_webui_backend):
    """Pre-fix both the success and the error reply were posted with a '*' target, handing
    the export to whoever was listening."""
    body = _handler_body(_read_component(open_webui_backend))

    assert "postMessage" in body
    assert "', *')" not in body.replace("event.origin", "").replace(" ", ""), (
        "the verify:chat reply is posted with a wildcard target instead of event.origin, so "
        "any origin listening on the opener received the chat statistics (#29918)"
    )
    assert "event.origin" in body.split("postMessage")[1]


def test_outbound_messages_name_the_community_origins_as_targets(open_webui_backend):
    """The same handler also pushes progress to the opener; those sends must be explicit."""
    body = _post_to_opener_body(_read_component(open_webui_backend))

    assert "COMMUNITY_ORIGINS" in body, (
        "the modal's outbound messages do not name their targets, so they can only be "
        "sent with a wildcard origin (#29918)"
    )


def test_the_origin_list_is_the_shared_constant(open_webui_backend):
    """Broad: the origins must come from the one constant, or the five handlers drift again."""
    body = _handler_body(_read_component(open_webui_backend))
    assert "from '$lib/constants'" in _read_component(open_webui_backend)
    assert "COMMUNITY_ORIGINS" in body


# --- nearby ---------------------------------------------------------------------


def test_the_community_origin_constant_matches_the_community_site(open_webui_backend):
    constants = (
        open_webui_backend.parent / "src" / "lib" / "constants.ts"
    ).read_text(encoding="utf-8")
    match = re.search(r"export const COMMUNITY_ORIGINS = \[(?P<entries>[^\]]*)\]", constants)
    assert match, "COMMUNITY_ORIGINS is gone from constants.ts"
    entries = re.findall(r"'([^']+)'", match.group("entries"))
    assert set(entries) == set(COMMUNITY_ORIGINS), (
        f"the community origin list changed to {entries}; update the pinned list if the "
        "community site moved"
    )


def test_every_window_message_handler_checks_the_shared_origin_list(open_webui_backend):
    """Broad: the other four handlers that read community postMessage traffic must use the
    same gate, since #29918 collected them all behind the constant."""
    root = open_webui_backend.parent / "src"
    offenders = []
    for path in sorted(root.rglob("*.svelte")):
        source = path.read_text(encoding="utf-8")
        if "addEventListener('message'" not in source:
            continue
        relative = path.relative_to(root)
        if relative.as_posix() == "lib/components/chat/Chat.svelte":
            continue  # same-origin chat events, not community traffic
        for handler in re.finditer(
            r"const (?:handleMessage|onMessage|messageHandler) = .+? => \{(?P<body>.*?)\n\t\};",
            source,
            re.DOTALL,
        ):
            body = handler.group("body")
            if "openwebui.com" in body and "COMMUNITY_ORIGINS" not in body:
                offenders.append(relative.as_posix())
    assert not offenders, (
        f"window message handlers still compare origins against an inline openwebui.com "
        f"list instead of COMMUNITY_ORIGINS: {offenders} (#29918)"
    )
