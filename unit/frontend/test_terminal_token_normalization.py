"""Regression: a stray space in a terminal API key closed the interactive terminal.

open-webui issue #25613, fixed by 58f917031 (PR #25686). The key a user types for an Open
Terminal server can carry trailing whitespace. HTTP calls send it in the Authorization header,
where the space is dropped, so file listing and tool calls worked; the terminal WebSocket sends it
inside a JSON `auth` message that the server compares exactly, so the terminal showed
"[Connection closed]". The fix trims the key in the WebSocket auth message (and on save).

Stays a unit audit: the key only reaches the WebSocket from a browser, and getting there takes a
user terminal saved in Settings, picked for a chat, the files panel and then the terminal dock
opened, all against a server answering the config, file and session calls. That is a long UI
journey for a one-call trim.

Discriminates: passes on bbfa876af; fails with `.trim()` dropped from the WebSocket auth token in
XTerminal.svelte (narrow), or from the Bearer header of `terminalRequest` (broad).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

WEBSOCKET_CLIENT = "lib/components/chat/XTerminal.svelte"
TERMINAL_API = "lib/apis/terminal/index.ts"


def frontend_source(open_webui_backend: Path, relative: str) -> str:
    path = open_webui_backend.parent / "src" / relative
    assert path.is_file(), f"{relative} is gone; retarget this audit at the terminal client"
    return path.read_text(encoding="utf-8")


def is_trimmed(expression: str) -> bool:
    return re.search(r"\.trim\(\s*\)", expression) is not None


def enclosing_object(source: str, position: int) -> str:
    """The `{...}` literal around `position`, found by balancing braces outwards."""
    start, depth = position, 0
    while depth >= 0:
        start -= 1
        depth += {"}": 1, "{": -1}.get(source[start], 0)
    end, depth = position, 0
    while depth >= 0:
        depth += {"{": 1, "}": -1}.get(source[end], 0)
        end += 1
    return source[start:end]


def test_the_websocket_auth_message_sends_the_key_trimmed(open_webui_backend):
    source = frontend_source(open_webui_backend, WEBSOCKET_CLIENT)
    auth_messages = [
        enclosing_object(source, marker.start())
        for marker in re.finditer(r"\btype:\s*['\"]auth['\"]", source)
    ]
    assert auth_messages, f"no `type: 'auth'` message in {WEBSOCKET_CLIENT}; retarget this audit"

    tokens = [re.search(r"\btoken:\s*([^,}]+)", message).group(1) for message in auth_messages]
    assert all(is_trimmed(token) for token in tokens), (
        f"the terminal WebSocket authenticates with an untrimmed key ({tokens}), so a key "
        "saved with a trailing space closes the terminal (#25613)"
    )


def test_every_bearer_header_to_a_terminal_trims_the_key(open_webui_backend):
    """Only Open WebUI's own session token, sent to its own API, may go untrimmed."""
    source = frontend_source(open_webui_backend, TERMINAL_API)
    declarations = re.split(r"\n(?=(?:export )?const )", source)
    with_bearer = [block for block in declarations if "Bearer ${" in block]
    assert with_bearer, f"no Bearer header in {TERMINAL_API}; retarget this audit"

    untrimmed = [
        block.split("=", 1)[0].strip()
        for block in with_bearer
        if not all(is_trimmed(value) for value in re.findall(r"Bearer \$\{([^}]*)\}", block))
        and "WEBUI_API_BASE_URL" not in block
    ]
    assert not untrimmed, f"these send a terminal key untrimmed: {untrimmed} (#25613)"
