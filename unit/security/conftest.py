"""Shared helpers for the security unit tests."""

import json
from types import SimpleNamespace


class FakeWebSocket:
    """Minimal stand-in for the starlette WebSocket the terminal route drives:
    replays the first-message auth handshake and records the close."""

    def __init__(self, token: str = "jwt"):
        self._first_message = json.dumps({"type": "auth", "token": token})
        self.app = SimpleNamespace(state=SimpleNamespace(redis=None))
        self.close_code = None
        self.close_reason = None

    async def receive_text(self) -> str:
        return self._first_message

    async def close(self, code=None, reason=None):
        self.close_code = code
        self.close_reason = reason
