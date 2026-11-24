"""Tell a sub-agent's own requests from the report that brings its result back to the lead.

Since dev 8522af2d6 every delegation from a saved chat runs on after the lead's reply has ended.
The finished sub-agent's report reaches the lead model as a user message of its own that names
the task, so a match on the task alone takes that report as well. `delegated(task)` matches only
the sub-agent's requests, `report_of(task)` only the lead's request carrying the report, and
`wait_for_report(upstream, task)` waits until that report reached the model and returns it.
"""

from __future__ import annotations

import time
from typing import Callable

from harness import upstream as reply
from harness.upstream import MockUpstream

REPORT_MARK = "[ASYNC SUBAGENT COMPLETE"


def _latest_user_message(body: dict) -> str:
    users = [entry for entry in body.get("messages", []) if entry.get("role") == "user"]
    return str(users[-1].get("content")) if users else ""


def report_of(task: str) -> Callable[[dict], bool]:
    """A `match` for the lead's request that carries the finished sub-agent's report."""

    def matches(body: dict) -> bool:
        latest = _latest_user_message(body)
        return REPORT_MARK in latest and task in latest

    return matches


def delegated(task: str) -> Callable[[dict], bool]:
    """A `match` for the requests of the sub-agent working on `task`, never the lead's report."""
    working_on = reply.answering(task)
    return lambda body: working_on(body) and REPORT_MARK not in _latest_user_message(body)


def report_content(body: dict) -> str:
    """The report as the lead model got it."""
    return _latest_user_message(body)


def wait_for_report(upstream: MockUpstream, task: str, timeout: float = 60.0) -> dict:
    """The lead's request carrying the report of `task`, once the sub-agent finished."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        reports = [body for body in upstream.chat_requests() if report_of(task)(body)]
        if reports:
            return reports[0]
        time.sleep(0.1)
    raise AssertionError(f"the report of the sub-agent on {task!r} never reached the lead model")
