"""A real Open Terminal with a stand-in coding agent installed, for terminal agent models.

`serving_agent_terminal()` runs `harness.open_terminal` with the terminal's OpenCode command
pointed at `harness/fake_acp_agent.py`, so the terminal discovers it as the `opencode` agent
with two models and runs it for each chat turn over ACP. Every other agent the terminal knows
points at a path that does not exist, so none installed on the machine running the suite
shows up. `AgentTerminal.log()` reads what the agent processes received.

Open WebUI lists the agent's models in `/api/models` only when asked for that terminal
(`terminal_id`), under ids built by `model_id(...)`.
"""

from __future__ import annotations

import contextlib
import json
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

from harness.open_terminal import OpenTerminal, serving_open_terminal
from harness.terminal_server import configure_terminals, read_grant

PROVIDER = "opencode"
FAST, DEEP = "fake-fast", "fake-deep"
AGENT_SCRIPT = Path(__file__).with_name("fake_acp_agent.py")
OTHER_AGENT_COMMANDS = (
    "OPEN_TERMINAL_CLAUDE_COMMAND",
    "OPEN_TERMINAL_CODEX_COMMAND",
    "OPEN_TERMINAL_CURSOR_COMMAND",
    "OPEN_TERMINAL_GROK_COMMAND",
    "OPEN_TERMINAL_CLINE_COMMAND",
    "OPEN_TERMINAL_GEMINI_COMMAND",
    "OPEN_TERMINAL_PI_COMMAND",
)


def model_id(terminal_id: str, model: str = FAST, provider: str = PROVIDER) -> str:
    """The id Open WebUI gives a terminal agent's model."""
    return f"agent:{quote(terminal_id, safe='')}:{provider}:{quote(model, safe='')}"


@dataclass
class AgentTerminal:
    terminal: OpenTerminal
    log_path: Path

    def connection(self, **fields) -> dict:
        return self.terminal.connection(**fields)

    def log(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        return [json.loads(line) for line in self.log_path.read_text().splitlines() if line]

    def received(self, method: str) -> list[dict]:
        """The params of every `method` request the agent processes got, oldest first."""
        return [
            entry["received"].get("params") or {}
            for entry in self.log()
            if (entry.get("received") or {}).get("method") == method
        ]


def connect_agent_terminal(admin, terminal: AgentTerminal, *people, **fields) -> dict:
    """Save a connection to `terminal` that `people` can use, replacing the current ones.

    Preserve `TERMINAL_SERVERS_CONFIG` first.
    """
    connection = terminal.connection(**fields)
    connection["config"]["access_grants"] = [read_grant(person.id) for person in people]
    with admin.client() as client:
        configure_terminals(client, connection)
    return connection


@contextlib.contextmanager
def serving_agent_terminal() -> Iterator[AgentTerminal]:
    scratch = Path(tempfile.mkdtemp(prefix="fake-agent-"))
    command = scratch / "opencode"
    command.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{AGENT_SCRIPT}" "$@"\n')
    command.chmod(command.stat().st_mode | stat.S_IXUSR)
    log_path = scratch / "agent.jsonl"
    env = {name: str(scratch / "not-installed") for name in OTHER_AGENT_COMMANDS}
    env.update(OPEN_TERMINAL_OPENCODE_COMMAND=str(command), FAKE_AGENT_LOG=str(log_path))
    try:
        with serving_open_terminal(env=env) as terminal:
            yield AgentTerminal(terminal=terminal, log_path=log_path)
    finally:
        for path in scratch.iterdir():
            path.unlink()
        scratch.rmdir()
