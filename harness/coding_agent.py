"""A real Open Terminal with a stand-in coding agent installed, for terminal agent models.

`serving_agent_terminal()` runs `harness.open_terminal` with the terminal's OpenCode command
pointed at `harness/fake_acp_agent.py`, so the terminal discovers it as the `opencode` agent
with two models and runs it for each chat turn over ACP. Every other agent the terminal knows
points at a path that does not exist, so none installed on the machine running the suite
shows up. `AgentTerminal.log()` reads what the agent processes received.

`serving_agent_terminal(native=True)` installs `harness/fake_native_agent.py` instead, as the
terminal's Claude Code, Codex and Pi: the three agents that speak a protocol of their own.
`messages(agent)` and `started(agent)` read what each of them received and was started with.

`install_agent(client, terminal_id)` puts the same stand-in into a machine of a terminal on the
Terminals orchestrator, where the agent has to be inside the container: as `opencode` on the
machine user's own path, through Open WebUI's proxy.

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

import httpx

from harness.open_terminal import OpenTerminal, serving_open_terminal
from harness.terminal_server import configure_terminals, read_grant

PROVIDER = "opencode"
FAST, DEEP = "fake-fast", "fake-deep"
AGENT_SCRIPT = Path(__file__).with_name("fake_acp_agent.py")
NATIVE_AGENT_SCRIPT = Path(__file__).with_name("fake_native_agent.py")
ACP_COMMANDS = ("OPEN_TERMINAL_OPENCODE_COMMAND",)
NATIVE_COMMANDS = (
    "OPEN_TERMINAL_CLAUDE_COMMAND",
    "OPEN_TERMINAL_CODEX_COMMAND",
    "OPEN_TERMINAL_PI_COMMAND",
)
OTHER_AGENT_COMMANDS = (
    "OPEN_TERMINAL_CURSOR_COMMAND",
    "OPEN_TERMINAL_GROK_COMMAND",
    "OPEN_TERMINAL_CLINE_COMMAND",
    "OPEN_TERMINAL_GEMINI_COMMAND",
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

    def messages(self, agent: str) -> list[dict]:
        """Every message the processes of `agent` (`claude-code`, `codex`, `pi`) got."""
        return [
            entry["received"]
            for entry in self.log()
            if entry.get("agent") == agent and "received" in entry
        ]

    def conversations(self, agent: str) -> list[str]:
        """The agent's own conversation each of its turns ran in, oldest first."""
        return [
            entry["conversation"]
            for entry in self.log()
            if entry.get("agent") == agent and "conversation" in entry
        ]

    def started(self, agent: str) -> list[list[str]]:
        """The arguments each process of `agent` was started with, discovery runs included."""
        return [
            entry["started"]
            for entry in self.log()
            if entry.get("agent") == agent and "started" in entry
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


# where a machine's agent keeps its log; give it to the template as its `env`
MACHINE_AGENT_LOG = "agent-log.jsonl"
MACHINE_AGENT_ENV = {"FAKE_AGENT_LOG": f"/home/user/{MACHINE_AGENT_LOG}"}


def install_agent(client: httpx.Client, terminal_id: str, **context) -> None:
    """Install the stand-in agent as `opencode` in the caller's machine of an orchestrator
    terminal; `context` is `chat_id` or `model_id` for a machine per chat or per model."""
    from harness.terminals import execute, output_of

    script = "#!/usr/bin/env python3\n" + AGENT_SCRIPT.read_text()
    install = (
        "mkdir -p ~/.local/bin && cat > ~/.local/bin/opencode <<'FAKE_AGENT'\n"
        f"{script}\nFAKE_AGENT\n"
        "chmod +x ~/.local/bin/opencode && echo installed"
    )
    printed = output_of(execute(client, terminal_id, install, **context))
    assert printed.strip() == "installed", f"installing the agent failed: {printed}"


@contextlib.contextmanager
def serving_agent_terminal(native: bool = False) -> Iterator[AgentTerminal]:
    scratch = Path(tempfile.mkdtemp(prefix="fake-agent-"))
    command = scratch / "agent"
    script = NATIVE_AGENT_SCRIPT if native else AGENT_SCRIPT
    command.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    command.chmod(command.stat().st_mode | stat.S_IXUSR)
    log_path = scratch / "agent.jsonl"
    installed, missing = (
        (NATIVE_COMMANDS, ACP_COMMANDS) if native else (ACP_COMMANDS, NATIVE_COMMANDS)
    )
    env = {name: str(scratch / "not-installed") for name in (*OTHER_AGENT_COMMANDS, *missing)}
    env.update({name: str(command) for name in installed}, FAKE_AGENT_LOG=str(log_path))
    try:
        with serving_open_terminal(env=env) as terminal:
            yield AgentTerminal(terminal=terminal, log_path=log_path)
    finally:
        for path in scratch.iterdir():
            path.unlink()
        scratch.rmdir()
