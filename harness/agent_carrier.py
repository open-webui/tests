"""A scripted coding agent on the fake terminal, for what a real terminal cannot be made to do.

`serving_agents(terminal)` teaches a `harness.terminal_server` fake the terminal's side of coding
agents: the catalog (`GET /agents`), the working directory (`/files/cwd`, `agents.folder` until a
chat moves its own), a session per chat
turn (`POST /agents/{provider}/sessions`) and that session's socket, which carries the agent's
JSON lines in numbered frames with acknowledgements and a replay after a reconnect, as Open
Terminal does. Behind the socket sits a small ACP agent. The test decides the catalog, what a
turn does (`agents.turn = lambda run, params: ...`) and how the carrier misbehaves: `run.drop()`
hangs up mid-turn, `run.skip_a_frame()` loses a piece of output, `run.ask(method, params)` sends
a request of the agent's own and returns Open WebUI's answer, `agents.swallow` leaves an input
unacknowledged and hangs up, `agents.refuse_start` answers the session start with an error.

A test reads `agents.starts` (each session start with its headers and options), `run.connections`
(the sign-in of every socket Open WebUI opened on the session, with its `cursor`) and
`run.inputs` (every input frame as it arrived, repeats included).
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import parse_qs, urlsplit

from harness.listener import json_answer
from harness.terminal_server import (
    FakeTerminalServer,
    SessionEnded,
    TerminalRequest,
    TerminalSession,
)

PROVIDER = "opencode"
MODEL = "scripted-one"
FOLDER = "/home/user/work"
SESSION = "scripted-session"


def catalog(*agents: dict, protocol_version: int = 1) -> dict:
    return {"protocol_version": protocol_version, "agents": list(agents)}


def agent(provider: str = PROVIDER, *models: dict, available: bool = True, **fields) -> dict:
    """One agent of the catalog; without `models` it has the one scripted model."""
    return {
        "id": provider,
        "name": provider.title(),
        "models": list(models) or [{"id": MODEL, "name": "Scripted One"}],
        "available": available,
        **fields,
    }


@dataclass
class AgentRun:
    """One session start: the agent process of one chat turn and the sockets opened on it."""

    id: str
    provider: str
    options: dict
    headers: dict[str, str]
    agents: ScriptedAgents
    connections: list[dict] = field(default_factory=list)
    inputs: list[dict] = field(default_factory=list)
    received: list[dict] = field(default_factory=list)
    outputs: list[str | None] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)
    socket: TerminalSession | None = None
    handled: set[int] = field(default_factory=set)
    acknowledged: int = 0
    backlog: list[dict] = field(default_factory=list)
    deleted: bool = False

    def requests(self, method: str) -> list[dict]:
        """The params of every `method` request the agent got on this session."""
        return [m.get("params") or {} for m in self.received if m.get("method") == method]

    def emit(self, message: dict) -> None:
        """One JSON line from the agent: numbered, kept for a replay, sent when a socket is on."""
        with self.lock:
            self.outputs.append(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
            self._send_output(len(self.outputs))

    def say(self, text: str, thought: bool = False) -> None:
        kind = "agent_thought_chunk" if thought else "agent_message_chunk"
        update = {"sessionUpdate": kind, "content": {"type": "text", "text": text}}
        self.emit({"method": "session/update", "params": {"sessionId": SESSION, "update": update}})

    def wait_until_read(self, timeout: float = 10) -> None:
        """Wait until Open WebUI has acknowledged every piece of output sent so far."""
        while self.acknowledged < len(self.outputs):
            self._next_frame(timeout, keep_inputs=True)

    def ask(self, method: str, params: dict, timeout: float = 30) -> dict:
        """Send Open WebUI a request of the agent's own and wait for its answer."""
        request_id = 9000 + len(self.outputs)
        self.emit({"id": request_id, "method": method, "params": params})
        while True:
            frame = self._next_frame(timeout)
            if frame is None:
                continue
            self.inputs.append(frame)
            message = json.loads(frame["data"])
            self.socket.send(json.dumps({"type": "ack", "sequence": frame["sequence"]}))
            self.handled.add(frame["sequence"])
            self.received.append(message)
            if message.get("id") == request_id and "method" not in message:
                return message

    def skip_a_frame(self) -> None:
        """Lose one piece of output: its number is used up and it is never sent."""
        with self.lock:
            self.outputs.append(None)

    def drop(self) -> None:
        """Hang up the socket without a close frame; the agent itself runs on."""
        with self.lock:
            live, self.socket = self.socket, None
        if live is not None:
            live.drop()

    def _send_output(self, sequence: int) -> None:
        data = self.outputs[sequence - 1]
        if self.socket is None or data is None:
            return
        try:
            self.socket.send(json.dumps({"type": "output", "sequence": sequence, "data": data}))
        except Exception:
            self.socket = None

    def _attach(self, socket: TerminalSession) -> None:
        sign_in = socket.authenticated()
        with self.lock:
            self.connections.append(sign_in)
            self.socket = socket
            for sequence in range(int(sign_in.get("cursor") or 0) + 1, len(self.outputs) + 1):
                self._send_output(sequence)

    def _next_frame(self, timeout: float, keep_inputs: bool = False) -> dict | None:
        """The next input frame; acknowledgements on the way are counted."""
        if self.backlog and not keep_inputs:
            return self.backlog.pop(0)
        frame = json.loads(self.socket.receive(timeout))
        if frame.get("type") == "ack":
            self.acknowledged = max(self.acknowledged, frame["sequence"])
            return None
        if frame.get("type") != "input":
            return None
        if keep_inputs:
            self.backlog.append(frame)
            return None
        return frame

    def serve(self, socket: TerminalSession) -> None:
        self._attach(socket)
        while self.socket is socket:
            try:
                frame = self._next_frame(timeout=90)
            except (SessionEnded, TimeoutError):
                return
            if frame is None:
                continue
            self.inputs.append(frame)
            message = json.loads(frame["data"])
            swallow = self.agents.swallow
            if swallow and swallow(message):
                self.agents.swallow = None
                self.drop()
                return
            with self.lock:
                if self.socket is socket:
                    socket.send(json.dumps({"type": "ack", "sequence": frame["sequence"]}))
            if frame["sequence"] in self.handled:
                continue
            self.handled.add(frame["sequence"])
            self.received.append(message)
            self._answer(message)

    def _answer(self, message: dict) -> None:
        method, params, request_id = (
            message.get("method"),
            message.get("params") or {},
            message.get("id"),
        )
        if request_id is None or method is None:
            return
        if method == "initialize":
            result = {
                "protocolVersion": 1,
                "agentCapabilities": {"loadSession": True, "promptCapabilities": {"image": True}},
                "authMethods": [],
            }
        elif method in ("session/new", "session/load"):
            models = [{"modelId": self.options.get("model") or MODEL, "name": "Scripted"}]
            result = {"sessionId": SESSION, "models": {"availableModels": models}}
        elif method == "session/prompt":
            self.agents.turn(self, params)
            result = {"stopReason": "end_turn"}
        else:
            result = {}
        self.emit({"id": request_id, "result": result})


def heard(run: AgentRun, params: dict) -> None:
    """The plain turn: the agent says what it was asked."""
    texts = [block.get("text", "") for block in params.get("prompt") or []]
    run.say("Scripted agent heard: " + texts[-1])


@dataclass
class ScriptedAgents:
    terminal: FakeTerminalServer
    catalog: dict | Callable[[str | None], dict] = field(default_factory=lambda: catalog(agent()))
    folder: str | None = FOLDER
    turn: Callable[[AgentRun, dict], None] = heard
    swallow: Callable[[dict], bool] | None = None
    refuse_start: tuple[int, dict] | None = None
    starts: list[AgentRun] = field(default_factory=list)
    moved: dict[str | None, str] = field(default_factory=dict)

    def catalog_requests(self) -> list[dict[str, str]]:
        """The query of every catalog request, oldest first."""
        return [
            {key: values[0] for key, values in parse_qs(urlsplit(request.path).query).items()}
            for request in self.terminal.requests_to("/agents")
        ]

    def _folder(self, request: TerminalRequest):
        folder = self.moved.get(request.headers.get("x-session-id"), self.folder)
        return json_answer({"cwd": folder} if folder else {})

    def _move(self, request: TerminalRequest):
        """As Open Terminal: the working directory is kept per chat (`X-Session-Id`)."""
        folder = json.loads(request.body)["path"]
        self.moved[request.headers.get("x-session-id")] = folder
        return json_answer({"cwd": folder})

    def _catalog(self, request: TerminalRequest):
        folder = parse_qs(urlsplit(request.path).query).get("cwd", [None])[0]
        listed = self.catalog(folder) if callable(self.catalog) else self.catalog
        return listed if isinstance(listed, tuple) else json_answer(listed)

    def _start(self, provider: str, request: TerminalRequest):
        if self.refuse_start:
            status, body = self.refuse_start
            return json_answer(body, status=status)
        run = AgentRun(
            id=f"run-{uuid.uuid4().hex[:12]}",
            provider=provider,
            options=json.loads(request.body),
            headers=request.headers,
            agents=self,
        )
        self.starts.append(run)
        self.terminal.script_sessions(run.serve, f"/agents/sessions/{run.id}/connect")
        self.terminal.route("DELETE", f"/agents/sessions/{run.id}", lambda _: self._end(run))
        session = {"id": run.id, "agent": provider, "cwd": run.options.get("cwd"), "running": True}
        return json_answer({**session, "protocol_version": 1})

    @staticmethod
    def _end(run: AgentRun):
        run.deleted = True
        return json_answer({"status": True})


def serving_agents(terminal: FakeTerminalServer, *providers: str, **settings) -> ScriptedAgents:
    """Script the terminal's coding agents; `providers` are the ones a session can start for."""
    agents = ScriptedAgents(terminal, **settings)
    terminal.route("GET", "/agents", agents._catalog)
    terminal.route("GET", "/files/cwd", agents._folder)
    terminal.route("POST", "/files/cwd", agents._move)
    for provider in providers or (PROVIDER,):
        terminal.route(
            "POST",
            f"/agents/{provider}/sessions",
            lambda request, provider=provider: agents._start(provider, request),
        )
    return agents
