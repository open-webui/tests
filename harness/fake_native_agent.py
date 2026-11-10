"""A stand-in for the three coding agents that speak a protocol of their own, over stdio.

Open Terminal starts it in place of the Claude Code, Codex and Pi CLIs (`harness.coding_agent`
points the terminal's command overrides at it), once to discover the models and once per chat
turn. The arguments Open Terminal starts it with decide which agent it is: `app-server` is Codex
(JSON-RPC threads and turns), `--mode rpc` is Pi (commands, events and dialogs), anything else is
Claude Code (stream-json with control requests). Every message it receives goes to the JSON-lines
file named by `FAKE_AGENT_LOG` with the agent's name, after a `started` line with the arguments;
each turn adds a `conversation` line with the id of the agent's own conversation it runs in.
What a turn does is chosen by a marker in the prompt, as in `harness/fake_acp_agent.py`:

- `[tool]` runs one tool that reads a file, then answers.
- `[permission]` asks to be allowed a command and answers with the decision it got back.
- `[ask]` asks the person which colour and answers with the value it got back.
- `[refuse]` fails the turn the way that agent reports a failed turn.
- `[slow]` starts answering, then waits to be interrupted.
- anything else thinks once and streams back what it heard, with its model and folder.
"""

import json
import os
import sys
import time
import uuid

REFUSAL = "The fake agent refused the prompt"
QUESTION = "Which colour should the button be?"
COMMAND = "rm -rf build"
THOUGHT = "Thinking it over."
NOTES = "first line of notes"
CLAUDE_MODELS = [
    {"value": "sonnet", "displayName": "Fake Sonnet", "resolvedModel": "claude-fake-sonnet"},
    {"value": "default", "displayName": "Fake Sonnet", "resolvedModel": "claude-fake-sonnet"},
    {"value": "claude-fake-haiku", "displayName": "Fake Haiku"},
]
CODEX_MODELS = [
    {
        "model": "codex-fake",
        "displayName": "Fake Codex",
        "supportedReasoningEfforts": [
            {"reasoningEffort": "low"},
            {"reasoningEffort": "medium"},
            {"reasoningEffort": "high"},
        ],
        "defaultReasoningEffort": "medium",
        "serviceTiers": [{"id": "standard", "name": "Standard"}, {"id": "fast", "name": "Fast"}],
    },
    {"model": "codex-fake-mini"},
]
PI_MODELS = [{"provider": "acme", "id": "pi-fake", "name": "Fake Pi"}]

arguments = sys.argv[1:]
if arguments[:1] == ["app-server"]:
    AGENT = "codex"
elif arguments[:2] == ["--mode", "rpc"]:
    AGENT = "pi"
else:
    AGENT = "claude-code"


def argument(name: str) -> str | None:
    return arguments[arguments.index(name) + 1] if name in arguments else None


def log(entry: dict) -> None:
    path = os.environ.get("FAKE_AGENT_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "agent": AGENT, **entry}) + "\n")


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def receive() -> dict:
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    message = json.loads(line)
    log({"received": message})
    return message


def heard(text: str, model: str | None) -> list[str]:
    return [f"{AGENT} heard: ", text, f" [model {model}, folder {os.getcwd()}]"]


def claude() -> None:
    session = argument("--resume") or f"claude-{uuid.uuid4().hex[:12]}"

    def event(kind: str, **fields) -> None:
        send({"type": "stream_event", "event": {"type": kind, **fields}, "session_id": session})

    def say(text: str, thought: bool = False) -> None:
        delta = (
            {"type": "thinking_delta", "thinking": text}
            if thought
            else {"type": "text_delta", "text": text}
        )
        event("content_block_delta", index=0, delta=delta)

    def may_use(tool: str, tool_input: dict) -> dict:
        request_id = uuid.uuid4().hex
        request = {"subtype": "can_use_tool", "tool_name": tool, "input": tool_input}
        send({"type": "control_request", "request_id": request_id, "request": request})
        while True:
            answer = receive()
            response = answer.get("response") or {}
            if (
                answer.get("type") == "control_response"
                and response.get("request_id") == request_id
            ):
                return response.get("response") or {}

    def result(**fields) -> None:
        usage = {"input_tokens": 11, "output_tokens": 7}
        send({"type": "result", "session_id": session, "usage": usage, **fields})

    def turn(text: str) -> None:
        log({"conversation": session})
        send({"type": "system", "subtype": "init", "session_id": session})
        event("message_start")
        if "[refuse]" in text:
            return result(subtype="error_during_execution", is_error=True, result=REFUSAL)
        if "[slow]" in text:
            say("Working on it... ")
            time.sleep(0.2)
            say("still going")
            while (receive().get("request") or {}).get("subtype") != "interrupt":
                pass
            sys.exit(0)
        if "[tool]" in text:
            use = {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "Read",
                "input": {"path": "notes.txt"},
            }
            done = {"type": "tool_result", "tool_use_id": "toolu_1", "content": NOTES}
            send({"type": "assistant", "message": {"role": "assistant", "content": [use]}})
            send({"type": "user", "message": {"role": "user", "content": [done]}})
            say(f"The notes start with: {NOTES}")
        elif "[permission]" in text:
            answer = may_use("Bash", {"command": COMMAND})
            say(f"Permission outcome: {answer.get('behavior')}")
        elif "[ask]" in text:
            options = [{"label": "teal", "description": ""}, {"label": "amber", "description": ""}]
            questions = [{"question": QUESTION, "header": "Colour", "options": options}]
            answer = may_use("AskUserQuestion", {"questions": questions})
            picked = (answer.get("updatedInput") or {}).get("answers") or {}
            say(f"Colour answer: {picked.get(QUESTION) if picked else answer.get('behavior')}")
        else:
            say(THOUGHT, thought=True)
            for piece in heard(text, argument("--model")):
                say(piece)
        result(subtype="success", is_error=False)

    while True:
        message = receive()
        if message.get("type") == "control_request":
            answer = {"subtype": "success", "request_id": message["request_id"]}
            send(
                {
                    "type": "control_response",
                    "response": {**answer, "response": {"models": CLAUDE_MODELS}},
                }
            )
        elif message.get("type") == "user":
            blocks = message["message"]["content"]
            turn("\n".join(b.get("text", "") for b in blocks if b.get("type") == "text"))


def codex() -> None:
    state = {"model": None, "thread": None, "turn": f"turn-{uuid.uuid4().hex[:8]}", "next": 500}

    def notify(method: str, **params) -> None:
        send({"method": method, "params": {"turnId": state["turn"], **params}})

    def say(text: str, thought: bool = False) -> None:
        notify("item/reasoning/textDelta" if thought else "item/agentMessage/delta", delta=text)

    def ask(method: str, **params) -> dict:
        state["next"] += 1
        send({"id": state["next"], "method": method, "params": {"turnId": state["turn"], **params}})
        while True:
            answer = receive()
            if answer.get("id") == state["next"] and "method" not in answer:
                return answer.get("result") or {}

    def turn(text: str) -> None:
        log({"conversation": state["thread"]})
        if "[refuse]" in text:
            return notify("turn/completed", turn={"status": "failed", "error": REFUSAL})
        if "[slow]" in text:
            say("Working on it... ")
            time.sleep(0.2)
            say("still going")
            while True:
                message = receive()
                if message.get("method") == "turn/interrupt":
                    send({"id": message["id"], "result": {}})
                    sys.exit(0)
        if "[tool]" in text:
            item = {"id": "cmd_1", "type": "commandExecution", "command": "cat notes.txt"}
            notify("item/started", item=item)
            notify("item/completed", item={**item, "aggregatedOutput": NOTES})
            say(f"The notes start with: {NOTES}")
        elif "[permission]" in text:
            answer = ask("item/commandExecution/requestApproval", itemId="cmd_2", command=COMMAND)
            say(f"Permission outcome: {answer.get('decision')}")
        elif "[ask]" in text:
            options = [{"label": "teal", "description": ""}, {"label": "amber", "description": ""}]
            question = {
                "id": "colour",
                "header": "Colour",
                "question": QUESTION,
                "options": options,
            }
            answer = ask("item/tool/requestUserInput", questions=[question])
            picked = ((answer.get("answers") or {}).get("colour") or {}).get("answers") or []
            say(f"Colour answer: {', '.join(picked)}")
        else:
            say(THOUGHT, thought=True)
            for piece in heard(text, state["model"]):
                say(piece)
        last = {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18}
        notify("thread/tokenUsage/updated", tokenUsage={"last": last})
        notify("turn/completed", turn={"id": state["turn"], "status": "completed"})

    while True:
        message = receive()
        method, params, request_id = (
            message.get("method"),
            message.get("params") or {},
            message.get("id"),
        )
        if request_id is None:
            continue
        if method == "model/list":
            send({"id": request_id, "result": {"data": CODEX_MODELS, "nextCursor": None}})
        elif method in ("thread/start", "thread/resume"):
            state["model"] = params.get("model")
            state["thread"] = params.get("threadId") or f"thread-{uuid.uuid4().hex[:12]}"
            send({"id": request_id, "result": {"thread": {"id": state["thread"]}}})
        elif method == "turn/start":
            send({"id": request_id, "result": {"turn": {"id": state["turn"]}}})
            texts = [part.get("text", "") for part in params["input"] if part.get("type") == "text"]
            turn("\n".join(texts))
        else:
            send({"id": request_id, "result": {}})


def pi() -> None:
    state = {"model": None, "session": argument("--session-id")}

    def say(text: str, thought: bool = False) -> None:
        delta = {"type": "thinking_delta" if thought else "text_delta", "delta": text}
        send({"type": "message_update", "assistantMessageEvent": delta})

    def dialog(method: str, **fields) -> dict:
        request_id = uuid.uuid4().hex
        send({"type": "extension_ui_request", "id": request_id, "method": method, **fields})
        while True:
            answer = receive()
            if answer.get("type") == "extension_ui_response" and answer.get("id") == request_id:
                return answer

    def end(**fields) -> None:
        usage = {"input": 9, "cacheRead": 2, "output": 7, "totalTokens": 18}
        send({"type": "message_end", "message": {"role": "assistant", "usage": usage, **fields}})
        send({"type": "agent_settled"})

    def turn(text: str) -> None:
        log({"conversation": state["session"]})
        send({"type": "message_start"})
        if "[refuse]" in text:
            return end(stopReason="error", errorMessage=REFUSAL)
        if "[slow]" in text:
            say("Working on it... ")
            time.sleep(0.2)
            say("still going")
            while receive().get("type") != "abort":
                pass
            sys.exit(0)
        if "[tool]" in text:
            call = {"toolCallId": "call_1", "toolName": "read", "args": {"path": "notes.txt"}}
            send({"type": "tool_execution_start", **call})
            send({"type": "tool_execution_end", "toolCallId": "call_1", "result": NOTES})
            say(f"The notes start with: {NOTES}")
        elif "[permission]" in text:
            answer = dialog("confirm", title="Remove the build folder", message=COMMAND)
            say(f"Permission outcome: {'confirmed' if answer.get('confirmed') else 'declined'}")
        elif "[ask]" in text:
            answer = dialog("select", title=QUESTION, options=["teal", "amber"])
            say(f"Colour answer: {answer.get('value', 'cancelled')}")
        else:
            say(THOUGHT, thought=True)
            for piece in heard(text, state["model"]):
                say(piece)
        end(stopReason="stop")

    while True:
        message = receive()
        kind, data = message.get("type"), {}
        if kind == "get_available_models":
            data = {"models": PI_MODELS}
        elif kind == "set_model":
            state["model"] = f"{message['provider']}/{message['modelId']}"
        elif kind == "get_state":
            data = {"sessionId": state["session"], "isStreaming": False}
        send({"type": "response", "id": message.get("id"), "success": True, "data": data})
        if kind == "prompt":
            turn(message["message"])


if __name__ == "__main__":
    log({"started": arguments, "cwd": os.getcwd()})
    {"codex": codex, "pi": pi, "claude-code": claude}[AGENT]()
