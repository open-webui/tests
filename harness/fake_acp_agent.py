"""A stand-in coding agent speaking ACP v1 over stdio, for an Open Terminal to run.

Open Terminal starts it in place of an installed agent CLI (`harness.coding_agent` points the
terminal's command override at it), once to discover its models and once per chat turn. Every
JSON-RPC message it receives goes to the JSON-lines file named by `FAKE_AGENT_LOG`, with a
`started` line per process, so a test reads what the agent was asked. What a turn does is chosen
by a marker in the prompt text:

- `[tool]` reports a tool call that reads a file, then answers.
- `[permission]` asks the client to allow a command and answers with the option it got back.
- `[ask]` asks the user a form question and answers with the value it got back.
- `[refuse]` answers the prompt with a JSON-RPC error.
- `[crash]` exits with status 3 halfway through the turn.
- `[slow]` starts answering, then waits for the client to cancel the turn.
- anything else thinks once and streams back what it heard, with its model and folder.

Its sessions offer a reasoning effort as a `thought_level` config option, with the efforts of the
session's model; picking another model announces that model's efforts with `config_option_update`.
"""

import json
import os
import sys
import time
import uuid

MODELS = [
    {"modelId": "fake-fast", "name": "Fake Fast"},
    {"modelId": "fake-deep", "name": "Fake Deep"},
]
# the reasoning efforts each model takes
EFFORTS = {"fake-fast": ("low", "medium", "high"), "fake-deep": ("low", "medium", "high", "max")}
REFUSAL = "The fake agent refused the prompt"
CRASH_NOTE = "fake agent crashed on purpose"

state = {"model": "fake-fast", "session": None, "next_id": 1000}


def log(entry: dict) -> None:
    path = os.environ.get("FAKE_AGENT_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")


def send(message: dict) -> None:
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
    sys.stdout.flush()


def receive() -> dict | None:
    line = sys.stdin.readline()
    if not line:
        return None
    message = json.loads(line)
    log({"pid": os.getpid(), "received": message})
    return message


def update(session_update: str, **fields) -> None:
    send(
        {
            "method": "session/update",
            "params": {
                "sessionId": state["session"],
                "update": {"sessionUpdate": session_update, **fields},
            },
        }
    )


def config_options() -> list[dict]:
    efforts = [
        {"value": effort, "name": effort.title()} for effort in EFFORTS.get(state["model"], ())
    ]
    effort = {
        "id": "effort",
        "name": "Effort",
        "category": "thought_level",
        "type": "select",
        "currentValue": "medium",
        "options": efforts,
    }
    return [effort]


def say(text: str) -> None:
    update("agent_message_chunk", content={"type": "text", "text": text})


def ask_client(method: str, params: dict) -> dict:
    """Send a request to the client and wait for its answer, logging what comes in between."""
    state["next_id"] += 1
    request_id = state["next_id"]
    send({"id": request_id, "method": method, "params": params})
    while (message := receive()) is not None:
        if message.get("id") == request_id and "method" not in message:
            return message
    sys.exit(0)


def prompt_text(params: dict) -> str:
    blocks = params.get("prompt") or []
    texts = [
        b.get("text", "") for b in blocks if b.get("type") == "text" and not b.get("annotations")
    ]
    return "\n".join(texts)


def run_turn(request_id, params: dict) -> None:
    text = prompt_text(params)
    if "[refuse]" in text:
        send({"id": request_id, "error": {"code": -32000, "message": REFUSAL}})
        return
    if "[crash]" in text:
        say("Starting the work... ")
        sys.stderr.write(CRASH_NOTE + "\n")
        sys.stderr.flush()
        os._exit(3)
    if "[slow]" in text:
        say("Working on it... ")
        # a second piece past Open WebUI's 50ms send throttle, so the first one is sent too
        time.sleep(0.2)
        say("still going")
        while (message := receive()) is not None:
            if message.get("method") == "session/cancel":
                send({"id": request_id, "result": {"stopReason": "cancelled"}})
                return
        sys.exit(0)
    if "[tool]" in text:
        call = {"toolCallId": "read-1", "title": "Read notes.txt", "kind": "read"}
        update("tool_call", **call, status="in_progress", rawInput={"path": "notes.txt"})
        update(
            "tool_call_update",
            toolCallId="read-1",
            status="completed",
            rawOutput="first line of notes",
        )
        say("The notes start with: first line of notes")
    elif "[permission]" in text:
        answer = ask_client(
            "session/request_permission",
            {
                "sessionId": state["session"],
                "toolCall": {
                    "toolCallId": "shell-1",
                    "title": "Remove the build folder",
                    "kind": "execute",
                    "rawInput": {"command": "rm -rf build"},
                },
                "options": [
                    {"optionId": "allow-it", "name": "Allow", "kind": "allow_once"},
                    {"optionId": "deny-it", "name": "Reject", "kind": "reject_once"},
                ],
            },
        )
        outcome = (answer.get("result") or {}).get("outcome") or {}
        say(f"Permission outcome: {outcome.get('optionId') or outcome.get('outcome')}")
    elif "[ask]" in text:
        answer = ask_client(
            "elicitation/create",
            {
                "sessionId": state["session"],
                "mode": "form",
                "message": "Which colour should the button be?",
                "requestedSchema": {
                    "type": "object",
                    "properties": {
                        "colour": {"type": "string", "title": "Colour", "enum": ["teal", "amber"]}
                    },
                    "required": ["colour"],
                },
            },
        )
        result = answer.get("result") or {}
        picked = (result.get("content") or {}).get("colour")
        say(
            f"Colour answer: {picked if result.get('action') == 'accept' else result.get('action')}"
        )
    else:
        update("agent_thought_chunk", content={"type": "text", "text": "Thinking it over."})
        for piece in ("Agent heard: ", text, f" [model {state['model']}, folder {os.getcwd()}]"):
            say(piece)
    send(
        {
            "id": request_id,
            "result": {
                "stopReason": "end_turn",
                "usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18},
            },
        }
    )


def main() -> None:
    log({"pid": os.getpid(), "started": sys.argv[1:], "cwd": os.getcwd()})
    while (message := receive()) is not None:
        method, params, request_id = (
            message.get("method"),
            message.get("params") or {},
            message.get("id"),
        )
        if request_id is None:
            continue
        if method == "initialize":
            result = {
                "protocolVersion": 1,
                "agentCapabilities": {"loadSession": True, "promptCapabilities": {"image": False}},
                "authMethods": [],
            }
        elif method == "session/new":
            state["session"] = f"fake-{uuid.uuid4().hex[:12]}"
            result = {
                "sessionId": state["session"],
                "models": {"currentModelId": state["model"], "availableModels": MODELS},
                "configOptions": config_options(),
            }
        elif method == "session/load":
            state["session"] = params.get("sessionId")
            result = {
                "models": {"currentModelId": state["model"], "availableModels": MODELS},
                "configOptions": config_options(),
            }
        elif method == "session/set_model":
            state["model"] = params.get("modelId")
            update("config_option_update", configOptions=config_options())
            result = {}
        elif method == "session/set_config_option":
            result = {"configOptions": config_options()}
        elif method == "session/prompt":
            run_turn(request_id, params)
            continue
        else:
            send({"id": request_id, "error": {"code": -32601, "message": f"no {method}"}})
            continue
        send({"id": request_id, "result": result})


if __name__ == "__main__":
    main()
