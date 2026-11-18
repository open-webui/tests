"""Write the stored fat chats under `data/fat_chats/`, deterministically from fixed seeds.

Each data set is one long conversation in the shapes current Open WebUI stores: hundreds of
messages with edits, regenerations and side-by-side replies of several models, native tool calls
with large results, reasoning, citations, attached files, code, tables and status updates, a
stretch answered by a coding agent in a terminal (signed `meta.agent` sessions) and the sub-agent
chats it delegated to. The tests load the stored files through `harness.fat_chats`; this script
only exists to make them again, byte for byte, when the shapes change.

    python scripts/generate_fat_chats.py            # every data set
    python scripts/generate_fat_chats.py fat-1mb    # one
"""

from __future__ import annotations

import argparse
import json
import lzma
import random
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import jwt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.fat_chats import (  # noqa: E402
    DATA_DIR,
    FORMAT,
    OWNER_ID,
    OWNER_NAME,
    TERMINAL_CWD,
    TERMINAL_ID,
)
from harness.instance import WEBUI_SECRET_KEY  # noqa: E402

# name: (seed, bytes of chat content to reach, average bytes per turn)
DATA_SETS = {
    "fat-1mb": (1, 1_000_000, 500),
    "fat-10mb": (10, 10_000_000, 25_000),
    "fat-50mb": (50, 50_000_000, 80_000),
    "fat-130mb": (130, 130_000_000, 150_000),
}

START = 1_767_225_600  # 2026-01-01, the first message
MODELS = ["mock-model", "example-large", "example-reasoner"]
AGENT_PROVIDER, AGENT_MODEL = "claude-code", "claude-fake-haiku"
AGENT_MODEL_ID = f"agent:{TERMINAL_ID}:{AGENT_PROVIDER}:{AGENT_MODEL}"
SEARCH_TERM = "quillwort"  # in one early message only, so a search has to reach it

WORDS = (
    "account action adapter address agent alert allocation archive array audit backup balance "
    "batch boundary branch buffer build bundle cache call capacity catalog certificate change "
    "channel check client cluster column command commit config connection consumer container "
    "context counter cursor dashboard data database deadline default delay dependency deploy "
    "detail device digest directory disk document domain draft driver endpoint engine entry "
    "environment error event export extension feature field file filter flag folder format frame "
    "function gateway graph group handler hash header heap history host image import index input "
    "instance interval issue item job journal kernel key label latency layer layout limit link "
    "list loader lock log loop manifest map memory merge message metric migration mirror mode "
    "model module monitor mount network node notice object offset option order origin output "
    "owner package page parameter parser partition patch path payload peer permission pipeline "
    "plan policy pool port prefix preview priority process profile project prompt property proxy "
    "query queue quota range rate record region registry release replica report request resource "
    "response result retry review revision role route rule runner sample schedule schema scope "
    "script search secret segment selector server service session setting shard signal size "
    "snapshot socket source span stage state status step storage stream string structure summary "
    "switch table tag target task template tenant test thread threshold timeout token tool topic "
    "trace track transfer trigger type update upload user value variable version view volume "
    "warning window worker workspace zone"
).split()
VERBS = (
    "adds applies builds caches checks clears collects compares copies counts creates drops "
    "expands fetches finds flushes groups handles keeps limits lists loads locks maps marks "
    "merges moves opens parses passes reads records reduces refreshes removes renames repairs "
    "replaces resolves returns rotates runs saves scans sends sets skips sorts splits starts "
    "stores streams syncs tracks trims updates validates watches writes"
).split()
LINKS = ["the", "a", "each", "every", "one", "that", "this", "its", "no", "the next", "the old"]
LEVELS = ["INFO", "INFO", "INFO", "DEBUG", "WARN", "ERROR"]
TOOLS = [
    ("run_command", "tool_shell"),
    ("read_file", "tool_files"),
    ("list_files", "tool_files"),
    ("query_knowledge_files", "builtin:query_knowledge_files"),
    ("search_web", "builtin:search_web"),
    ("fetch_url", "builtin:fetch_url"),
]


class Writer:
    """Deterministic text and ids, all drawn from one seeded generator."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def uuid(self) -> str:
        return str(uuid.UUID(int=self.rng.getrandbits(128), version=4))

    def hex(self, length: int = 24) -> str:
        return f"{self.rng.getrandbits(length * 4):0{length}x}"

    def words(self, count: int) -> str:
        return " ".join(self.rng.choices(WORDS, k=count))

    def sentence(self) -> str:
        subject = f"{self.rng.choice(LINKS)} {self.words(self.rng.randint(1, 2))}"
        rest = self.words(self.rng.randint(4, 14))
        return f"{subject.capitalize()} {self.rng.choice(VERBS)} {rest}."

    def paragraph(self) -> str:
        return " ".join(self.sentence() for _ in range(self.rng.randint(3, 7)))

    def prose(self, size: int) -> str:
        parts, length = [], 0
        while length < size:
            parts.append(self.paragraph())
            length += len(parts[-1]) + 2
        return "\n\n".join(parts)

    def name(self) -> str:
        return "_".join(self.rng.sample(WORDS, 2)).replace("-", "_")

    def path(self) -> str:
        folders = "/".join(self.rng.sample(WORDS, self.rng.randint(1, 3)))
        return f"{TERMINAL_CWD}/{folders}/{self.name()}.py"

    def url(self) -> str:
        host = self.rng.choice(["example.com", "docs.example.com", "api.example.com"])
        return f"https://{host}/{'/'.join(self.rng.sample(WORDS, 2))}"

    def python(self) -> str:
        function, items, key, amount = (
            self.name(),
            self.rng.choice(WORDS),
            *self.rng.sample(WORDS, 2),
        )
        return (
            f"def {function}({items}s: list[dict]) -> dict:\n"
            f'    """{self.sentence()}"""\n'
            "    totals = {}\n"
            f"    for {items} in {items}s:\n"
            f'        key = {items}.get("{key}")\n'
            "        if key is None:\n"
            "            continue\n"
            f'        totals[key] = totals.get(key, 0) + {items}["{amount}"]\n'
            "    return totals\n"
        )

    def code_block(self) -> str:
        kind = self.rng.choice(["python", "bash", "sql", "json"])
        if kind == "python":
            body = "\n".join(self.python() for _ in range(self.rng.randint(1, 3)))
        elif kind == "bash":
            body = "\n".join(
                f"{self.rng.choice(['grep -rn', 'ls -la', 'cat', 'wc -l', 'python -m pytest'])} "
                f"{self.path()}"
                for _ in range(self.rng.randint(2, 6))
            )
        elif kind == "sql":
            table, column, other = self.rng.sample(WORDS, 3)
            body = (
                f"SELECT {column}, COUNT(*) AS total\nFROM {table}\n"
                f"WHERE {other} > {self.rng.randint(1, 900)}\nGROUP BY {column}\n"
                "ORDER BY total DESC\nLIMIT 50;"
            )
        else:
            body = json.dumps(
                {word: self.rng.randint(0, 5000) for word in self.rng.sample(WORDS, 6)}, indent=2
            )
        return f"```{kind}\n{body}\n```"

    def table(self) -> str:
        columns = self.rng.sample(WORDS, self.rng.randint(3, 5))
        lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        for _ in range(self.rng.randint(3, 12)):
            cells = [self.words(1) if i == 0 else str(self.rng.randint(0, 9999)) for i in columns]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def answer(self, size: int, citations: int = 0) -> str:
        """A markdown reply: headings, prose, lists, code, tables and citations."""
        parts, length = [], 0
        while length < size:
            choice = self.rng.random()
            if choice < 0.12:
                parts.append(f"## {self.words(self.rng.randint(2, 4)).capitalize()}")
            elif choice < 0.3:
                parts.append(self.code_block())
            elif choice < 0.4:
                parts.append(self.table())
            elif choice < 0.55:
                parts.append(
                    "\n".join(f"- {self.sentence()}" for _ in range(self.rng.randint(2, 6)))
                )
            else:
                text = self.paragraph()
                if citations and self.rng.random() < 0.5:
                    text += f" [{self.rng.randint(1, citations)}]"
                parts.append(text)
            length += len(parts[-1]) + 2
        return "\n\n".join(parts)

    def log(self, size: int) -> str:
        lines, length, moment = [], 0, self.rng.randint(0, 80000)
        while length < size:
            moment += self.rng.randint(0, 3)
            hours, rest = divmod(moment % 86400, 3600)
            day = 1 + moment // 86400 % 28
            stamp = f"2026-01-{day:02d}T{hours:02d}:{rest // 60:02d}:{rest % 60:02d}Z"
            level, worker, number = (
                self.rng.choice(LEVELS),
                self.rng.choice(WORDS),
                self.rng.randint(1, 16),
            )
            lines.append(
                f"{stamp} {level} {worker}[{number}] "
                f"{self.rng.choice(VERBS)} {self.words(self.rng.randint(2, 6))} "
                f"in {self.rng.randint(1, 900)}ms path={self.path()}"
            )
            length += len(lines[-1]) + 1
        return "\n".join(lines)

    def listing(self, size: int) -> str:
        lines, length = [], 0
        while length < size:
            size_bytes, day, hour, minute = (
                self.rng.randint(10, 990000),
                self.rng.randint(1, 28),
                self.rng.randint(0, 23),
                self.rng.randint(0, 59),
            )
            lines.append(
                f"-rw-r--r-- 1 user user {size_bytes:>7} Jan {day:>2} {hour:02d}:{minute:02d} "
                f"{self.path()}"
            )
            length += len(lines[-1]) + 1
        return "\n".join(lines)

    def source_file(self, size: int) -> str:
        parts, length = [f'"""{self.sentence()}"""\n'], 0
        while length < size:
            parts.append(self.python())
            length += len(parts[-1]) + 1
        return "\n".join(parts)


@dataclass
class Chat:
    """One conversation as it is stored: the chat record and its message tree."""

    id: str
    title: str
    messages: dict[str, dict] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    current_id: str = ""

    def add(self, message: dict) -> dict:
        self.messages[message["id"]] = message
        parent = self.messages.get(message["parentId"])
        if parent:
            parent["childrenIds"].append(message["id"])
        return message

    def record(self, current_id: str, models: list[str], created_at: int, updated_at: int) -> dict:
        first_user = next(m for m in self.messages.values() if m["role"] == "user")
        chat = {
            "id": self.id,
            "title": self.title,
            "models": models,
            "terminal_id": None,
            "working_directory": "",
            "history": {"currentId": current_id, "messages": self.messages},
            "messages": [
                {
                    "role": "user",
                    "content": first_user["content"],
                    "user_id": OWNER_ID,
                    "user": {"id": OWNER_ID, "name": OWNER_NAME},
                }
            ],
            "files": [],
            "tags": [],
            "timestamp": created_at * 1000,
        }
        return {
            "id": self.id,
            "title": self.title,
            "chat": chat,
            "meta": self.meta,
            "created_at": created_at,
            "updated_at": updated_at,
            "current_message_id": current_id,
        }


class Conversation:
    """Grows one long conversation turn by turn until it holds `target` bytes."""

    def __init__(self, writer: Writer, target: int, per_turn: int):
        self.w = writer
        self.target = target
        self.per_turn = per_turn
        self.clock = START
        self.size = 0
        self.main = Chat(writer.uuid(), f"Fixture project review {writer.words(2)}")
        self.subagents: list[Chat] = []
        self.notable: dict[str, str] = {}

    # -- message shapes ----------------------------------------------------------------------

    def tick(self, seconds: int = 0) -> int:
        self.clock += seconds or self.w.rng.randint(20, 240)
        return self.clock

    def user(self, chat: Chat, parent_id: str | None, content: str, models=None, **extra) -> dict:
        self.size += len(content)
        return chat.add(
            {
                "id": self.w.uuid(),
                "parentId": parent_id,
                "childrenIds": [],
                "role": "user",
                "content": content,
                "models": models or ["mock-model"],
                "timestamp": self.tick(),
                "user_id": OWNER_ID,
                "user": {"id": OWNER_ID, "name": OWNER_NAME},
                **extra,
            }
        )

    def file_item(self) -> dict:
        file_id = self.w.uuid()
        name = f"{self.w.name()}.{self.w.rng.choice(['txt', 'csv', 'md', 'py', 'pdf'])}"
        size = self.w.rng.randint(200, 900_000)
        return {
            "type": "file",
            "file": {
                "id": file_id,
                "user_id": OWNER_ID,
                "filename": name,
                "meta": {"name": name, "content_type": "text/plain", "size": size},
                "created_at": self.clock,
                "updated_at": self.clock,
            },
            "id": file_id,
            "url": file_id,
            "name": name,
            "collection_name": f"file-{file_id}",
            "status": "uploaded",
            "size": size,
            "error": "",
            "itemId": self.w.uuid(),
            "content_type": "text/plain",
        }

    def reasoning(self, text: str) -> dict:
        started = self.clock + self.w.rng.random()
        duration = self.w.rng.randint(1, 40)
        self.size += len(text)
        return {
            "type": "reasoning",
            "id": f"r_{self.w.hex()}",
            "status": "completed",
            "start_tag": "<think>",
            "end_tag": "</think>",
            "attributes": {"type": "reasoning_content"},
            "content": [{"type": "output_text", "text": text}],
            "summary": None,
            "started_at": started,
            "ended_at": started + duration,
            "duration": duration,
        }

    def text_item(self, text: str) -> dict:
        self.size += len(text) * 2  # the reply's content repeats it
        return {
            "type": "message",
            "id": f"msg_{self.w.hex()}",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }

    def tool_pair(self, name: str, tool_id: str, arguments: dict, result: str) -> list[dict]:
        call_id = f"call_{self.w.hex()}"
        started = self.clock + self.w.rng.random()
        duration = self.w.rng.random() * 3
        call = {
            "type": "function_call",
            "id": call_id,
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(arguments),
            "status": "completed",
            "started_at": started,
            "ended_at": started + duration,
            "duration": duration,
            "tool_id": tool_id,
        }
        if tool_id.startswith("builtin:"):
            call["tool_type"] = "builtin"
        self.size += len(result) + len(call["arguments"])
        return [
            call,
            {
                "type": "function_call_output",
                "id": f"fco_{self.w.hex()}",
                "call_id": call_id,
                "output": [{"type": "input_text", "text": result}],
                "status": "completed",
            },
        ]

    def sources(self, count: int) -> list[dict]:
        sources = []
        for _ in range(count):
            file_id = self.w.uuid()
            name = f"{self.w.name()}.md"
            documents = [self.w.paragraph() for _ in range(self.w.rng.randint(1, 4))]
            self.size += sum(map(len, documents))
            sources.append(
                {
                    "source": {"id": file_id, "name": name, "type": "file"},
                    "document": documents,
                    "metadata": [
                        {"file_id": file_id, "name": name, "source": name} for _ in documents
                    ],
                    "distances": [round(self.w.rng.random(), 6) for _ in documents],
                }
            )
        return sources

    def tool_result(self, name: str, size: int) -> tuple[dict, str]:
        if name == "run_command":
            return {"command": f"grep -rn {self.w.rng.choice(WORDS)} {TERMINAL_CWD}"}, self.w.log(
                size
            )
        if name == "read_file":
            return {"path": self.w.path()}, self.w.source_file(size)
        if name == "list_files":
            return {"path": TERMINAL_CWD}, self.w.listing(size)
        if name == "query_knowledge_files":
            chunks, length = [], 0
            while length < size:
                chunks.append(
                    {
                        "content": self.w.paragraph(),
                        "source": f"{self.w.name()}.md",
                        "file_id": self.w.uuid(),
                        "distance": round(self.w.rng.random(), 6),
                    }
                )
                length += len(chunks[-1]["content"]) + 120
            return {"query": self.w.words(3)}, json.dumps(chunks)
        if name == "search_web":
            results, length = [], 0
            while length < size:
                results.append(
                    {"title": self.w.words(4), "link": self.w.url(), "snippet": self.w.paragraph()}
                )
                length += len(results[-1]["snippet"]) + 80
            return {"query": self.w.words(3)}, json.dumps(results)
        return {"url": self.w.url()}, self.w.prose(size)

    def assistant(
        self, chat: Chat, parent: dict, budget: int, model: str = "mock-model", **extra
    ) -> dict:
        """A finished reply: thinking, tool rounds with their results and a markdown answer."""
        rng = self.w.rng
        output: list[dict] = []
        citations = 0
        if rng.random() < 0.6:
            output.append(self.reasoning(self.w.prose(max(200, int(budget * 0.08)))))
        rounds = rng.choice([0, 1, 1, 2, 3])
        for _ in range(rounds):
            if rng.random() < 0.3:
                output.append(self.text_item(self.w.sentence()))
            for _ in range(rng.randint(1, 3)):
                name, tool_id = rng.choice(TOOLS)
                share = budget * 0.6 / rounds * rng.lognormvariate(0, 0.6)
                # now and then a result far bigger than the rest, like a whole log
                size = int(share * (8 if rng.random() < 0.04 else 1))
                arguments, result = self.tool_result(name, max(80, size))
                output.extend(self.tool_pair(name, tool_id, arguments, result))
                citations += name in ("query_knowledge_files", "search_web")
            if rng.random() < 0.5:
                output.append(self.reasoning(self.w.prose(max(120, int(budget * 0.03)))))
        answer = self.w.answer(max(150, int(budget * 0.25)), citations=3 if citations else 0)
        output.append(self.text_item(answer))
        message = {
            "id": self.w.uuid(),
            "parentId": parent["id"],
            "childrenIds": [],
            "role": "assistant",
            "content": "\n".join(i["content"][0]["text"] for i in output if i["type"] == "message"),
            "done": True,
            "model": model,
            "modelIdx": extra.pop("modelIdx", None),
            "timestamp": self.tick(rng.randint(2, 90)),
            "user_id": OWNER_ID,
            "meta": {
                "task_id": self.w.uuid(),
                "response_started_at": self.clock + 0.4,
                "response_ended_at": self.clock + rng.randint(2, 90) + 0.7,
            },
            "output": output,
            "usage": self.usage(budget),
            **extra,
        }
        if citations:
            message["sources"] = self.sources(3)
        if rng.random() < 0.15:
            message["statusHistory"] = [
                {
                    "action": "web_search_queries_generated",
                    "queries": [self.w.words(3)],
                    "done": False,
                },
                {
                    "action": "web_search",
                    "description": "Searched {{count}} sites",
                    "urls": [self.w.url() for _ in range(3)],
                    "done": True,
                },
            ]
        return chat.add(message)

    def usage(self, budget: int) -> dict:
        prompt = self.w.rng.randint(500, 120_000)
        completion = max(1, budget // 4)
        return {
            "input_tokens": prompt,
            "output_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
        }

    def agent_reply(self, chat: Chat, parent: dict, budget: int, session_id: str) -> dict:
        """A turn the terminal's coding agent answered, with its signed session."""
        output = [self.reasoning(self.w.prose(max(200, budget // 10)))]
        for _ in range(self.w.rng.randint(1, 4)):
            name = self.w.rng.choice(["Read", "Bash", "Grep", "Edit"])
            arguments, result = self.tool_result(
                "read_file" if name in ("Read", "Edit") else "run_command",
                max(80, int(budget * 0.5 * self.w.rng.random())),
            )
            call, call_output = self.tool_pair(name, "agent", arguments, result)
            for key in ("tool_id", "started_at", "ended_at", "duration"):
                call.pop(key)
            call["provider_data"] = {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call["call_id"],
                            "content": result[:200],
                        }
                    ],
                },
            }
            call_output["id"] = f"out_{call['call_id']}"
            output.extend([call, call_output])
        text = self.text_item(self.w.answer(max(150, budget // 5)))
        text["id"] = self.w.hex(32)
        output.append(text)
        state = {
            "typ": "agent_session",
            "user": OWNER_ID,
            "chat": chat.id,
            "terminal_id": TERMINAL_ID,
            "provider": AGENT_PROVIDER,
            "model": AGENT_MODEL,
            "instance_user_id": OWNER_ID,
            "cwd": TERMINAL_CWD,
            "session_id": session_id,
            "turn": self.w.uuid(),
            "jti": self.w.uuid(),
            "iat": self.clock,
        }
        model_name = f"{AGENT_PROVIDER}/{AGENT_MODEL}"
        return chat.add(
            {
                "id": self.w.uuid(),
                "parentId": parent["id"],
                "childrenIds": [],
                "role": "assistant",
                "content": text["content"][0]["text"],
                "done": True,
                "model": AGENT_MODEL_ID,
                "timestamp": self.tick(),
                "meta": {
                    "task_id": self.w.uuid(),
                    "response_started_at": self.clock + 0.3,
                    "model_name": model_name,
                    "agent": jwt.encode(state, WEBUI_SECRET_KEY, algorithm="HS256"),
                    "response_ended_at": self.clock + 40.5,
                },
                "modelName": model_name,
                "user_id": OWNER_ID,
                "output": output,
                "usage": {
                    "input_tokens": self.w.rng.randint(100, 9000),
                    "output_tokens": budget // 4,
                },
            }
        )

    # -- the conversation --------------------------------------------------------------------

    def subagent_chat(self, task: str, parent_message_id: str, delegation_id: str) -> Chat:
        child = Chat(self.w.uuid(), f"Sub-agent: {task[:60]}")
        child.meta = {
            "internal": True,
            "type": "subagent",
            "parent_chat_id": self.main.id,
            "parent_message_id": parent_message_id,
            "delegation_id": delegation_id,
            "mode": "foreground",
        }
        prompt = self.user(child, None, f"{task}\n\n## Context\n{self.w.paragraph()}")
        reply = self.assistant(child, prompt, self.per_turn * 2)
        child.current_id = reply["id"]
        self.subagents.append(child)
        return child

    def delegation(self, parent: dict, background: bool) -> tuple[dict, Chat]:
        """A reply that hands a task to a sub-agent, and the sub-agent's own chat."""
        task = f"Check {self.w.words(4)} across the {self.w.rng.choice(WORDS)} module"
        delegation_id = f"deleg_{self.w.hex(8)}"
        reply = self.assistant(self.main, parent, self.per_turn)
        child = self.subagent_chat(task, reply["id"], delegation_id)
        if background:
            child.meta["mode"] = "background"
            result = json.dumps(
                {
                    "status": "dispatched",
                    "delegation_id": delegation_id,
                    "subagent_chat_id": child.id,
                    "mode": "background",
                    "task": task,
                }
            )
        else:
            result = child.messages[child.current_id]["content"]
        arguments = {"task": task, "context": self.w.sentence(), "background": background}
        pair = self.tool_pair("delegate_task", "builtin:delegate_task", arguments, result)
        reply["output"][-1:-1] = pair
        return reply, child

    def branch(self, parent: dict, depth: int) -> None:
        """A short side branch below `parent` that the conversation left again."""
        for _ in range(depth):
            question = self.user(self.main, parent["id"], self.w.prose(self.w.rng.randint(60, 400)))
            parent = self.assistant(
                self.main, question, self.budget(), model=self.w.rng.choice(MODELS)
            )

    def budget(self) -> int:
        return int(self.per_turn * self.w.rng.lognormvariate(0, 0.5))

    def build(self) -> list[dict]:
        rng = self.w.rng
        session_id = f"claude-{self.w.hex(12)}"
        delegations = [(0.3, False), (0.45, True)]
        parent: dict | None = None
        turn = 0
        branched = False
        # a side branch below a reply needs one more turn on top of it, so the final reply is a leaf
        while self.size < self.target or turn < 100 or branched:
            branched = False
            progress = self.size / self.target
            # a stretch late in the chat is answered by the terminal's coding agent
            by_agent = 0.62 <= progress < 0.78
            compare = not by_agent and rng.random() < 0.06
            models = MODELS[:2] if compare else [AGENT_MODEL_ID if by_agent else "mock-model"]
            parent_id = parent and parent["id"]
            if turn and rng.random() < 0.08:
                # an edit: the earlier wording stays as a sibling with its own reply
                earlier = self.user(self.main, parent_id, self.w.prose(rng.randint(40, 300)))
                self.assistant(self.main, earlier, self.budget())
                self.notable.setdefault("edited_parent", parent_id)
            text = self.w.prose(rng.randint(40, 900))
            if turn == 3:
                text += f" Note the {SEARCH_TERM} entry."
            files = [self.file_item() for _ in range(rng.randint(1, 3))]
            extra = {"files": files} if rng.random() < 0.12 else {}
            question = self.user(self.main, parent_id, text, models=models, **extra)
            self.notable.setdefault("first_user", question["id"])
            if by_agent:
                reply = self.agent_reply(self.main, question, self.budget(), session_id)
                self.notable["agent_reply"] = reply["id"]
            elif delegations and progress >= delegations[0][0]:
                _, background = delegations.pop(0)
                reply, child = self.delegation(question, background)
                self.notable.setdefault("delegation_reply", reply["id"])
                if background:
                    reply = self.background_result(reply, child)
            elif compare:
                self.assistant(self.main, question, self.budget(), model=MODELS[1], modelIdx=1)
                reply = self.assistant(
                    self.main, question, self.budget(), model=MODELS[0], modelIdx=0
                )
                self.notable.setdefault("compare_question", question["id"])
            else:
                if rng.random() < 0.12:
                    # regenerations: the earlier tries stay as siblings
                    for _ in range(rng.randint(1, 3)):
                        self.assistant(self.main, question, self.budget(), model=rng.choice(MODELS))
                    self.notable.setdefault("regenerated_question", question["id"])
                reply = self.assistant(self.main, question, self.budget())
            if not by_agent and rng.random() < 0.05:
                self.branch(reply, rng.randint(1, 3))
                branched = True
            parent = reply
            turn += 1
        self.notable["last_reply"] = parent["id"]
        chats = [self.main.record(parent["id"], ["mock-model"], START, self.clock)]
        for child in self.subagents:
            chats.append(child.record(child.current_id, ["mock-model"], START, self.clock))
        return chats

    def background_result(self, reply: dict, child: Chat) -> dict:
        """The background sub-agent's result, delivered as an internal message, and the answer."""
        meta = {
            "internal": True,
            "type": "subagent",
            "delegation_id": child.meta["delegation_id"],
            "subagent_chat_id": child.id,
        }
        content = "\n".join(
            [
                f"[ASYNC SUBAGENT COMPLETE - {child.meta['delegation_id']}]",
                "A background subagent you dispatched earlier has finished.",
                "",
                f"Subagent chat: {child.id}",
                "Status: completed   Duration: 41.2s",
                "--- RESULT ---",
                child.messages[child.current_id]["content"],
            ]
        )
        self.size += len(content)
        result = self.main.add(
            {
                "id": self.w.uuid(),
                "parentId": reply["id"],
                "childrenIds": [],
                "role": "user",
                "content": content,
                "model": "mock-model",
                "meta": meta,
                "timestamp": self.tick(),
                "user_id": OWNER_ID,
            }
        )
        self.notable["subagent_result"] = result["id"]
        return self.assistant(self.main, result, self.budget())


def manifest(name: str, seed: int, chats: list[dict], notable: dict, size: int) -> dict:
    main = chats[0]
    messages = main["chat"]["history"]["messages"]
    path, current = [], main["current_message_id"]
    while current:
        path.append(current)
        current = messages[current]["parentId"]
    largest = max(
        (m for m in messages.values() if m["role"] == "assistant"),
        key=lambda m: len(json.dumps(m.get("output"))),
    )
    return {
        "name": name,
        "seed": seed,
        "main_chat_id": main["id"],
        "subagent_chat_ids": [chat["id"] for chat in chats[1:]],
        "title": main["title"],
        "messages": len(messages),
        "current_path": len(path),
        "branch_points": sum(len(m["childrenIds"]) > 1 for m in messages.values()),
        "agent_replies": sum("agent" in (m.get("meta") or {}) for m in messages.values()),
        "content_bytes": size,
        "search_term": SEARCH_TERM,
        "largest_reply": largest["id"],
        **notable,
    }


def generate(name: str) -> Path:
    seed, target, per_turn = DATA_SETS[name]
    goal = target
    for _ in range(5):
        # the stored size runs ahead of the text counted while growing it; aim again until close
        conversation = Conversation(Writer(seed), goal, per_turn)
        chats = conversation.build()
        body = json.dumps(chats, separators=(",", ":"))
        if abs(len(body) - target) < target * 0.05:
            break
        goal = int(goal * target / len(body))
    data = {
        "format": FORMAT,
        "manifest": manifest(name, seed, chats, conversation.notable, len(body)),
        "chats": chats,
    }
    encoded = json.dumps(data, separators=(",", ":")).encode()
    path = DATA_DIR / f"{name}.json.xz"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(lzma.compress(encoded, preset=9))
    print(
        f"{name}: {len(encoded):,} bytes, {path.stat().st_size:,} packed, "
        f"{data['manifest']['messages']} messages"
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help=f"any of {', '.join(DATA_SETS)}; all by default")
    names = parser.parse_args().names or list(DATA_SETS)
    unknown = set(names) - set(DATA_SETS)
    if unknown:
        parser.error(f"no such data set: {', '.join(sorted(unknown))}")
    for name in names:
        generate(name)


if __name__ == "__main__":
    main()
