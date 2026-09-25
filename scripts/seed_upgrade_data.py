"""Fill an older Open WebUI release with data through its own API and save the result.

The upgrade tests in `integration/migrations/test_upgrade_from_release.py` boot the checkout
under test on these saved data directories. For each release this archives the tagged backend
out of an Open WebUI clone, boots it in this environment on an empty data directory (on SQLite,
and on an embedded Postgres when `pgserver` is installed), creates accounts, groups, chats,
notes, a knowledge base, workspace items, memories, a channel, feedback and changed settings,
then stops it and writes `<release>-<engine>.tar.gz` with a `<release>-<engine>.json` manifest
of what was created. Pinned packages an old release needs and this environment lacks go into a
scratch directory on that release's `PYTHONPATH`, never into the environment itself.

    python scripts/seed_upgrade_data.py --clone ../open-webui v0.9.6 v0.10.2 v0.11.4
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterator

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness import upstream as upstream_module  # noqa: E402
from harness.chat import send_message, wait_for_reply  # noqa: E402
from harness.prepared_data import serving  # noqa: E402
from harness.upstream import MOCK_MODEL_ID  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "integration" / "migrations" / "upgrade_data"

# only what a release imports at boot; the rest of its requirements load lazily
BOOT_PACKAGES = ("fpdf2", "fonttools")

ACCOUNTS = {
    "admin": {"name": "Ada Admin", "email": "admin@example.com", "password": "admin-pass-1"},
    "alice": {"name": "Alice Author", "email": "alice@example.com", "password": "alice-pass-1"},
    "bob": {"name": "Bob Reader", "email": "bob@example.com", "password": "bob-pass-1"},
    "carol": {"name": "Carol Outsider", "email": "carol@example.com", "password": "carol-pass-1"},
}

TOOL_SOURCE = '''"""
title: Weather
"""


from pydantic import BaseModel


class Tools:
    class Valves(BaseModel):
        units: str = "metric"

    def __init__(self):
        self.valves = self.Valves()

    def get_weather(self, city: str) -> str:
        """Report the weather in a city."""
        return f"Sunny in {city}"
'''

FUNCTION_SOURCE = '''"""
title: Shout filter
"""

from pydantic import BaseModel


class Filter:
    class Valves(BaseModel):
        suffix: str = "!"

    def __init__(self):
        self.valves = self.Valves()

    def inlet(self, body: dict) -> dict:
        return body
'''

KNOWLEDGE_TEXT = "The upgrade handbook says the lighthouse key hangs behind the blue door."
CHAT_FILE_TEXT = "Packing list: umbrella, passport, a very old map of Vienna."


def _ensure(response: httpx.Response, what: str) -> dict | list:
    if response.status_code != 200:
        raise SystemExit(f"{what} failed: HTTP {response.status_code} {response.text[:500]}")
    return response.json()


def _grant(principal_type: str, principal_id: str, permission: str = "read") -> dict:
    return {
        "principal_type": principal_type,
        "principal_id": principal_id,
        "permission": permission,
    }


def archived_backend(clone: Path, release: str, into: Path) -> Path:
    """The release's backend, with the changelog its version module reads at import."""
    archive = subprocess.run(
        ["git", "-C", str(clone), "archive", release, "backend", "CHANGELOG.md", "package.json"],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(into, filter="data")
    shutil.copy(into / "CHANGELOG.md", into / "backend" / "open_webui" / "CHANGELOG.md")
    return into / "backend"


def boot_overlay(backend: Path, into: Path) -> Path:
    """A directory with the release's pins of `BOOT_PACKAGES` this environment lacks."""
    pins = {}
    for line in (backend / "requirements.txt").read_text().splitlines():
        requirement = line.split("#")[0].strip()
        name = requirement.split("=")[0].split("~")[0].split(">")[0].strip()
        if name in BOOT_PACKAGES:
            pins[name] = requirement
    missing = []
    for name in BOOT_PACKAGES:
        try:
            version(name)
        except PackageNotFoundError:
            missing.append(pins.get(name, name))
    into.mkdir(parents=True, exist_ok=True)
    if missing:
        subprocess.run(
            ["uv", "pip", "install", "--python", sys.executable, "--target", str(into)]
            + ["--no-deps", *missing],
            check=True,
        )
    return into


@contextlib.contextmanager
def postgres(root: Path) -> Iterator[tuple[str, Path]]:
    """An embedded Postgres; yields its SQLAlchemy URL and the directory of its binaries."""
    import pgserver

    server = pgserver.get_server(str(root / "pgdata"), cleanup_mode=None)
    try:
        url = server.get_uri().replace("postgresql://", "postgresql+psycopg2://", 1)
        yield url, Path(pgserver.__file__).parent / "pginstall" / "bin"
    finally:
        server.cleanup()


class Seeder:
    """Creates the data set on a running release and records it for the manifest."""

    def __init__(self, base_url: str, provider: upstream_module.MockUpstream):
        self.base_url = base_url
        self.provider = provider
        self.tokens: dict[str, str] = {}
        self.manifest: dict = {"accounts": {}}

    def client(self, who: str) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.tokens[who]}"},
            timeout=120.0,
        )

    def run(self) -> dict:
        self.accounts()
        self.group()
        self.settings()
        self.workspace()
        self.knowledge()
        self.chats()
        self.notes_and_memories()
        self.channel()
        return self.manifest

    def accounts(self) -> None:
        admin = ACCOUNTS["admin"]
        signup = httpx.post(f"{self.base_url}/api/v1/auths/signup", json=admin, timeout=60)
        body = _ensure(signup, "admin signup")
        self.tokens["admin"] = body["token"]
        self.manifest["accounts"]["admin"] = {**admin, "id": body["id"], "role": "admin"}
        with self.client("admin") as client:
            for who in ("alice", "bob", "carol"):
                account = ACCOUNTS[who]
                added = _ensure(
                    client.post("/api/v1/auths/add", json={**account, "role": "user"}),
                    f"adding {who}",
                )
                self.tokens[who] = added["token"]
                self.manifest["accounts"][who] = {**account, "id": added["id"], "role": "user"}

    def _id(self, who: str) -> str:
        return self.manifest["accounts"][who]["id"]

    def group(self) -> None:
        with self.client("admin") as client:
            group = _ensure(
                client.post(
                    "/api/v1/groups/create",
                    json={"name": "Research", "description": "People who read the handbook"},
                ),
                "creating the group",
            )
            members = [self._id("alice"), self._id("bob")]
            _ensure(
                client.post(
                    f"/api/v1/groups/id/{group['id']}/users/add", json={"user_ids": members}
                ),
                "adding group members",
            )
        self.manifest["group"] = {
            "id": group["id"],
            "name": "Research",
            "members": ["alice", "bob"],
        }

    def settings(self) -> None:
        with self.client("admin") as client:
            config = _ensure(client.get("/api/v1/auths/admin/config"), "reading the admin config")
            config.update(
                {
                    "DEFAULT_USER_ROLE": "user",
                    "ENABLE_COMMUNITY_SHARING": False,
                    "ENABLE_CHANNELS": True,
                }
            )
            config["WEBUI_URL"] = "https://chat.example.com"
            _ensure(client.post("/api/v1/auths/admin/config", json=config), "saving admin config")

            permissions = _ensure(client.get("/api/v1/users/default/permissions"), "permissions")
            permissions["chat"]["delete"] = False
            permissions["workspace"]["prompts"] = True
            _ensure(
                client.post("/api/v1/users/default/permissions", json=permissions),
                "saving default permissions",
            )
            banner = {
                "id": "upgrade-banner",
                "type": "info",
                "title": "Maintenance",
                "content": "Maintenance on Sunday",
                "dismissible": True,
                "timestamp": 1700000000,
            }
            _ensure(client.post("/api/v1/configs/banners", json={"banners": [banner]}), "banners")
        with self.client("alice") as client:
            user_settings = {"ui": {"theme": "dark", "chatBubble": False, "notesSeen": 3}}
            _ensure(
                client.post("/api/v1/users/user/settings/update", json=user_settings),
                "saving alice's settings",
            )
        self.manifest["settings"] = {
            "admin": {
                "DEFAULT_USER_ROLE": "user",
                "ENABLE_COMMUNITY_SHARING": False,
                "ENABLE_CHANNELS": True,
                "WEBUI_URL": "https://chat.example.com",
            },
            "permissions": {"chat.delete": False, "workspace.prompts": True},
            "banner": banner,
            "alice_ui": user_settings["ui"],
        }

    def workspace(self) -> None:
        group_reads = _grant("group", self.manifest["group"]["id"])
        with self.client("admin") as client:
            _ensure(
                client.post(
                    "/api/v1/prompts/create",
                    json={
                        "command": "summarize",
                        "name": "Summarize",
                        "content": "Summarize the following text in three bullet points.",
                        "access_grants": [group_reads],
                    },
                ),
                "creating the shared prompt",
            )
            _ensure(
                client.post(
                    "/api/v1/tools/create",
                    json={
                        "id": "weather_tool",
                        "name": "Weather",
                        "content": TOOL_SOURCE,
                        "meta": {"description": "Weather lookups"},
                        "access_grants": [group_reads],
                    },
                ),
                "creating the tool",
            )
            _ensure(
                client.post(
                    "/api/v1/functions/create",
                    json={
                        "id": "shout_filter",
                        "name": "Shout filter",
                        "content": FUNCTION_SOURCE,
                        "meta": {"description": "Adds emphasis"},
                    },
                ),
                "creating the function",
            )
            _ensure(client.post("/api/v1/functions/id/shout_filter/toggle"), "activating it")
            _ensure(client.post("/api/v1/functions/id/shout_filter/toggle/global"), "globalising")
            _ensure(
                client.post(
                    "/api/v1/functions/id/shout_filter/valves/update", json={"suffix": "!!!"}
                ),
                "setting the function's valves",
            )
            _ensure(
                client.post(
                    "/api/v1/models/create",
                    json={
                        "id": "research-assistant",
                        "base_model_id": MOCK_MODEL_ID,
                        "name": "Research assistant",
                        "meta": {"description": "Answers from the handbook"},
                        "params": {"system": "You answer from the handbook.", "temperature": 0.2},
                        "access_grants": [_grant("user", self._id("bob"))],
                    },
                ),
                "creating the model preset",
            )
        with self.client("alice") as client:
            _ensure(
                client.post(
                    "/api/v1/prompts/create",
                    json={
                        "command": "alice-draft",
                        "name": "Alice's draft",
                        "content": "Draft a polite reply to this email.",
                        "access_grants": [],
                    },
                ),
                "creating alice's private prompt",
            )
        self.manifest["prompts"] = {
            "summarize": {
                "owner": "admin",
                "name": "Summarize",
                "content": "Summarize the following text in three bullet points.",
                "readers": ["alice", "bob"],
            },
            "alice-draft": {
                "owner": "alice",
                "name": "Alice's draft",
                "content": "Draft a polite reply to this email.",
                "readers": [],
            },
        }
        self.manifest["tool"] = {
            "id": "weather_tool",
            "name": "Weather",
            "readers": ["alice", "bob"],
        }
        self.manifest["function"] = {
            "id": "shout_filter",
            "name": "Shout filter",
            "valves": {"suffix": "!!!"},
        }
        self.manifest["model"] = {
            "id": "research-assistant",
            "name": "Research assistant",
            "system": "You answer from the handbook.",
            "readers": ["bob"],
        }

    def _upload(self, client: httpx.Client, filename: str, text: str) -> str:
        uploaded = client.post(
            "/api/v1/files/",
            params={"process": "true", "process_in_background": "false"},
            files={"file": (filename, text.encode(), "text/plain")},
        )
        return _ensure(uploaded, f"uploading {filename}")["id"]

    def knowledge(self) -> None:
        with self.client("admin") as client:
            knowledge = _ensure(
                client.post(
                    "/api/v1/knowledge/create",
                    json={
                        "name": "Handbook",
                        "description": "Where things are",
                        "access_grants": [_grant("group", self.manifest["group"]["id"])],
                    },
                ),
                "creating the knowledge base",
            )
            file_id = self._upload(client, "handbook.txt", KNOWLEDGE_TEXT)
            _ensure(
                client.post(
                    f"/api/v1/knowledge/{knowledge['id']}/file/add", json={"file_id": file_id}
                ),
                "adding the file to the knowledge base",
            )
        self.manifest["knowledge"] = {
            "id": knowledge["id"],
            "name": "Handbook",
            "file_id": file_id,
            "filename": "handbook.txt",
            "text": KNOWLEDGE_TEXT,
            "readers": ["alice", "bob"],
        }

    def chats(self) -> None:
        with self.client("alice") as client:
            folder = _ensure(client.post("/api/v1/folders/", json={"name": "Trips"}), "folder")
            subfolder = _ensure(
                client.post("/api/v1/folders/", json={"name": "Vienna", "parent_id": folder["id"]}),
                "subfolder",
            )
            file_id = self._upload(client, "packing.txt", CHAT_FILE_TEXT)
            branched = self._branched_chat(client, file_id)
            _ensure(
                client.post(
                    f"/api/v1/chats/{branched['id']}/folder", json={"folder_id": subfolder["id"]}
                ),
                "moving the chat into the folder",
            )
            _ensure(client.post(f"/api/v1/chats/{branched['id']}/pin"), "pinning the chat")
            _ensure(
                client.post(f"/api/v1/chats/{branched['id']}/tags", json={"name": "travel"}),
                "tagging the chat",
            )
            shared = _ensure(client.post(f"/api/v1/chats/{branched['id']}/share"), "sharing")
            _ensure(
                client.post(
                    f"/api/v1/chats/shared/{branched['id']}/access/update",
                    json={"access_grants": [_grant("user", self._id("bob"))]},
                ),
                "granting bob the shared chat",
            )
            live = self._live_chat(client)
            archived = _ensure(
                client.post(
                    "/api/v1/chats/new",
                    json={"chat": {"title": "Old plans", "models": [MOCK_MODEL_ID], "history": {}}},
                ),
                "creating the archived chat",
            )
            _ensure(client.post(f"/api/v1/chats/{archived['id']}/archive"), "archiving")
            feedback = _ensure(
                client.post(
                    "/api/v1/evaluations/feedback",
                    json={
                        "type": "rating",
                        "data": {"rating": 1, "model_id": MOCK_MODEL_ID, "reason": "Accurate"},
                        "meta": {
                            "chat_id": live["id"],
                            "message_id": live["reply_id"],
                        },
                        "snapshot": {"chat": {}},
                    },
                ),
                "leaving feedback",
            )
        self.manifest["folders"] = {
            "parent": {"id": folder["id"], "name": "Trips"},
            "child": {"id": subfolder["id"], "name": "Vienna"},
        }
        self.manifest["chats"] = {
            "branched": {**branched, "folder_id": subfolder["id"], "share_id": shared["share_id"]},
            "live": live,
            "archived": {"id": archived["id"], "title": "Old plans"},
        }
        self.manifest["chat_file"] = {
            "id": file_id,
            "filename": "packing.txt",
            "text": CHAT_FILE_TEXT,
        }
        self.manifest["feedback"] = {"id": feedback["id"], "reason": "Accurate", "rating": 1}

    def _branched_chat(self, client: httpx.Client, file_id: str) -> dict:
        """A chat as the web client saves it: a regenerated reply and a tool call."""
        now = int(time.time())
        ids = {
            name: f"{name}-0000-4000-8000-000000000000" for name in ("u1", "a1", "a1b", "u2", "a2")
        }
        tool_call = (
            '<details type="tool_calls" done="true" id="call_1" name="get_weather" '
            'arguments="{&quot;city&quot;: &quot;Vienna&quot;}" '
            'result="&quot;Sunny in Vienna&quot;">'
            "\n<summary>Tool Executed</summary>\n</details>\n"
        )
        attached = {"type": "file", "id": file_id, "name": "packing.txt", "status": "uploaded"}
        messages = {
            ids["u1"]: {
                "id": ids["u1"],
                "parentId": None,
                "childrenIds": [ids["a1"], ids["a1b"]],
                "role": "user",
                "content": "What should I pack for Vienna?",
                "files": [attached],
                "timestamp": now,
                "models": [MOCK_MODEL_ID],
            },
            ids["a1"]: {
                "id": ids["a1"],
                "parentId": ids["u1"],
                "childrenIds": [],
                "role": "assistant",
                "content": "Pack an umbrella.",
                "model": MOCK_MODEL_ID,
                "done": True,
                "timestamp": now + 1,
            },
            ids["a1b"]: {
                "id": ids["a1b"],
                "parentId": ids["u1"],
                "childrenIds": [ids["u2"]],
                "role": "assistant",
                "content": tool_call + "It is sunny, so pack sunglasses.",
                "model": MOCK_MODEL_ID,
                "done": True,
                "timestamp": now + 2,
            },
            ids["u2"]: {
                "id": ids["u2"],
                "parentId": ids["a1b"],
                "childrenIds": [ids["a2"]],
                "role": "user",
                "content": "And for the evening?",
                "timestamp": now + 3,
                "models": [MOCK_MODEL_ID],
            },
            ids["a2"]: {
                "id": ids["a2"],
                "parentId": ids["u2"],
                "childrenIds": [],
                "role": "assistant",
                "content": "A light jacket.",
                "model": MOCK_MODEL_ID,
                "done": True,
                "timestamp": now + 4,
            },
        }
        chat = {
            "title": "Packing for Vienna",
            "models": [MOCK_MODEL_ID],
            "history": {"messages": messages, "currentId": ids["a2"]},
            "messages": [messages[ids[name]] for name in ("u1", "a1b", "u2", "a2")],
            "files": [attached],
            "params": {},
        }
        created = _ensure(client.post("/api/v1/chats/new", json={"chat": chat}), "saving the chat")
        return {
            "id": created["id"],
            "title": "Packing for Vienna",
            "current_id": ids["a2"],
            "messages": {
                message_id: {
                    "parent": message["parentId"],
                    "children": message["childrenIds"],
                    "role": message["role"],
                    "content": message["content"],
                }
                for message_id, message in messages.items()
            },
        }

    def _live_chat(self, client: httpx.Client) -> dict:
        """A chat the server itself wrote while streaming the provider's replies."""
        _ensure(client.get("/api/models", params={"refresh": "true"}), "listing models")
        self.provider.queue(upstream_module.text("Schnitzel is a breaded cutlet."))
        first = send_message(client, "What is a Schnitzel?")
        first_reply = wait_for_reply(client, first)
        self.provider.queue(upstream_module.text("Try it with lemon."))
        history = [
            {"role": "user", "content": "What is a Schnitzel?"},
            {"role": "assistant", "content": first_reply["content"]},
        ]
        second = send_message(
            client,
            "How do I eat it?",
            chat_id=first.chat_id,
            parent_id=first.assistant_message_id,
            history=history,
        )
        wait_for_reply(client, second)
        client.post(
            f"/api/v1/chats/{first.chat_id}", json={"chat": {"title": "Schnitzel questions"}}
        )
        stored = _ensure(client.get(f"/api/v1/chats/{first.chat_id}"), "reading the live chat")
        messages = stored["chat"]["history"]["messages"]
        return {
            "id": first.chat_id,
            "title": stored["title"],
            "current_id": stored["chat"]["history"]["currentId"],
            "reply_id": first.assistant_message_id,
            "messages": {
                message_id: {
                    "parent": message.get("parentId"),
                    "children": message.get("childrenIds", []),
                    "role": message["role"],
                    "content": message["content"],
                }
                for message_id, message in messages.items()
            },
        }

    def notes_and_memories(self) -> None:
        body = "# Groceries\n\n- apples\n- *dark* chocolate"
        with self.client("alice") as client:
            note = _ensure(
                client.post(
                    "/api/v1/notes/create",
                    json={
                        "title": "Groceries",
                        "data": {"content": {"md": body, "html": "", "json": None}},
                        # the share dialog grants read along with write
                        "access_grants": [
                            _grant("user", self._id("bob")),
                            _grant("user", self._id("bob"), "write"),
                        ],
                    },
                ),
                "creating the note",
            )
            memories = ["Alice is allergic to peanuts.", "Alice prefers window seats."]
            for memory in memories:
                _ensure(client.post("/api/v1/memories/add", json={"content": memory}), "memory")
        self.manifest["note"] = {
            "id": note["id"],
            "title": "Groceries",
            "md": body,
            "writers": ["bob"],
        }
        self.manifest["memories"] = {"owner": "alice", "contents": memories}

    def channel(self) -> None:
        group_id = self.manifest["group"]["id"]
        grants = [_grant("group", group_id), _grant("group", group_id, "write")]
        with self.client("admin") as client:
            channel = _ensure(
                client.post(
                    "/api/v1/channels/create",
                    json={
                        "name": "lighthouse",
                        "description": "Team chatter",
                        "access_grants": grants,
                    },
                ),
                "creating the channel",
            )
        with self.client("alice") as client:
            posted = _ensure(
                client.post(
                    f"/api/v1/channels/{channel['id']}/messages/post",
                    json={"content": "Who has the lighthouse key?"},
                ),
                "posting to the channel",
            )
        with self.client("bob") as client:
            answer = _ensure(
                client.post(
                    f"/api/v1/channels/{channel['id']}/messages/post",
                    json={"content": "Behind the blue door.", "parent_id": posted["id"]},
                ),
                "replying in the thread",
            )
            _ensure(
                client.post(
                    f"/api/v1/channels/{channel['id']}/messages/{posted['id']}/reactions/add",
                    json={"name": "thumbsup"},
                ),
                "reacting",
            )
        self.manifest["channel"] = {
            "id": channel["id"],
            "name": "lighthouse",
            "readers": ["alice", "bob"],
            "message": {"id": posted["id"], "content": "Who has the lighthouse key?"},
            "reply": {"id": answer["id"], "content": "Behind the blue door."},
            "reaction": "thumbsup",
        }


def compact(database: Path) -> None:
    """Fold the write-ahead log into the file and shrink it, so the file alone is the data."""
    with contextlib.closing(sqlite3.connect(database)) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("VACUUM")


def seed(clone: Path, release: str, engine: str, work: Path) -> None:
    backend = archived_backend(clone, release, work / "source")
    overlay = boot_overlay(backend, work / "overlay")
    data_dir = work / "data"
    data_dir.mkdir()
    with contextlib.ExitStack() as stack:
        provider, shutdown = upstream_module.serve()
        stack.callback(shutdown)
        settings = {
            "PYTHONPATH": str(overlay),
            "WEBUI_AUTH": "true",
            "ENABLE_OPENAI_API": "true",
            "OPENAI_API_BASE_URL": provider.base_url,
            "OPENAI_API_KEY": "sk-mock",
            "OPENAI_API_BASE_URLS": provider.base_url,
            "OPENAI_API_KEYS": "sk-mock",
            "RAG_EMBEDDING_ENGINE": "openai",
            "RAG_OPENAI_API_BASE_URL": provider.base_url,
            "RAG_OPENAI_API_KEY": "sk-mock",
            "BYPASS_MODEL_ACCESS_CONTROL": "true",
            "ENABLE_TITLE_GENERATION": "false",
            "ENABLE_TAGS_GENERATION": "false",
            "ENABLE_FOLLOW_UP_GENERATION": "false",
        }
        dump_with = None
        if engine == "postgres":
            url, binaries = stack.enter_context(postgres(work))
            settings["DATABASE_URL"] = url
            dump_with = (binaries / "pg_dump", url.replace("+psycopg2", ""))
        with serving(data_dir, settings, backend=backend) as running:
            manifest = Seeder(running.base_url, provider).run()
        if dump_with:
            pg_dump, url = dump_with
            dump = subprocess.run(
                [str(pg_dump), "--no-owner", "--no-privileges", url],
                check=True,
                capture_output=True,
            ).stdout
            (data_dir / "webui.sql").write_bytes(dump)
    for database in [*data_dir.rglob("*.db"), *data_dir.rglob("*.sqlite3")]:
        compact(database)
    # uploads are stored by absolute path, which the tests point at their own copy
    manifest = {"release": release, "engine": engine, "data_dir": str(data_dir), **manifest}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{release}-{engine}"
    with tarfile.open(OUTPUT_DIR / f"{name}.tar.gz", "w:gz") as tar:
        for path in sorted(data_dir.rglob("*")):
            if path.is_file() and not path.name.endswith(("-wal", "-shm", "-journal")):
                tar.add(path, arcname=str(path.relative_to(data_dir)), recursive=False)
    (OUTPUT_DIR / f"{name}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{name}: {(OUTPUT_DIR / f'{name}.tar.gz').stat().st_size // 1024} KiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clone", type=Path, required=True, help="an Open WebUI git clone")
    parser.add_argument("--engine", choices=["sqlite", "postgres", "both"], default="both")
    parser.add_argument("releases", nargs="+", help="release tags, e.g. v0.11.4")
    options = parser.parse_args()
    engines = ["sqlite", "postgres"] if options.engine == "both" else [options.engine]
    if "postgres" in engines and importlib.util.find_spec("pgserver") is None:
        raise SystemExit("the Postgres data sets need pgserver; pass --engine sqlite without it")
    for release in options.releases:
        for engine in engines:
            with tempfile.TemporaryDirectory(prefix=f"seed-{release}-") as work:
                seed(options.clone.resolve(), release, engine, Path(work))


if __name__ == "__main__":
    main()
