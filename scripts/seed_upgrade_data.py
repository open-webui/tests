"""Fill an older Open WebUI release with data through its own API and save the result.

The upgrade tests in `integration/migrations/test_upgrade_from_release.py` boot the checkout
under test on these saved data directories. For each release this archives the tagged backend
out of an Open WebUI clone, boots it in this environment on an empty data directory (on SQLite,
and on an embedded Postgres when `pgserver` is installed), creates accounts, groups, chats,
notes, a knowledge base, workspace items, memories, a channel, feedback and changed settings,
then stops it and writes `<release>-<engine>.tar.gz` with a `<release>-<engine>.json` manifest
of what was created. Pinned packages an old release needs and this environment lacks go into a
scratch directory on that release's `PYTHONPATH`, never into the environment itself. From v0.13
on it also sends chats the way that release's web client does and records every stored message
whole: a tool call with its output, reasoning and usage, a reply edited and saved, three
answers to one question with the middle one deleted, two models side by side, a delegation with
the sub-agent's own chat, a chat started inside a folder and a share others may continue.

    python scripts/seed_upgrade_data.py --clone ../open-webui v0.9.6 v0.10.2 v0.11.4 v0.13.4
"""

from __future__ import annotations

import argparse
import contextlib
import copy
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
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterator

import httpx
import sqlalchemy

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness import upstream as upstream_module  # noqa: E402
from harness.chat import NO_BACKGROUND_TASKS, ChatTurn, send_message, wait_for_reply  # noqa: E402
from harness.prepared_data import serving  # noqa: E402
from harness.upstream import MOCK_MODEL_ID  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "integration" / "migrations" / "upgrade_data"

# only what a release imports at boot; the rest of its requirements load lazily
BOOT_PACKAGES = ("fpdf2", "fonttools")
# newer releases of these drop what an older Open WebUI imports, so the release's pin wins
PINNED_BOOT_PACKAGES = ("mcp",)

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

# what the group's members may do beyond the defaults
GROUP_PERMISSIONS = {"workspace": {"models": True}}

KNOWLEDGE_TEXT = "The upgrade handbook says the lighthouse key hangs behind the blue door."

TRIP_SKILL = """---
name: trip-planner
description: Plans multi-city trips by train.
---

# Trip planner

1. Ask for the cities and the dates.
2. Prefer night trains over flights.
3. Keep every transfer above 40 minutes.
"""
# what alice changes the trip planner to, so its last change is later than its creation
TRIP_SKILL_EDITED = TRIP_SKILL + "4. Name the platform for every departure.\n"
TRANSLATOR_SKILL = (
    "# Übersetzer \U0001f30d\r\n"
    "\r\n"
    "Übersetze höflich: «Grüß Gott» → 「こんにちは」, 你好, Привет, مرحبا, שלום.\r\n"
    "\tKeep combining marks (e\u0301, n\u0303) "
    "and emoji with modifiers (\U0001f44b\U0001f3fd).  \r\n"
    "Math stays: ∑ x² ≤ ∞; quotes stay: ‘single’ “double” `<$PATH>`.\r\n"
)
CHAT_FILE_TEXT = "Packing list: umbrella, passport, a very old map of Vienna."


# what alice shares the moment she creates it
SHARED_ON_CREATE = ("skills", "notes")
USAGE = {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49}


def writes_output_items(release: str) -> bool:
    """From v0.13 on replies are stored as typed output items; sub-agents and share modes exist."""
    major, minor = (int(part) for part in release.lstrip("v").split(".")[:2])
    return (major, minor) >= (0, 13)


def _message_path(messages: dict, leaf_id: str) -> list[dict]:
    path = []
    while leaf_id:
        path.insert(0, messages[leaf_id])
        leaf_id = messages[leaf_id].get("parentId")
    return path


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
        ["git", "-C", str(clone), "archive", release, "backend", "static", "CHANGELOG.md"]
        + ["package.json"],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(into, filter="data")
    shutil.copy(into / "CHANGELOG.md", into / "backend" / "open_webui" / "CHANGELOG.md")
    return into / "backend"


def boot_overlay(backend: Path, into: Path) -> Path:
    """A directory with the release's pins of `BOOT_PACKAGES` this environment lacks and of
    `PINNED_BOOT_PACKAGES`."""
    pins = {}
    for line in (backend / "requirements.txt").read_text().splitlines():
        requirement = line.split("#")[0].strip()
        name = requirement.split("=")[0].split("~")[0].split(">")[0].strip()
        if name in (*BOOT_PACKAGES, *PINNED_BOOT_PACKAGES):
            pins[name] = requirement
    missing = [pins[name] for name in PINNED_BOOT_PACKAGES if name in pins]
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

    def __init__(
        self,
        base_url: str,
        provider: upstream_module.MockUpstream,
        release: str,
        database_url: str,
    ):
        self.base_url = base_url
        self.database_url = database_url
        self.provider = provider
        self.release = release
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
        self.skills()
        self.knowledge()
        self.chats()
        if writes_output_items(self.release):
            self.output_item_chats()
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
            _ensure(
                client.post(
                    f"/api/v1/groups/id/{group['id']}/update",
                    json={
                        "name": "Research",
                        "description": "People who read the handbook",
                        "permissions": GROUP_PERMISSIONS,
                    },
                ),
                "giving the group its permissions",
            )
        self.manifest["group"] = {
            "id": group["id"],
            "name": "Research",
            "members": ["alice", "bob"],
            "permissions": GROUP_PERMISSIONS,
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
            permissions["workspace"]["skills"] = True
            # since v0.13 sharing on creation needs these; earlier releases share without them
            shares = SHARED_ON_CREATE if writes_output_items(self.release) else ()
            permissions.setdefault("sharing", {}).update(dict.fromkeys(shares, True))
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
        with self.client("bob") as client:
            bob_settings = {"ui": {"models": ["research-assistant"]}}
            _ensure(
                client.post("/api/v1/users/user/settings/update", json=bob_settings),
                "saving bob's default model",
            )
        self.manifest["settings"] = {
            "admin": {
                "DEFAULT_USER_ROLE": "user",
                "ENABLE_COMMUNITY_SHARING": False,
                "ENABLE_CHANNELS": True,
                "WEBUI_URL": "https://chat.example.com",
            },
            "permissions": {
                "chat.delete": False,
                "workspace.prompts": True,
                "workspace.skills": True,
                **{f"sharing.{key}": True for key in shares},
            },
            "banner": banner,
            "alice_ui": user_settings["ui"],
            "bob_ui": bob_settings["ui"],
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

    def _create_skill(self, who: str, form: dict) -> None:
        with self.client(who) as client:
            _ensure(client.post("/api/v1/skills/create", json=form), f"creating {form['id']}")

    def skills(self) -> None:
        """Skills of every shape an install holds, read back as the release stored them."""
        group_id = self.manifest["group"]["id"]
        self._create_skill(
            "alice",
            {
                "id": "trip-planner",
                "name": "Trip planner",
                "description": "Plans multi-city trips by train.",
                "content": TRIP_SKILL,
                "meta": {"tags": ["travel"]},
                "access_grants": [
                    _grant("group", group_id),
                    _grant("user", self._id("bob"), "write"),
                ],
            },
        )
        time.sleep(1.1)
        with self.client("alice") as client:
            _ensure(
                client.post(
                    "/api/v1/skills/id/trip-planner/update",
                    json={
                        "id": "trip-planner",
                        "name": "Trip planner",
                        "description": "Plans multi-city trips by train.",
                        "content": TRIP_SKILL_EDITED,
                        "meta": {"tags": ["travel"]},
                    },
                ),
                "editing the trip planner",
            )
        self._create_skill(
            "alice",
            {
                "id": "uebersetzer",
                "name": "Übersetzer 日本語 \U0001f680",
                "description": "",
                "content": TRANSLATOR_SKILL,
                "access_grants": [],
            },
        )
        # no description and no meta, then switched off
        self._create_skill(
            "bob", {"id": "old-habits", "name": "Old habits", "content": "Answer in Latin."}
        )
        with self.client("bob") as client:
            _ensure(client.post("/api/v1/skills/id/old-habits/toggle"), "switching off old-habits")
        self._create_skill(
            "admin",
            {
                "id": "house-style",
                "name": "House style",
                "description": "How every answer is formatted.",
                "content": "Use sentence case headings and no exclamation marks.",
                "access_grants": [_grant("user", "*")],
            },
        )
        # an account that leaves: its skill stays, owned by an id no account has any more
        dave = {"name": "Dave Departed", "email": "dave@example.com", "password": "dave-pass-1"}
        with self.client("admin") as client:
            added = _ensure(
                client.post("/api/v1/auths/add", json={**dave, "role": "user"}), "adding dave"
            )
        self.tokens["dave"] = added["token"]
        self._create_skill(
            "dave",
            {
                "id": "dave-notes",
                "name": "Dave's notes",
                "description": "Left behind.",
                "content": "Hand over to the next shift at six.",
                "access_grants": [_grant("group", group_id)],
            },
        )
        with self.client("admin") as client:
            deleted = _ensure(client.delete(f"/api/v1/users/{added['id']}"), "deleting dave")
            assert deleted is True, deleted
            stored = {}
            for skill_id in (
                "trip-planner",
                "uebersetzer",
                "old-habits",
                "house-style",
                "dave-notes",
            ):
                skill = _ensure(client.get(f"/api/v1/skills/id/{skill_id}"), f"reading {skill_id}")
                stored[skill_id] = {
                    key: skill[key]
                    for key in (
                        "id",
                        "user_id",
                        "name",
                        "description",
                        "content",
                        "meta",
                        "is_active",
                    )
                }
                stored[skill_id]["created_at"] = skill["created_at"]
                stored[skill_id]["updated_at"] = skill["updated_at"]
                stored[skill_id]["grants"] = sorted(
                    (
                        {
                            key: grant[key]
                            for key in ("principal_type", "principal_id", "permission")
                        }
                        for grant in skill.get("access_grants", [])
                    ),
                    key=lambda grant: (
                        grant["principal_type"],
                        grant["principal_id"],
                        grant["permission"],
                    ),
                )
        expected_grants = {
            "trip-planner": 2,
            "uebersetzer": 0,
            "old-habits": 0,
            "house-style": 1,
            "dave-notes": 1,
        }
        for skill_id, count in expected_grants.items():
            if len(stored[skill_id]["grants"]) != count:
                raise SystemExit(f"{skill_id} was stored with grants {stored[skill_id]['grants']}")
        if (
            stored["old-habits"]["is_active"]
            or stored["trip-planner"]["content"] != TRIP_SKILL_EDITED
        ):
            raise SystemExit("the release did not keep the toggle or the edit")
        owners = {
            "trip-planner": "alice",
            "uebersetzer": "alice",
            "old-habits": "bob",
            "house-style": "admin",
        }
        for skill_id, who in owners.items():
            stored[skill_id]["owner"] = who
        stored["dave-notes"]["owner"] = None
        readers = {
            "trip-planner": ["alice", "bob"],
            "uebersetzer": ["alice"],
            "old-habits": ["bob"],
            "house-style": ["alice", "bob", "carol"],
            "dave-notes": ["alice", "bob"],
        }
        writers = {
            "trip-planner": ["alice", "bob"],
            "uebersetzer": ["alice"],
            "old-habits": ["bob"],
        }
        for skill_id in stored:
            stored[skill_id]["readers"] = readers[skill_id]
            stored[skill_id]["writers"] = writers.get(skill_id, [])
        self.manifest["skills"] = stored
        self.manifest["deleted_account"] = {**dave, "id": added["id"]}

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
            cited = self._cited_chat(client)
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
            "cited": cited,
            "archived": {"id": archived["id"], "title": "Old plans"},
        }
        self.manifest["chat_file"] = {
            "id": file_id,
            "filename": "packing.txt",
            "text": CHAT_FILE_TEXT,
        }
        self.manifest["feedback"] = {"id": feedback["id"], "reason": "Accurate", "rating": 1}
        with self.client("bob") as client:
            self.manifest["chats"]["rated"] = self._rated_chat(client)

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
        stored = self._titled(client, first.chat_id, "Schnitzel questions")
        return {**stored, "reply_id": first.assistant_message_id}

    def _cited_chat(self, client: httpx.Client) -> dict:
        """A chat the server answered with the knowledge base attached, citing its file."""
        question = "Where does the lighthouse key hang?"
        answer = "Behind the blue door [1]."
        answers_the_chat = upstream_module.answering(question)
        knowledge = self.manifest["knowledge"]
        if writes_output_items(self.release):
            # attached knowledge is searched by the model's own tool call since v0.13
            search = {"query": "lighthouse key", "knowledge_ids": [knowledge["id"]]}
            self.provider.queue(
                upstream_module.tool_call("query_knowledge_files", search, match=answers_the_chat)
            )
        self.provider.queue(
            upstream_module.text(
                answer, match=lambda body: bool(body.get("stream")) and answers_the_chat(body)
            )
        )
        attached = {"type": "collection", "id": knowledge["id"], "name": knowledge["name"]}
        turn = send_message(client, question, chat_files=[attached])
        reply = wait_for_reply(client, turn)
        cited = {source["source"].get("name") for source in reply.get("sources") or []}
        if knowledge["filename"] not in cited and knowledge["name"] not in cited:
            raise SystemExit(f"the reply cites nothing from the knowledge base: {reply}")
        stored = self._titled(client, turn.chat_id, "Lighthouse key")
        return {**stored, "question": question, "answer": answer, "cited": sorted(cited)}

    def _rated_chat(self, client: httpx.Client) -> dict:
        """A reply with its token usage, rated with follow-ups shown, and a reply that failed."""
        _ensure(client.get("/api/models", params={"refresh": "true"}), "listing models")
        usage = {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
        self.provider.queue(upstream_module.text("Take the U-Bahn.", usage=usage))
        first = send_message(client, "How do I get to the Prater?")
        first_reply = wait_for_reply(client, first)
        self.provider.queue(upstream_module.error(500, "the provider is down"))
        second = send_message(
            client,
            "And back?",
            chat_id=first.chat_id,
            parent_id=first.assistant_message_id,
            history=[
                {"role": "user", "content": "How do I get to the Prater?"},
                {"role": "assistant", "content": first_reply["content"]},
            ],
        )
        failed = self._failed_reply(client, second)
        feedback = _ensure(
            client.post(
                "/api/v1/evaluations/feedback",
                json={
                    "type": "rating",
                    "data": {"rating": 1, "model_id": MOCK_MODEL_ID, "reason": "Accurate"},
                    "meta": {"chat_id": first.chat_id, "message_id": first.assistant_message_id},
                    "snapshot": {"chat": {}},
                },
            ),
            "rating the reply",
        )
        # the web client keeps the rating, follow-ups and embeds on the message and saves the chat
        stored = _ensure(client.get(f"/api/v1/chats/{first.chat_id}"), "reading the rated chat")
        history = stored["chat"]["history"]
        history["messages"][first.assistant_message_id].update(
            {
                "annotation": {
                    "rating": 1,
                    "reason": "Accurate",
                    "comment": "Exactly the line I needed.",
                    "tags": ["transport"],
                },
                "feedbackId": feedback["id"],
                "followUps": ["How long does it take?", "Is there a night line?"],
                "embeds": ["<p>U2 to Praterstern</p>"],
            }
        )
        _ensure(
            client.post(f"/api/v1/chats/{first.chat_id}", json={"chat": {"history": history}}),
            "saving the rating",
        )
        rated = self._titled(client, first.chat_id, "Getting to the Prater")
        return {**rated, "rated_id": first.assistant_message_id, "failed": failed}

    def _failed_reply(self, client: httpx.Client, turn) -> dict:
        """The reply the server stored for a provider that failed."""
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            stored = _ensure(client.get(f"/api/v1/chats/{turn.chat_id}"), "reading the chat")
            message = stored["chat"]["history"]["messages"].get(turn.assistant_message_id, {})
            if message.get("error"):
                return {"id": turn.assistant_message_id, "error": message["error"]}
            time.sleep(0.2)
        raise SystemExit(f"the failed reply never stored its error: {message}")

    def _titled(self, client: httpx.Client, chat_id: str, title: str | None) -> dict:
        """Name a chat the server wrote and record it as stored; `None` keeps its title."""
        if title is not None:
            client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": title}})
        stored = _ensure(client.get(f"/api/v1/chats/{chat_id}"), f"reading the chat {title}")
        messages = stored["chat"]["history"]["messages"]
        return {
            "id": chat_id,
            "title": stored["title"],
            "current_id": stored["chat"]["history"]["currentId"],
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

    def output_item_chats(self) -> None:
        """Chats as v0.13 writes them, each sent and changed the way its web client does."""
        with self.client("alice") as client:
            _ensure(client.get("/api/models", params={"refresh": "true"}), "listing models")
            tooled = self._tooled_chat(client)
            edited = self._edited_chat(client)
            regenerated = self._regenerated_chat(client)
            side_by_side = self._side_by_side_chat(client)
            delegating, subagent = self._subagent_chats(client)
            foldered = self._folder_started_chat(client)
            shared = _ensure(
                client.post(f"/api/v1/chats/{tooled['id']}/share", json={"share_mode": "continue"}),
                "sharing the tool chat to continue",
            )
            _ensure(
                client.post(
                    f"/api/v1/chats/shared/{tooled['id']}/access/update",
                    json={
                        "access_grants": [_grant("user", self._id("bob"))],
                        "share_mode": "continue",
                    },
                ),
                "granting bob the chat to continue",
            )
        tooled.update(share_id=shared["share_id"], share_mode="continue", shared_with=["bob"])
        self.manifest["chats"].update(
            {
                "tooled": tooled,
                "edited": edited,
                "regenerated": regenerated,
                "side_by_side": side_by_side,
                "delegating": delegating,
                "subagent": subagent,
                "foldered": foldered,
            }
        )

    def _send(
        self,
        client: httpx.Client,
        content: str = "",
        *,
        chat_id: str | None = None,
        user_message: dict | None = None,
        models: tuple[str, ...] = (MOCK_MODEL_ID,),
        **extra,
    ) -> tuple[str, str, list[str]]:
        """One send as the v0.13 web client makes it: no message list, one reply id per model.

        A given `user_message` is answered again, as Regenerate does.
        """
        replies = [
            {"model_id": model, "message_id": str(uuid.uuid4()), "modelIdx": index}
            for index, model in enumerate(models)
        ]
        reply_ids = [entry["message_id"] for entry in replies]
        if user_message is None:
            user_message = {
                "id": str(uuid.uuid4()),
                "parentId": None,
                "childrenIds": [],
                "role": "user",
                "user_id": self._id("alice"),
                "content": content,
                "models": list(models),
                "timestamp": int(time.time()),
            }
        user_message = {
            **user_message,
            "childrenIds": [*user_message.get("childrenIds", []), *reply_ids],
        }
        payload = {
            "stream": True,
            "model": models[0],
            "params": {},
            "features": {},
            "variables": {},
            "session_id": f"seed-{uuid.uuid4().hex[:8]}",
            "id": reply_ids[0],
            "message_ids": replies,
            "parent_id": user_message["parentId"],
            "user_message": user_message,
            "background_tasks": NO_BACKGROUND_TASKS,
            **extra,
        }
        if chat_id:
            payload["chat_id"] = chat_id
        accepted = _ensure(client.post("/api/chat/completions", json=payload), f"sending {content}")
        chat_id = accepted["chat_id"]
        for reply_id in reply_ids:
            wait_for_reply(client, ChatTurn(chat_id, user_message["id"], reply_id))
        return chat_id, user_message["id"], reply_ids

    def _recorded(self, client: httpx.Client, chat_id: str, title: str | None) -> dict:
        """The chat as `_titled` records it, plus every stored message whole and its folder."""
        recorded = self._titled(client, chat_id, title)
        stored = _ensure(client.get(f"/api/v1/chats/{chat_id}"), f"reading the chat {title}")
        return {
            **recorded,
            "folder_id": stored.get("folder_id"),
            "stored": stored["chat"]["history"]["messages"],
        }

    def _tooled_chat(self, client: httpx.Client) -> dict:
        """A tool call with its output, the model's reasoning and the token usage."""
        prompt = "What is the weather in Graz?"
        answer = "It is sunny in Graz, so leave the umbrella at home."
        reasoning = "The weather tool says sunny, so no umbrella is needed."
        answers = upstream_module.answering(prompt)
        self.provider.queue(
            upstream_module.tool_call("get_weather", {"city": "Graz"}, match=answers),
            upstream_module.text(answer, reasoning=reasoning, usage=USAGE, match=answers),
        )
        chat_id, _, (reply_id,) = self._send(client, prompt, tool_ids=["weather_tool"])
        recorded = self._recorded(client, chat_id, "Weather in Graz")
        reply = recorded["stored"][reply_id]
        written = json.dumps(reply.get("output"))
        if "Sunny in Graz" not in written or reasoning not in written or not reply.get("usage"):
            raise SystemExit(f"the tool reply lacks its output, reasoning or usage: {reply}")
        return {**recorded, "reply_id": reply_id, "answer": answer, "reasoning": reasoning}

    def _edited_chat(self, client: httpx.Client) -> dict:
        """A reply whose text the user changed in the editor and saved."""
        prompt = "Name a castle near Vienna."
        edited_text = "Burg Kreuzenstein, north of Vienna."
        self.provider.queue(
            upstream_module.text("Burg Kreuzenstein.", match=upstream_module.answering(prompt))
        )
        chat_id, _, (reply_id,) = self._send(client, prompt)
        opened = _ensure(client.get(f"/api/v1/chats/{chat_id}"), "opening the chat to edit")
        history = opened["chat"]["history"]
        reply = history["messages"][reply_id]
        if not reply.get("output"):
            raise SystemExit(f"the reply was stored without output items: {reply}")
        output = copy.deepcopy(reply["output"])
        [text_item] = [
            part
            for item in output
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        ]
        text_item["text"] = edited_text
        reply.update(output=output, content=edited_text)
        path = _message_path(history["messages"], history["currentId"])
        _ensure(
            client.post(
                f"/api/v1/chats/{chat_id}",
                json={
                    "chat": {
                        "models": [MOCK_MODEL_ID],
                        "history": history,
                        "messages": path,
                        "files": [],
                    }
                },
            ),
            "saving the edited reply",
        )
        recorded = self._recorded(client, chat_id, "Castle near Vienna")
        if recorded["messages"][reply_id]["content"] != edited_text:
            raise SystemExit(f"the edit was not kept: {recorded['stored'][reply_id]}")
        return {**recorded, "reply_id": reply_id, "edited_text": edited_text}

    def _regenerated_chat(self, client: httpx.Client) -> dict:
        """Three answers to one question, the middle one deleted again."""
        prompt = "Suggest a day trip from Vienna."
        answers = upstream_module.answering(prompt)
        texts = ["Bratislava by boat.", "Melk Abbey by train.", "The Wachau by bike."]
        self.provider.queue(upstream_module.text(texts[0], match=answers))
        chat_id, user_id, (first,) = self._send(client, prompt)
        reply_ids = [first]
        for text in texts[1:]:
            self.provider.queue(upstream_module.text(text, match=answers))
            opened = _ensure(client.get(f"/api/v1/chats/{chat_id}"), "opening the chat")
            question = opened["chat"]["history"]["messages"][user_id]
            _, _, (again,) = self._send(client, chat_id=chat_id, user_message=question)
            reply_ids.append(again)
        deleted = reply_ids[1]
        _ensure(client.delete(f"/api/v1/chats/{chat_id}/messages/{deleted}"), "deleting a version")
        recorded = self._recorded(client, chat_id, "Day trip")
        kept = [reply_ids[0], reply_ids[2]]
        if recorded["messages"][user_id]["children"] != kept or deleted in recorded["messages"]:
            raise SystemExit(f"the versions are not as left: {recorded['messages']}")
        return {**recorded, "versions": kept, "deleted_id": deleted, "texts": [texts[0], texts[2]]}

    def _side_by_side_chat(self, client: httpx.Client) -> dict:
        """One question answered by two models next to each other."""
        prompt = "Which museum first?"
        answers = upstream_module.answering(prompt)

        def from_handbook(body: dict) -> bool:
            return "You answer from the handbook." in json.dumps(body.get("messages"))

        self.provider.queue(
            upstream_module.text(
                "The Albertina.", match=lambda body: answers(body) and not from_handbook(body)
            ),
            upstream_module.text(
                "The handbook says the Leopold.",
                match=lambda body: answers(body) and from_handbook(body),
            ),
        )
        models = (MOCK_MODEL_ID, self.manifest["model"]["id"])
        chat_id, user_id, reply_ids = self._send(client, prompt, models=models)
        recorded = self._recorded(client, chat_id, "Museum order")
        answered_by = [recorded["stored"][reply_id].get("model") for reply_id in reply_ids]
        if answered_by != list(models):
            raise SystemExit(f"the columns were answered by {answered_by}")
        return {**recorded, "user_id": user_id, "columns": dict(zip(models, reply_ids))}

    def _subagent_chats(self, client: httpx.Client) -> tuple[dict, dict]:
        """A reply that delegated a task, and the sub-agent's own chat."""
        prompt = "Plan my museum day."
        task = "List three museums near the Ring."
        found = "Albertina, Leopold Museum and Kunsthistorisches Museum."
        answer = "Start at the Albertina, then walk to the Leopold Museum."
        answers = upstream_module.answering(prompt)
        self.provider.queue(
            upstream_module.text(found, match=upstream_module.answering(task)),
            upstream_module.tool_call("delegate_task", {"task": task}, match=answers),
            upstream_module.text(answer, match=answers),
        )
        chat_id, _, (reply_id,) = self._send(client, prompt)
        delegating = self._recorded(client, chat_id, "Museum day")
        if found not in json.dumps(delegating["stored"][reply_id].get("output")):
            raise SystemExit(f"the sub-agent's result is missing: {delegating['stored'][reply_id]}")
        # v0.13 offers no route that lists a chat's sub-agents, so the database names it
        engine = sqlalchemy.create_engine(self.database_url)
        try:
            with engine.connect() as connection:
                rows = connection.execute(sqlalchemy.text("SELECT id, meta FROM chat")).all()
        finally:
            engine.dispose()
        [subagent_id] = [
            row.id
            for row in rows
            for meta in [json.loads(row.meta) if isinstance(row.meta, str) else row.meta or {}]
            if meta.get("type") == "subagent" and meta.get("parent_chat_id") == chat_id
        ]
        subagent = self._recorded(client, subagent_id, None)
        return (
            {**delegating, "reply_id": reply_id, "task": task, "subagent_id": subagent_id},
            {**subagent, "parent_chat_id": chat_id, "answer": found},
        )

    def _folder_started_chat(self, client: httpx.Client) -> dict:
        """A chat started from inside a folder, which the server files there itself."""
        prompt = "Where do trains to Graz leave from?"
        self.provider.queue(
            upstream_module.text("From Wien Hauptbahnhof.", match=upstream_module.answering(prompt))
        )
        folder_id = self.manifest["folders"]["parent"]["id"]
        chat_id, _, _ = self._send(client, prompt, folder_id=folder_id)
        recorded = self._recorded(client, chat_id, "Trains to Graz")
        if recorded["folder_id"] != folder_id:
            raise SystemExit(f"the chat did not land in its folder: {recorded['folder_id']}")
        return recorded

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
            "ENABLE_SUBAGENTS": "true",
        }
        dump_with = None
        if engine == "postgres":
            url, binaries = stack.enter_context(postgres(work))
            settings["DATABASE_URL"] = url
            dump_with = (binaries / "pg_dump", url.replace("+psycopg2", ""))
        database_url = settings.get("DATABASE_URL", f"sqlite:///{data_dir / 'webui.db'}")
        with serving(data_dir, settings, backend=backend) as running:
            manifest = Seeder(running.base_url, provider, release, database_url).run()
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
