"""Journey: workspace and admin data written under one JSON codec reads the same under the other.

Two instances share one database, one with `ENABLE_ORJSON` off and one with it on. Each family the
workspace and the admin panel keep in JSON columns (models, tools, functions, prompts, skills,
notes, folders, channels, groups, calendar events, automations, memories, feedback) is created
with accents, CJK, right-to-left text, emoji and nested metadata through one instance, read back
by get and by list through the other, updated through the other and read again through the first.
The families with their own routes get their own journeys: model export and import, tool, skill and
function export with the function synced back, prompt history, tool and function valves, files and
knowledge, channel messages with reactions and threads, users, configs with banners, feedback with
the leaderboard and the analytics summary. The same values must come back on both codec values and
across the switch.

Discriminates: passes on dev 176d31d1d. In backend copies of dev 176d31d1d, the orjson codec
writing mojibake or orjson request parsing that mangles non-ASCII turns every crossing with an
orjson side red and the stdlib codec writing mojibake every crossing with a stdlib side, leaving
the stdlib-only and orjson-only crossings green respectively; the analytics summary counting one
extra message with the switch on turns the analytics test red. Retargeted for 9bbb95048, where a
skill's export holds its instructions as the file SKILL.md.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import textwrap
import uuid
import zipfile
from dataclasses import dataclass
from typing import Callable

import httpx
import pytest

from harness import upstream as reply
from harness.actors import create_user
from harness.calendar_api import HOUR_NS, to_ns, to_utc
from harness.chat import ask
from harness.json_codecs import (
    ALL_MIXED,
    CODECS,
    CROSSINGS,
    LONG_TEXT,
    MIXED_TEXT,
    codec_pair,
    nested,
)
from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.journey,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

ADMIN_CONFIG = "/api/v1/auths/admin/config"


@pytest.fixture(scope="module")
def pair(instance_with):
    return codec_pair(instance_with)


@pytest.fixture(scope="module")
def admins(pair):
    """The admin's client on each instance; the token is valid on both."""
    clients = {
        codec: httpx.Client(
            base_url=pair[codec].base_url,
            headers={"Authorization": f"Bearer {pair['stdlib'].admin_token}"},
            timeout=120.0,
        )
        for codec in CODECS
    }
    yield clients
    for client in clients.values():
        client.close()


@pytest.fixture(scope="module", autouse=True)
def channels_on(admins):
    for client in admins.values():
        current = _ok(client.get(ADMIN_CONFIG), "reading the admin config").json()
        _ok(
            client.post(ADMIN_CONFIG, json={**current, "ENABLE_CHANNELS": True}),
            "enabling channels",
        )


def _tag() -> str:
    return uuid.uuid4().hex[:8]


def _ok(response: httpx.Response, what: str) -> httpx.Response:
    assert response.status_code == 200, (
        f"{what} failed: HTTP {response.status_code} {response.text}"
    )
    return response


def _covers(expected, actual, path: str = "") -> None:
    """Every value in `expected` is in `actual` (which may hold more keys), same types and order."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path or 'value'}: {actual!r} is not an object"
        for key, value in expected.items():
            assert key in actual, f"{path}.{key} is missing from {actual!r}"
            _covers(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), (
            f"{path}: {actual!r} differs from {expected!r}"
        )
        for index, value in enumerate(expected):
            _covers(value, actual[index], f"{path}[{index}]")
    else:
        assert actual == expected, f"{path}: {actual!r} differs from {expected!r}"
        assert type(actual) is type(expected), f"{path}: {actual!r} changed type"


# ---------------------------------------------------------------- families


def _source(code: str) -> str:
    return textwrap.dedent(code).strip() + "\n"


def tool_source(note: str, default: str) -> str:
    return _source(
        f'''
        from pydantic import BaseModel

        class Tools:
            class Valves(BaseModel):
                greeting: str = "{default}"
                retries: int = 3
                ratio: float = 0.75
                verbose: bool = True
                labels: list[str] = []

            class UserValves(BaseModel):
                nickname: str = ""
                limit: int = 10

            def __init__(self):
                self.valves = self.Valves()

            def describe(self, topic: str) -> str:
                """{note}"""
                return topic
        '''
    )


def function_source(default: str) -> str:
    return _source(
        f'''
        from pydantic import BaseModel

        class Pipe:
            class Valves(BaseModel):
                label: str = "{default}"
                retries: int = 3
                ratio: float = 0.75
                labels: list[str] = []

            class UserValves(BaseModel):
                nickname: str = ""
                limit: int = 10

            def __init__(self):
                self.valves = self.Valves()

            def pipe(self, body):
                return "answered by " + self.valves.label
        '''
    )


VALVES = {
    "greeting": MIXED_TEXT["german"],
    "retries": 5,
    "ratio": 0.25,
    "verbose": False,
    "labels": [MIXED_TEXT["chinese"], MIXED_TEXT["emoji"], "plain"],
}
FUNCTION_VALVES = {
    "label": MIXED_TEXT["arabic"],
    "retries": 7,
    "ratio": 0.125,
    "labels": [MIXED_TEXT["hebrew"], MIXED_TEXT["pdf_paste"]],
}
USER_VALVES = {"nickname": MIXED_TEXT["korean"], "limit": 25}


@dataclass
class Family:
    """A resource with create, get, list and update routes; the checks are what the forms sent."""

    name: str
    create_path: str
    form: Callable[[str], dict]
    get_path: str
    list_path: str
    update_path: str
    changes: Callable[[dict], dict]
    items: Callable[[dict | list], list] = lambda body: body
    list_params: Callable[[dict], dict] = lambda form: {}
    update_form: Callable[[dict, dict], dict] = lambda form, changes: {**form, **changes}
    list_keys: tuple[str, ...] | None = None
    get_params: Callable[[dict], dict] = lambda entity: {}
    update_params: Callable[[dict], dict] = lambda entity: {}
    id_field: str = "id"
    omit: tuple[str, ...] = ()
    created_omit: tuple[str, ...] = ()


def _model_form(tag: str) -> dict:
    return {
        "id": f"codec-model-{tag}",
        "name": f"{MIXED_TEXT['french']} {tag}",
        "base_model_id": MOCK_MODEL_ID,
        "meta": {
            "description": MIXED_TEXT["japanese"],
            "tags": [{"name": "größe"}, {"name": "北京"}],
            "suggestion_prompts": [
                {"content": MIXED_TEXT["chinese"]},
                {"content": MIXED_TEXT["markdown"]},
            ],
            "capabilities": {"vision": True, "citations": False},
            "extra": nested(),
        },
        "params": {
            "system": MIXED_TEXT["hindi"],
            "temperature": 0.7,
            "top_p": 0.95,
            "seed": 42,
            "max_tokens": 2048,
            "stop": [MIXED_TEXT["pdf_paste"], "###"],
            "extra": nested(),
        },
        "is_active": True,
    }


def _tool_form(tag: str) -> dict:
    return {
        "id": f"codec_tool_{tag}",
        "name": f"{MIXED_TEXT['russian']} {tag}",
        "content": tool_source(MIXED_TEXT["math"], MIXED_TEXT["french"]),
        "meta": {"description": MIXED_TEXT["hindi"]},
    }


def _function_form(tag: str) -> dict:
    return {
        "id": f"codec_function_{tag}",
        "name": f"{MIXED_TEXT['german']} {tag}",
        "content": function_source(MIXED_TEXT["chinese"]),
        "meta": {"description": MIXED_TEXT["arabic"], "extra": nested()},
    }


def _prompt_form(tag: str) -> dict:
    return {
        "command": f"codec-{tag}",
        "name": MIXED_TEXT["german"],
        "content": f"{ALL_MIXED}\n{{{{name}}}}",
        "data": nested(),
        "meta": {"description": MIXED_TEXT["korean"], "extra": nested()},
        "tags": ["größe", "北京", "plain"],
        "commit_message": MIXED_TEXT["emoji"],
    }


def _event_form(tag: str) -> dict:
    start = to_ns(dt.datetime(2030, 6, 1, 10, tzinfo=dt.timezone.utc))
    return {
        "title": f"{MIXED_TEXT['japanese']} {tag}",
        "description": MIXED_TEXT["markdown"],
        "location": MIXED_TEXT["french"],
        "start_at": start,
        "end_at": start + HOUR_NS,
        "color": "#336699",
        "data": nested(),
        "meta": {"alert_minutes": 15, "extra": nested()},
    }


FAMILIES = [
    Family(
        name="models",
        create_path="/api/v1/models/create",
        form=_model_form,
        get_path="/api/v1/models/model",
        get_params=lambda entity: {"id": entity["id"]},
        list_path="/api/v1/models/list",
        list_params=lambda form: {"query": form["id"].split("-")[-1]},
        items=lambda body: body["items"],
        list_keys=("id", "name", "meta", "params"),
        update_path="/api/v1/models/model/update",
        changes=lambda form: {
            "name": f"{MIXED_TEXT['korean']} {form['id'][-8:]}",
            "meta": {**form["meta"], "description": MIXED_TEXT["hindi"], "extra": nested()},
            "params": {**form["params"], "temperature": 0.2, "seed": 7},
        },
    ),
    Family(
        name="tools",
        create_path="/api/v1/tools/create",
        form=_tool_form,
        get_path="/api/v1/tools/id/{id}",
        list_path="/api/v1/tools/list",
        list_keys=("id", "name", "meta"),
        created_omit=("content",),
        update_path="/api/v1/tools/id/{id}/update",
        changes=lambda form: {
            "name": MIXED_TEXT["hebrew"],
            "content": tool_source(MIXED_TEXT["emoji"], MIXED_TEXT["arabic"]),
            "meta": {"description": MIXED_TEXT["chinese"]},
        },
    ),
    Family(
        name="functions",
        create_path="/api/v1/functions/create",
        form=_function_form,
        get_path="/api/v1/functions/id/{id}",
        list_path="/api/v1/functions/",
        list_keys=("id", "name", "meta"),
        created_omit=("content",),
        update_path="/api/v1/functions/id/{id}/update",
        changes=lambda form: {
            "name": MIXED_TEXT["russian"],
            "content": function_source(MIXED_TEXT["emoji"]),
            "meta": {"description": MIXED_TEXT["korean"], "extra": nested()},
        },
    ),
    Family(
        name="prompts",
        create_path="/api/v1/prompts/create",
        form=_prompt_form,
        get_path="/api/v1/prompts/id/{id}",
        list_path="/api/v1/prompts/list",
        items=lambda body: body["items"],
        list_params=lambda form: {"query": form["command"]},
        omit=("commit_message",),
        update_path="/api/v1/prompts/id/{id}/update",
        changes=lambda form: {
            "content": LONG_TEXT[:3000],
            "data": {**nested(), "count": 43},
            "tags": ["übermaß", "東京"],
            "commit_message": MIXED_TEXT["chinese"],
        },
    ),
    Family(
        name="skills",
        create_path="/api/v1/skills/create",
        form=lambda tag: {
            "id": f"codec-skill-{tag}",
            "name": f"{MIXED_TEXT['chinese']} {tag}",
            "description": MIXED_TEXT["emoji"],
            "content": ALL_MIXED,
            "meta": {"tags": ["größe", "北京"]},
            "is_active": True,
        },
        get_path="/api/v1/skills/id/{id}",
        list_path="/api/v1/skills/list",
        items=lambda body: body["items"],
        list_keys=("id", "name", "description", "meta"),
        created_omit=("content",),
        update_path="/api/v1/skills/id/{id}/update",
        changes=lambda form: {
            "name": f"{MIXED_TEXT['arabic']} {form['id'][-8:]}",
            "content": MIXED_TEXT["markdown"],
            "meta": {"tags": ["日本語"]},
        },
    ),
    Family(
        name="notes",
        create_path="/api/v1/notes/create",
        form=lambda tag: {
            "title": f"{MIXED_TEXT['russian']} {tag}",
            "data": {
                "content": {"md": MIXED_TEXT["markdown"], "html": MIXED_TEXT["html"]},
                "versions": [nested()],
            },
            "meta": {"tags": ["größe"], "extra": nested()},
        },
        get_path="/api/v1/notes/{id}",
        list_path="/api/v1/notes/search",
        items=lambda body: body["items"],
        list_params=lambda form: {"query": form["title"]},
        list_keys=("title", "meta"),
        update_path="/api/v1/notes/{id}/update",
        changes=lambda form: {
            "title": f"{MIXED_TEXT['hindi']} {form['title'].split()[-1]}",
            "data": {"content": {"md": ALL_MIXED, "html": "<p>größe 北京</p>"}},
            "meta": {"extra": nested()},
        },
    ),
    Family(
        name="folders",
        create_path="/api/v1/folders/",
        form=lambda tag: {
            "name": f"{MIXED_TEXT['korean']} {tag}",
            "data": {"system_prompt": ALL_MIXED, "extra": nested()},
            "meta": {"icon": MIXED_TEXT["emoji"], "extra": nested()},
        },
        get_path="/api/v1/folders/{id}",
        list_path="/api/v1/folders/",
        list_keys=("name",),
        update_path="/api/v1/folders/{id}/update",
        changes=lambda form: {
            "name": f"{MIXED_TEXT['hebrew']} {form['name'].split()[-1]}",
            "data": {"system_prompt": MIXED_TEXT["arabic"], "extra": nested()},
            "meta": {"icon": "📁", "extra": nested()},
        },
        update_form=lambda form, changes: changes,
    ),
    Family(
        name="channels",
        create_path="/api/v1/channels/create",
        form=lambda tag: {
            "name": f"codec-{tag}",
            "description": MIXED_TEXT["japanese"],
            "data": nested(),
            "meta": {"topic": MIXED_TEXT["arabic"], "extra": nested()},
        },
        get_path="/api/v1/channels/{id}",
        list_path="/api/v1/channels/list",
        list_keys=("name", "description", "data", "meta"),
        update_path="/api/v1/channels/{id}/update",
        changes=lambda form: {
            "name": form["name"],
            "description": MIXED_TEXT["hindi"],
            "data": {"pinned": [MIXED_TEXT["emoji"]]},
            "meta": {"topic": MIXED_TEXT["hebrew"]},
        },
    ),
    Family(
        name="groups",
        create_path="/api/v1/groups/create",
        form=lambda tag: {
            "name": f"{MIXED_TEXT['french']} {tag}",
            "description": MIXED_TEXT["chinese"],
            "permissions": {
                "workspace": {"models": True, "prompts": False},
                "chat": {"file_upload": True, "temporary": False},
            },
            "data": {"notes": MIXED_TEXT["russian"], "extra": nested()},
        },
        get_path="/api/v1/groups/id/{id}",
        list_path="/api/v1/groups/",
        list_keys=("name", "description", "permissions", "data"),
        update_path="/api/v1/groups/id/{id}/update",
        changes=lambda form: {
            "name": f"{MIXED_TEXT['german']} {_tag()}",
            "description": MIXED_TEXT["korean"],
            "permissions": {"workspace": {"models": False, "prompts": True}},
        },
    ),
    Family(
        name="calendar events",
        create_path="/api/v1/calendars/events/create",
        form=lambda tag: {"calendar_id": "", **_event_form(tag)},
        get_path="/api/v1/calendars/events/{id}",
        list_path="/api/v1/calendars/events",
        list_params=lambda form: {
            "start": to_utc(form["start_at"] - 2 * HOUR_NS).isoformat(),
            "end": to_utc(form["end_at"] + 2 * HOUR_NS).isoformat(),
        },
        list_keys=("title", "description", "location", "meta"),
        update_path="/api/v1/calendars/events/{id}/update",
        changes=lambda form: {
            "title": MIXED_TEXT["arabic"],
            "description": MIXED_TEXT["emoji"],
            "location": MIXED_TEXT["korean"],
            "meta": {"alert_minutes": 30, "extra": nested()},
        },
    ),
    Family(
        name="automations",
        create_path="/api/v1/automations/create",
        form=lambda tag: {
            "name": f"{MIXED_TEXT['german']} report {tag}",
            "data": {
                "prompt": f"{MIXED_TEXT['chinese']} report {tag}",
                "model_id": MOCK_MODEL_ID,
                "rrule": "RRULE:FREQ=DAILY",
            },
            "meta": {"extra": nested()},
            "is_active": False,
        },
        get_path="/api/v1/automations/{id}",
        list_path="/api/v1/automations/list",
        items=lambda body: body["items"],
        list_params=lambda form: {"query": form["name"].split()[-1]},
        update_path="/api/v1/automations/{id}/update",
        changes=lambda form: {
            "name": f"{MIXED_TEXT['russian']} report {form['name'].split()[-1]}",
            "data": {**form["data"], "prompt": f"{MIXED_TEXT['emoji']} {form['name'].split()[-1]}"},
        },
    ),
    Family(
        name="memories",
        create_path="/api/v1/memories/add",
        form=lambda tag: {"content": f"{MIXED_TEXT['japanese']} {tag}"},
        get_path="",
        list_path="/api/v1/memories/",
        update_path="/api/v1/memories/{id}/update",
        changes=lambda form: {"content": MIXED_TEXT["hindi"]},
    ),
    Family(
        name="feedback",
        create_path="/api/v1/evaluations/feedback",
        form=lambda tag: {
            "type": "rating",
            "data": {
                "rating": 1,
                "model_id": MOCK_MODEL_ID,
                "reason": MIXED_TEXT["german"],
                "comment": MIXED_TEXT["chinese"],
                "details": {"rating": 8, "extra": nested()},
            },
            "meta": {"arena": False, "chat_id": tag, "tags": ["größe", "北京"], "extra": nested()},
            "snapshot": {"chat": {"title": MIXED_TEXT["russian"], "history": nested()}},
        },
        get_path="/api/v1/evaluations/feedback/{id}",
        list_path="/api/v1/evaluations/feedbacks/all/export",
        update_path="/api/v1/evaluations/feedback/{id}",
        changes=lambda form: {
            "data": {**form["data"], "rating": -1, "comment": MIXED_TEXT["arabic"]},
            "meta": {**form["meta"], "tags": ["日本語"]},
        },
    ),
]


def _send(client: httpx.Client, family: Family, form: dict):
    return client.post(family.create_path, json=form)


def _read(client: httpx.Client, family: Family, entity: dict) -> dict | None:
    if not family.get_path:
        return next(
            item for item in _list(client, family, {}, entity) if item["id"] == entity["id"]
        )
    path = family.get_path.format(id=entity[family.id_field])
    return _ok(client.get(path, params=family.get_params(entity)), f"reading {family.name}").json()


def _list(client: httpx.Client, family: Family, form: dict, entity: dict) -> list[dict]:
    listed = _ok(
        client.get(family.list_path, params=family.list_params(form)), f"listing {family.name}"
    )
    return family.items(listed.json())


def _one_of(items: list[dict], entity: dict, family: Family) -> dict:
    found = [item for item in items if item.get(family.id_field) == entity[family.id_field]]
    assert len(found) == 1, f"{family.name} {entity[family.id_field]} is not listed once: {items}"
    return found[0]


def _pick(form: dict, family: Family, keys: tuple[str, ...] | None = None, *omit: str) -> dict:
    keys = keys or tuple(form)
    skipped = {*family.omit, *omit}
    return {key: form[key] for key in keys if key in form and key not in skipped}


@pytest.mark.parametrize("family", FAMILIES, ids=[family.name for family in FAMILIES])
@pytest.mark.parametrize(("writer", "reader"), CROSSINGS)
def test_a_resource_reads_and_updates_the_same_across_the_switch(admins, family, writer, reader):
    form = family.form(_tag())
    if "calendar_id" in form:
        calendars = _ok(admins[writer].get("/api/v1/calendars/"), "listing calendars").json()
        form["calendar_id"] = next(item["id"] for item in calendars if item["is_default"])

    created = _ok(_send(admins[writer], family, form), f"creating {family.name}").json()
    _covers(_pick(form, family, None, *family.created_omit), created)
    _covers(_pick(form, family, family.list_keys), _read(admins[reader], family, created))
    listed = _one_of(_list(admins[reader], family, form, created), created, family)
    _covers(_pick(form, family, family.list_keys), listed)

    update = family.update_form(form, family.changes(form))
    path = family.update_path.format(id=created[family.id_field])
    _ok(admins[reader].post(path, json=update, params=family.update_params(created)), "updating")

    _covers(_pick(update, family, family.list_keys), _read(admins[writer], family, created))
    again = _one_of(_list(admins[writer], family, form, created), created, family)
    _covers(_pick(update, family, family.list_keys), again)


# ---------------------------------------------------------------- model export and import


@pytest.mark.parametrize(("writer", "reader"), CROSSINGS)
def test_a_model_exports_and_imports_the_same_across_the_switch(admins, writer, reader):
    form = _model_form(_tag())
    _ok(admins[writer].post("/api/v1/models/create", json=form), "creating the model")

    exported = _ok(
        admins[reader].get("/api/v1/models/export", params={"ids": form["id"]}), "exporting"
    ).json()
    [item] = exported
    _covers({key: form[key] for key in ("id", "name", "meta", "params")}, item)

    _ok(admins[writer].post("/api/v1/models/model/delete", json={"id": form["id"]}), "deleting")
    _ok(admins[reader].post("/api/v1/models/import", json={"models": exported}), "importing")

    restored = _ok(
        admins[writer].get("/api/v1/models/model", params={"id": form["id"]}), "reading"
    ).json()
    _covers({key: form[key] for key in ("id", "name", "meta", "params")}, restored)


@pytest.mark.parametrize(("writer", "reader"), CROSSINGS)
def test_tools_skills_and_functions_export_and_sync_the_same_across_the_switch(
    admins, writer, reader
):
    tool, function = _tool_form(_tag()), _function_form(_tag())
    tag = _tag()
    skill = {
        "id": f"codec-skill-{tag}",
        "name": f"{MIXED_TEXT['chinese']} {tag}",
        "description": MIXED_TEXT["emoji"],
        "content": ALL_MIXED,
        "meta": {"tags": ["größe", "北京"]},
        "is_active": True,
    }
    _ok(admins[writer].post("/api/v1/tools/create", json=tool), "creating the tool")
    _ok(admins[writer].post("/api/v1/skills/create", json=skill), "creating the skill")
    _ok(admins[writer].post("/api/v1/functions/create", json=function), "creating the function")
    path = f"/api/v1/functions/id/{function['id']}/valves/update"
    _ok(admins[writer].post(path, json=FUNCTION_VALVES), "saving the valves")

    def exported(kind: str, entity_id: str, **params) -> dict:
        listed = _ok(admins[reader].get(f"/api/v1/{kind}/export", params=params), kind).json()
        return next(item for item in listed if item["id"] == entity_id)

    _covers({key: tool[key] for key in ("id", "name", "meta")}, exported("tools", tool["id"]))
    # a skill's export carries its instructions as the file SKILL.md
    skill_export = {key: value for key, value in skill.items() if key != "content"}
    skill_export["files"] = [{"path": "SKILL.md", "content": skill["content"]}]
    _covers(skill_export, exported("skills", skill["id"]))
    function_export = exported("functions", function["id"], include_valves="true")
    _covers({key: function[key] for key in ("id", "name")}, function_export)
    _covers(FUNCTION_VALVES, function_export["valves"])

    imported = {**function_export, "name": MIXED_TEXT["hebrew"]}
    synced = admins[reader].post("/api/v1/functions/sync", json={"functions": [imported]})
    _ok(synced, "syncing the export back")
    restored = _ok(admins[writer].get(f"/api/v1/functions/id/{function['id']}"), "reading").json()
    assert restored["name"] == MIXED_TEXT["hebrew"]
    valves = _ok(admins[writer].get(f"/api/v1/functions/id/{function['id']}/valves"), "valves")
    assert valves.json() == FUNCTION_VALVES


# ---------------------------------------------------------------- prompt history


def _history(client: httpx.Client, prompt_id: str) -> list[dict]:
    listed = _ok(client.get(f"/api/v1/prompts/id/{prompt_id}/history"), "reading the history")
    return listed.json()


@pytest.mark.parametrize(("writer", "reader"), CROSSINGS)
def test_a_prompt_keeps_its_version_history_across_the_switch(admins, writer, reader):
    form = _prompt_form(_tag())
    prompt = _ok(admins[writer].post("/api/v1/prompts/create", json=form), "creating").json()
    second = {**form, "content": LONG_TEXT[:2000], "commit_message": MIXED_TEXT["chinese"]}
    _ok(admins[reader].post(f"/api/v1/prompts/id/{prompt['id']}/update", json=second), "saving")

    for codec in (writer, reader):
        entries = _history(admins[codec], prompt["id"])
        by_content = {entry["snapshot"]["content"]: entry for entry in entries}
        assert set(by_content) == {form["content"], second["content"]}
        assert by_content[second["content"]]["commit_message"] == MIXED_TEXT["chinese"]
        assert by_content[form["content"]]["commit_message"] == MIXED_TEXT["emoji"]
        _covers(form["data"], by_content[form["content"]]["snapshot"]["data"])
        assert by_content[form["content"]]["snapshot"]["tags"] == form["tags"]

    first = next(
        entry
        for entry in _history(admins[writer], prompt["id"])
        if entry["snapshot"]["content"] == form["content"]
    )
    restored = admins[writer].post(
        f"/api/v1/prompts/id/{prompt['id']}/update/version", json={"version_id": first["id"]}
    )
    _ok(restored, "restoring")
    current = _ok(admins[reader].get(f"/api/v1/prompts/id/{prompt['id']}"), "reading").json()
    assert current["content"] == form["content"]
    _covers(form["data"], current["data"])
    assert current["tags"] == form["tags"]


# ---------------------------------------------------------------- valves and specs


VALVED = [
    ("tools", "/api/v1/tools/id/{id}", _tool_form, VALVES),
    ("functions", "/api/v1/functions/id/{id}", _function_form, FUNCTION_VALVES),
]


@pytest.mark.parametrize("kind", VALVED, ids=[entry[0] for entry in VALVED])
@pytest.mark.parametrize(("writer", "reader"), CROSSINGS)
def test_valves_and_user_valves_read_the_same_across_the_switch(admins, kind, writer, reader):
    name, path, make_form, valves = kind
    form = make_form(_tag())
    _ok(admins[writer].post(f"/api/v1/{name}/create", json=form), f"creating the {name}")
    path = path.format(id=form["id"])

    if name == "functions":
        _ok(admins[writer].post(f"{path}/toggle"), "switching the function on")
    else:
        specs = _ok(admins[reader].get(path), "reading the tool").json()["specs"]
        assert MIXED_TEXT["math"] in json.dumps(specs, ensure_ascii=False)

    _ok(admins[writer].post(f"{path}/valves/update", json=valves), "saving valves")
    assert _ok(admins[reader].get(f"{path}/valves"), "reading valves").json() == valves
    spec = _ok(admins[reader].get(f"{path}/valves/spec"), "reading the spec").json()
    assert spec["properties"], spec

    changed = {**valves, "retries": 9, "ratio": 0.5, "labels": [MIXED_TEXT["russian"]]}
    _ok(admins[reader].post(f"{path}/valves/update", json=changed), "updating valves")
    assert _ok(admins[writer].get(f"{path}/valves"), "reading valves").json() == changed

    _ok(admins[writer].post(f"{path}/valves/user/update", json=USER_VALVES), "user valves")
    assert _ok(admins[reader].get(f"{path}/valves/user"), "user valves").json() == USER_VALVES
    other = {"nickname": MIXED_TEXT["emoji"], "limit": 3}
    _ok(admins[reader].post(f"{path}/valves/user/update", json=other), "user valves")
    assert _ok(admins[writer].get(f"{path}/valves/user"), "user valves").json() == other


# ---------------------------------------------------------------- files and knowledge

FILE_NAME = "Übersicht_日本語_🚀.txt"
# text extraction folds the full-width punctuation in the rest, on both values
EXTRACTED_KEYS = ("german", "french", "arabic", "hebrew", "russian", "emoji", "math", "markdown")
FILE_METADATA = {"source": MIXED_TEXT["german"], "extra": nested()}


def _upload(client: httpx.Client, name: str, text: str, metadata: dict | None = None) -> dict:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (name, text.encode(), "text/plain")},
        data={"metadata": json.dumps(metadata)} if metadata else None,
    )
    return _ok(uploaded, "uploading").json()


@pytest.mark.parametrize(("writer", "reader"), CROSSINGS)
def test_an_uploaded_file_reads_and_updates_the_same_across_the_switch(admins, writer, reader):
    text = f"{ALL_MIXED}\n{_tag()}"
    uploaded = _upload(admins[writer], FILE_NAME, text, FILE_METADATA)
    file_id = uploaded["id"]

    assert uploaded["filename"] == FILE_NAME
    _covers(FILE_METADATA, uploaded["meta"]["data"])
    got = _ok(admins[reader].get(f"/api/v1/files/{file_id}"), "reading the file").json()
    assert got["filename"] == FILE_NAME
    _covers(FILE_METADATA, got["meta"]["data"])
    assert got["meta"]["name"] == FILE_NAME
    stored = admins[reader].get(f"/api/v1/files/{file_id}/data/content")
    content = _ok(stored, "reading the content").json()["content"]
    assert all(MIXED_TEXT[key] in content for key in EXTRACTED_KEYS)
    assert content.endswith(text.splitlines()[-1])
    assert _ok(admins[writer].get(f"/api/v1/files/{file_id}/data/content"), "content").json() == {
        "content": content
    }
    assert _ok(admins[reader].get(f"/api/v1/files/{file_id}/content"), "download").text == text

    changed = f"{MIXED_TEXT['arabic']}\n{MIXED_TEXT['markdown']}"
    update = admins[reader].post(
        f"/api/v1/files/{file_id}/data/content/update", json={"content": changed}
    )
    _ok(update, "updating the content")
    reread = admins[writer].get(f"/api/v1/files/{file_id}/data/content")
    assert _ok(reread, "reading the content").json() == {"content": changed}
    found = admins[writer].get("/api/v1/files/search", params={"filename": FILE_NAME})
    assert file_id in [item["id"] for item in _ok(found, "searching files").json()]


@pytest.mark.parametrize(("writer", "reader"), CROSSINGS)
def test_a_knowledge_base_with_a_text_file_reads_the_same_across_the_switch(admins, writer, reader):
    tag = _tag()
    name, description = f"{MIXED_TEXT['french']} {tag}", MIXED_TEXT["japanese"]
    created = admins[writer].post(
        "/api/v1/knowledge/create", json={"name": name, "description": description}
    )
    knowledge_id = _ok(created, "creating the knowledge base").json()["id"]

    text = f"{ALL_MIXED}\n{tag}"
    file_id = _upload(admins[reader], FILE_NAME, text)["id"]
    added = admins[reader].post(
        f"/api/v1/knowledge/{knowledge_id}/file/add", json={"file_id": file_id}
    )
    _ok(added, "adding the file")

    for codec in (writer, reader):
        client = admins[codec]
        base = _ok(client.get(f"/api/v1/knowledge/{knowledge_id}"), "reading").json()
        assert (base["name"], base["description"]) == (name, description)
        files = _ok(client.get(f"/api/v1/knowledge/{knowledge_id}/files"), "listing").json()
        assert [(item["id"], item["meta"]["name"]) for item in files["items"]] == [
            (file_id, FILE_NAME)
        ]
        content = _ok(client.get(f"/api/v1/files/{file_id}/data/content"), "content").json()
        assert all(MIXED_TEXT[key] in content["content"] for key in EXTRACTED_KEYS)
        listed = _ok(client.get("/api/v1/knowledge/search", params={"query": tag}), "search")
        assert [item["id"] for item in listed.json()["items"]] == [knowledge_id]

    renamed = {"name": f"{MIXED_TEXT['korean']} {tag}", "description": MIXED_TEXT["arabic"]}
    _ok(admins[reader].post(f"/api/v1/knowledge/{knowledge_id}/update", json=renamed), "renaming")
    base = _ok(admins[writer].get(f"/api/v1/knowledge/{knowledge_id}"), "reading").json()
    assert (base["name"], base["description"]) == (renamed["name"], renamed["description"])

    exported = admins[reader].get(f"/api/v1/knowledge/{knowledge_id}/export")
    with zipfile.ZipFile(io.BytesIO(_ok(exported, "exporting").content)) as archive:
        assert archive.namelist() == [FILE_NAME]
        assert archive.read(FILE_NAME).decode() == content["content"]


# ---------------------------------------------------------------- channel messages


@pytest.mark.parametrize(("writer", "reader"), CROSSINGS)
def test_channel_messages_reactions_and_threads_read_the_same_across_the_switch(
    admins, writer, reader
):
    channel = _ok(
        admins[writer].post("/api/v1/channels/create", json={"name": f"talk-{_tag()}"}), "channel"
    ).json()
    base = f"/api/v1/channels/{channel['id']}/messages"
    content, data, meta = ALL_MIXED, {"extra": nested()}, {"origin": MIXED_TEXT["hindi"]}
    posted = admins[writer].post(
        f"{base}/post", json={"content": content, "data": data, "meta": meta}
    )
    message = _ok(posted, "posting").json()
    reply_form = {"content": MIXED_TEXT["arabic"], "parent_id": message["id"]}
    thread_reply = _ok(admins[reader].post(f"{base}/post", json=reply_form), "replying").json()
    for name in ("🚀", "heart"):
        added = admins[reader].post(f"{base}/{message['id']}/reactions/add", json={"name": name})
        assert _ok(added, "reacting").json() is True

    for codec in (writer, reader):
        client = admins[codec]
        [listed] = [
            item for item in _ok(client.get(base), "listing").json() if item["id"] == message["id"]
        ]
        assert listed["content"] == content
        assert sorted(reaction["name"] for reaction in listed["reactions"]) == ["heart", "🚀"]
        assert listed["reply_count"] == 1
        one = _ok(client.get(f"{base}/{message['id']}"), "reading").json()
        assert (one["content"], one["meta"]) == (content, meta)
        stored = _ok(client.get(f"{base}/{message['id']}/data"), "reading the data").json()
        _covers(data, stored)
        thread = _ok(client.get(f"{base}/{message['id']}/thread"), "thread").json()
        assert sorted(entry["content"] for entry in thread) == sorted(
            [content, MIXED_TEXT["arabic"]]
        )
        assert thread_reply["id"] in [entry["id"] for entry in thread]

    edited = {"content": MIXED_TEXT["emoji"], "data": {"extra": nested(), "edited": True}}
    _ok(admins[reader].post(f"{base}/{message['id']}/update", json=edited), "editing")
    removed = admins[writer].post(f"{base}/{message['id']}/reactions/remove", json={"name": "🚀"})
    _ok(removed, "removing a reaction")
    one = _ok(admins[writer].get(f"{base}/{message['id']}"), "reading").json()
    assert one["content"] == MIXED_TEXT["emoji"]
    assert [reaction["name"] for reaction in one["reactions"]] == ["heart"]
    _covers(edited["data"], _ok(admins[reader].get(f"{base}/{message['id']}/data"), "data").json())


# ---------------------------------------------------------------- users


@pytest.mark.parametrize(("writer", "reader"), CROSSINGS)
def test_user_settings_info_and_profile_read_the_same_across_the_switch(
    admins, pair, writer, reader
):
    account = create_user(pair["stdlib"], name=f"{MIXED_TEXT['german']} {_tag()}")
    users = {
        codec: httpx.Client(
            base_url=pair[codec].base_url,
            headers={"Authorization": f"Bearer {account.token}"},
            timeout=120.0,
        )
        for codec in CODECS
    }
    try:
        settings = {
            "ui": {
                "system": MIXED_TEXT["markdown"],
                "params": {"temperature": 0.7, "seed": 42, "stop": [MIXED_TEXT["pdf_paste"]]},
                "notes": nested(),
                "title": {"auto": False},
            },
            "extra": nested(),
        }
        _ok(users[writer].post("/api/v1/users/user/settings/update", json=settings), "settings")
        _covers(settings, _ok(users[reader].get("/api/v1/users/user/settings"), "settings").json())
        info = {"bio": ALL_MIXED, "links": [MIXED_TEXT["html"]], "extra": nested()}
        _ok(users[reader].post("/api/v1/users/user/info/update", json=info), "info")
        _covers(info, _ok(users[writer].get("/api/v1/users/user/info"), "info").json())

        changed = {"ui": {**settings["ui"], "system": MIXED_TEXT["chinese"]}}
        _ok(users[reader].post("/api/v1/users/user/settings/update", json=changed), "settings")
        _covers(changed, _ok(users[writer].get("/api/v1/users/user/settings"), "settings").json())
        more = {"bio": MIXED_TEXT["emoji"]}
        _ok(users[writer].post("/api/v1/users/user/info/update", json=more), "info")
        _covers({**info, **more}, _ok(users[reader].get("/api/v1/users/user/info"), "info").json())

        new_name = f"{MIXED_TEXT['korean']} {_tag()}"
        profile = {"name": new_name, "email": account.email, "profile_image_url": "/user.png"}
        _ok(admins[reader].post(f"/api/v1/users/{account.id}/update", json=profile), "profile")
        for codec in (writer, reader):
            assert (
                _ok(admins[codec].get(f"/api/v1/users/{account.id}"), "user").json()["name"]
                == new_name
            )
            listed = _ok(
                admins[codec].get("/api/v1/users/", params={"query": new_name.split()[-1]}), "users"
            ).json()
            assert [user["name"] for user in listed["users"]] == [new_name]
    finally:
        for client in users.values():
            client.close()


# ---------------------------------------------------------------- configs and banners

BANNERS = "/api/v1/configs/banners"


def _banner(text: str, tag: str) -> dict:
    return {
        "id": f"codec-{tag}",
        "type": "info",
        "title": f"{text} {tag}",
        "content": ALL_MIXED,
        "dismissible": True,
        "timestamp": 1767225600,
    }


@pytest.mark.parametrize(("writer", "reader"), CROSSINGS)
def test_banners_and_config_export_and_import_read_the_same_across_the_switch(
    admins, writer, reader
):
    before = _ok(admins[writer].get("/api/v1/configs/export"), "exporting").json()
    try:
        banners = [_banner(MIXED_TEXT["chinese"], _tag()), _banner(MIXED_TEXT["emoji"], _tag())]
        _ok(admins[writer].post(BANNERS, json={"banners": banners}), "saving banners")
        listed = _ok(admins[reader].get(BANNERS), "reading banners").json()
        _covers(banners, listed)

        exported = _ok(admins[reader].get("/api/v1/configs/export"), "exporting").json()
        _covers(banners, exported["ui.banners"])
        replaced = [_banner(MIXED_TEXT["arabic"], _tag())]
        imported = admins[reader].post(
            "/api/v1/configs/import", json={"config": {**exported, "ui.banners": replaced}}
        )
        _ok(imported, "importing")
        _covers(replaced, _ok(admins[writer].get(BANNERS), "reading banners").json())
        _covers(
            replaced,
            _ok(admins[writer].get("/api/v1/configs/export"), "exporting").json()["ui.banners"],
        )
    finally:
        _ok(admins[writer].post("/api/v1/configs/import", json={"config": before}), "restoring")
    assert _ok(admins[reader].get("/api/v1/configs/export"), "exporting").json() == before


# ---------------------------------------------------------------- evaluations and analytics


def _duel(winner: str, loser: str, comment: str) -> dict:
    return {
        "type": "rating",
        "data": {
            "rating": 1,
            "model_id": winner,
            "sibling_model_ids": [loser],
            "reason": MIXED_TEXT["german"],
            "comment": comment,
            "details": {"extra": nested()},
        },
        "meta": {"arena": True, "tags": ["größe", "北京"], "extra": nested()},
        "snapshot": {"chat": {"title": comment, "history": nested()}},
    }


def test_the_leaderboard_is_the_same_on_both_codecs_after_feedback_from_each(admins):
    tag = _tag()
    winner, loser = f"arena-{tag}-a", f"arena-{tag}-b"
    for codec, comment in zip(CODECS, (MIXED_TEXT["chinese"], MIXED_TEXT["arabic"])):
        form = _duel(winner, loser, comment)
        _ok(admins[codec].post("/api/v1/evaluations/feedback", json=form), "rating")

    boards = {
        codec: _ok(admins[codec].get("/api/v1/evaluations/leaderboard"), "leaderboard").json()
        for codec in CODECS
    }
    assert boards["stdlib"] == boards["orjson"]
    ranked = {entry["model_id"]: entry for entry in boards["orjson"]["entries"]}
    assert (ranked[winner]["won"], ranked[loser]["lost"]) == (2, 2)
    assert ranked[winner]["rating"] > ranked[loser]["rating"]
    history = {
        codec: _ok(
            admins[codec].get(f"/api/v1/evaluations/leaderboard/{winner}/history"), "hist"
        ).json()
        for codec in CODECS
    }
    assert history["stdlib"] == history["orjson"]

    exported = _ok(
        admins["orjson"].get("/api/v1/evaluations/feedbacks/all/export"), "export"
    ).json()
    ours = [item for item in exported if item["data"].get("model_id") == winner]
    assert sorted(item["data"]["comment"] for item in ours) == sorted(
        [MIXED_TEXT["chinese"], MIXED_TEXT["arabic"]]
    )
    for item in ours:
        _covers(nested(), item["snapshot"]["chat"]["history"])
        assert item["meta"]["tags"] == ["größe", "北京"]


ANALYTICS_ROUTES = ("summary", "models", "users", "daily", "tokens")


def test_the_analytics_routes_agree_on_both_codecs_after_a_chat_on_each(pair, admins):
    def numbers(codec: str) -> dict:
        return {
            route: _ok(admins[codec].get(f"/api/v1/analytics/{route}"), route).json()
            for route in ANALYTICS_ROUTES
        }

    before = numbers("stdlib")["summary"]["total_messages"]
    for turn, codec in enumerate(CODECS, start=1):
        pair[codec].upstream.reset()
        pair[codec].upstream.queue(reply.text(MIXED_TEXT["french"]))
        ask(admins[codec], f"{MIXED_TEXT['chinese']} {_tag()}")

        seen = {name: numbers(name) for name in CODECS}
        assert seen["stdlib"] == seen["orjson"], f"after the chat on {codec}"
        assert seen["orjson"]["summary"]["total_messages"] == before + turn
        assert MOCK_MODEL_ID in [entry["model_id"] for entry in seen["orjson"]["models"]["models"]]
