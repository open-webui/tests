"""Journey: the skills kept on a terminal an admin connected reach the model of a chat on it.

Open WebUI reads such a terminal itself. With the terminal picked, each skill the terminal lists
(`GET /skills`) is offered to the model with its id `terminal:<name>`, marked as the terminal's
and with where it lies; a chat without the terminal lists none, and a terminal that refuses the
list (401, 403, 404, 500) leaves the reply without them. `view_skill` reads one from the terminal
as the person (`GET /skills/read?name=`): its instructions, its folder and its files.
`read_skill_file` and `grep_skill_files` read a terminal skill's SKILL.md only and tell the model
to use the terminal's own tools for its other files. A skill that changed on the terminal between
`view_skill` and a later read is refused until the model views it again. On a model with the
Skills tool off, a terminal skill picked with `$` arrives whole in the system prompt, the mention
is taken out of the message, and a message that was only a mention without a label carries the
skill's name.

Discriminates: passes on dev c15f28649. In a backend copy, each turned the named tests red: the
terminal's skills never listed (list test, and the view, other files and changed skill tests,
since a skill that is not listed cannot be opened), a refused list read as a list (refused list
test), `view_skill` not reading `terminal:` ids from the terminal (view, other files and changed
skill tests), the name sent as it stands in the id (view test, spaced name), the SKILL.md-only
check dropped from `read_skill_file` or from `grep_skill_files` (other files test), the
fingerprint check dropped (changed skill test), a picked terminal skill not loaded with the
Skills tool off (picked skill and mention-only tests), and the fallback text dropped
(mention-only test).
"""

from __future__ import annotations

import contextlib
import json
import uuid
from typing import Iterator
from urllib.parse import parse_qs, urlsplit

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.listener import json_answer
from harness.python_tools import EVERYONE_READS
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    FakeTerminalServer,
    TerminalRequest,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from harness.tool_calls import skill_list
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

HOME = "/home/user"
SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "terminal", "version": "1"},
    "paths": {
        "/execute": {
            "post": {"operationId": "run_command", "responses": {"200": {"description": "ok"}}}
        }
    },
}
NOT_FOUND = json_answer({"detail": "Skill not found"}, status=404)
SKILL_MD_ONLY = "Terminal skills support SKILL.md only; use terminal tools for supporting files."
CHANGED = "Skill changed while reading. Call view_skill again and restart reading."


class SkillShelf:
    """The skills a fake terminal keeps, served on its `/skills` routes."""

    def __init__(self, server: FakeTerminalServer):
        self.server = server
        self.skills: dict[str, dict] = {}
        server.route("GET", "/skills", lambda _request: json_answer(self.listing()))
        server.route("GET", "/skills/read", self._read)

    def put(self, name: str, content: str, resources: list[str] | None = None) -> str:
        """Keep a skill; returns the id the `$` picker and the model use for it."""
        location = f"{HOME}/.skills/{name}/SKILL.md"
        self.skills[name] = {
            "name": name,
            "description": f"How to {name}",
            "content": content,
            "location": location,
            "resources": resources or [],
        }
        return self.listing()[-1]["id"]

    def listing(self) -> list[dict]:
        return [
            {
                "id": f"terminal:{name.replace(' ', '%20')}",
                "name": name,
                "description": skill["description"],
                "location": skill["location"],
            }
            for name, skill in self.skills.items()
        ]

    def _read(self, request: TerminalRequest):
        [name] = parse_qs(urlsplit(request.path).query).get("name", [""])
        return json_answer(self.skills[name]) if name in self.skills else NOT_FOUND

    def reads(self) -> list[TerminalRequest]:
        return self.server.requests_to("/skills/read")


@pytest.fixture
def member(make_user):
    return make_user()


@pytest.fixture
def shelf(admin, preserve, member) -> Iterator[SkillShelf]:
    """A terminal the admin connected and shared with `member`, with a shelf for its skills."""
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminal() as server:
        server.route("GET", "/openapi.json", json_answer(SPEC))
        server.route("GET", "/files/cwd", json_answer({"cwd": HOME, "home": HOME}))
        connection = server.connection(config={"access_grants": [read_grant(member.id)]})
        with admin.client() as client:
            configure_terminals(client, connection)
        server.clear()
        skills = SkillShelf(server)
        skills.terminal_id = connection["id"]
        yield skills


def _tag() -> str:
    return uuid.uuid4().hex[:8]


def _requests_answering(upstream, prompt: str) -> list[dict]:
    return [request for request in upstream.chat_requests() if reply.mentioning(prompt)(request)]


def _first_request(member, upstream, prompt: str, **options) -> dict:
    """One message the model answers at once; returns what the model was sent."""
    upstream.queue(reply.text("ready", match=reply.mentioning(prompt)))
    with member.client() as client:
        _, message = ask(client, prompt, **options)
    assert message["content"] == "ready", f"the reply did not arrive: {message}"
    return _requests_answering(upstream, prompt)[0]


def _skill_calls(member, upstream, terminal_id: str, *calls: tuple[str, dict]) -> list[dict]:
    """Have the model make `calls` one after the other in one reply; returns each result."""
    prompt = f"work with the skill {_tag()}"
    mentioned = reply.mentioning(prompt)
    upstream.queue(
        *[
            reply.tool_call(name, arguments, call_id=f"call_{number}", match=mentioned)
            for number, (name, arguments) in enumerate(calls, start=1)
        ],
        reply.text("ready", match=mentioned),
    )
    with member.client() as client:
        _, message = ask(client, prompt, terminal_id=terminal_id)
    assert message["content"] == "ready", f"the reply did not arrive: {message}"
    final = reply.conversation(_requests_answering(upstream, prompt)[-1])
    results = {
        entry["tool_call_id"]: json.loads(entry["content"])
        for entry in final
        if entry["role"] == "tool"
    }
    assert len(results) == len(calls), f"not every call was answered: {results}"
    return [results[f"call_{number}"] for number in range(1, len(calls) + 1)]


def test_the_skills_of_the_picked_terminal_are_listed_for_the_model(shelf, member, upstream):
    name = f"ship-{_tag()}"
    skill_id = shelf.put(name, "Run make ship.")

    on_the_terminal = _first_request(
        member, upstream, f"what can you do {_tag()}", terminal_id=shelf.terminal_id
    )
    without = _first_request(member, upstream, f"what can you do {_tag()}")

    listed = skill_list(on_the_terminal)
    entry = (
        f"<skill>\n<id>{skill_id}</id>\n<name>{name}</name>\n"
        f"<description>How to {name}</description>\n<source>terminal</source>\n"
        f"<location>{HOME}/.skills/{name}/SKILL.md</location>\n</skill>"
    )
    assert entry in listed, f"the terminal's skill is not listed as its own: {listed!r}"
    assert skill_id not in skill_list(without), (
        "a chat without the terminal lists the terminal's skill"
    )
    [asked] = shelf.server.requests_to("/skills")
    assert asked.headers.get("x-user-id") == member.id, "the list was not read as the person"


@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_a_terminal_that_refuses_its_skill_list_leaves_the_reply_without_skills(
    shelf, member, upstream, status
):
    skill_id = shelf.put(f"ship-{_tag()}", "Run make ship.")
    shelf.server.route("GET", "/skills", json_answer(shelf.listing(), status=status))

    request = _first_request(
        member, upstream, f"what can you do {_tag()}", terminal_id=shelf.terminal_id
    )

    assert shelf.server.requests_to("/skills"), "the terminal was never asked for its skills"
    assert skill_id not in json.dumps(request["messages"]), (
        f"a skill list the terminal answered with {status} reached the model"
    )
    assert "run_command" in {tool["function"]["name"] for tool in request.get("tools") or []}


@pytest.mark.parametrize("name", ["ship-it", "release notes"], ids=["plain", "spaced"])
def test_view_skill_reads_a_terminal_skill_from_the_terminal_as_the_person(
    shelf, member, upstream, name
):
    body = f"Run make ship {_tag()}, then tag the release."
    skill_id = shelf.put(name, body, resources=["scripts/tag.sh"])

    [viewed] = _skill_calls(member, upstream, shelf.terminal_id, ("view_skill", {"id": skill_id}))

    assert viewed.get("name") == name and body in str(viewed.get("content")), (
        f"view_skill on {skill_id} gave the model {viewed}"
    )
    assert viewed.get("directory") == f"{HOME}/.skills/{name}", viewed
    assert "scripts/tag.sh" in json.dumps(viewed), f"the skill's files are not named: {viewed}"
    [read] = shelf.reads()
    assert read.headers.get("x-user-id") == member.id, "the skill was not read as the person"


def test_the_skill_file_tools_read_only_the_skill_md_of_a_terminal_skill(shelf, member, upstream):
    marker = f"anchor-{_tag()}"
    skill_id = shelf.put("ship-it", f"Step one.\nThen {marker}.\n", resources=["scripts/tag.sh"])

    viewed, page, other, found, other_search = _skill_calls(
        member,
        upstream,
        shelf.terminal_id,
        ("view_skill", {"id": skill_id}),
        ("read_skill_file", {"id": skill_id, "path": "SKILL.md"}),
        ("read_skill_file", {"id": skill_id, "path": "scripts/tag.sh"}),
        ("grep_skill_files", {"id": skill_id, "pattern": marker}),
        ("grep_skill_files", {"id": skill_id, "pattern": marker, "path": "scripts/tag.sh"}),
    )

    assert "error" not in viewed, viewed
    assert page.get("path") == "SKILL.md" and marker in str(page.get("content")), (
        f"read_skill_file did not return the terminal skill's SKILL.md: {page}"
    )
    assert marker in json.dumps(found) and "error" not in found, (
        f"grep_skill_files did not find the text in SKILL.md: {found}"
    )
    for refused in (other, other_search):
        assert SKILL_MD_ONLY in str(refused.get("error")), (
            f"another file of a terminal skill was not refused: {refused}"
        )


def test_a_terminal_skill_that_changed_since_it_was_viewed_is_not_read_on(shelf, member, upstream):
    skill_id = shelf.put("ship-it", "Run make ship.")
    served = iter(["Run make ship.", "Run make sink.", "Run make sink.", "Run make sink."])

    def changing(_request):
        return json_answer({**shelf.skills["ship-it"], "content": next(served)})

    shelf.server.route("GET", "/skills/read", changing)
    reading = ("read_skill_file", {"id": skill_id, "path": "SKILL.md"})

    viewed, stale, viewed_again, fresh = _skill_calls(
        member,
        upstream,
        shelf.terminal_id,
        ("view_skill", {"id": skill_id}),
        reading,
        ("view_skill", {"id": skill_id}),
        reading,
    )

    assert "make ship" in str(viewed.get("content")), viewed
    assert CHANGED in str(stale.get("error")), (
        f"a skill changed on the terminal was read on as if it were the one viewed: {stale}"
    )
    assert "make sink" in str(viewed_again.get("content")), viewed_again
    assert "make sink" in str(fresh.get("content")) and "error" not in fresh, (
        f"after viewing the skill again it still cannot be read: {fresh}"
    )


@contextlib.contextmanager
def _model_without_the_skills_tool(admin) -> Iterator[str]:
    """A model with the editor's Skills switch off, as the model editor saves it."""
    model_id = f"no-skills-tool-{_tag()}"
    form = {
        "id": model_id,
        "name": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "meta": {"capabilities": {}, "builtinTools": {"skills": False}},
        "params": {"function_calling": "native"},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, f"creating {model_id} failed: {created.text}"
        try:
            client.get("/api/models").raise_for_status()  # registers the model for chats
            yield model_id
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def _system_text(request: dict) -> str:
    return "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )


def test_a_terminal_skill_picked_with_the_skills_tool_off_arrives_whole(
    admin, shelf, member, upstream
):
    body = f"Run make ship {_tag()}, then tag the release."
    skill_id = shelf.put("ship-it", body)
    other_body = f"Never deploy on a Friday {_tag()}."
    shelf.put("deploy-rules", other_body)
    prompt = f"get the release out {_tag()}"

    with _model_without_the_skills_tool(admin) as model_id:
        request = _first_request(
            member,
            upstream,
            prompt,
            terminal_id=shelf.terminal_id,
            model=model_id,
            skill_ids=[skill_id],
        )

    system = _system_text(request)
    assert '<skill name="ship-it">' in system and body in system, (
        f"the picked terminal skill did not reach the model whole: {system!r}"
    )
    assert other_body not in system, "a terminal skill nobody picked was loaded as well"


def test_a_message_that_only_mentions_a_terminal_skill_carries_its_name(
    admin, shelf, member, upstream
):
    name = f"ship-{_tag()}"
    body = f"Run make ship {_tag()}."
    skill_id = shelf.put(name, body)
    upstream.queue(reply.text("ready", match=reply.mentioning(name)))

    with _model_without_the_skills_tool(admin) as model_id, member.client() as client:
        _, message = ask(client, f"<${skill_id}>", terminal_id=shelf.terminal_id, model=model_id)

    assert message["content"] == "ready", f"the reply did not arrive: {message}"
    [request] = _requests_answering(upstream, name)
    asked = [entry["content"] for entry in request["messages"] if entry["role"] == "user"][-1]
    assert asked == name, f"the model was asked {asked!r} for a message that was only the mention"
    assert body in _system_text(request), "the mentioned terminal skill was not loaded"
