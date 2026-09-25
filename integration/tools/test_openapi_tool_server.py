"""Journey: a model calling the operations of an OpenAPI tool server during a chat.

An admin adds a tool server by the URL of its OpenAPI spec; each operation becomes a tool whose
parameters come from its path, query and path-level parameters and its JSON request body
(`$ref` schemas resolved). When the model calls one, Open WebUI fills the path, encodes the
query, sends only the body's own fields as JSON and hands the model back what the server
answered: JSON, plain text, an error with its status, an image the model can see, or an inline
HTML card shown to the user in place of the raw page.

Discriminates: in a backend copy, dropping path-level parameters fails the spec test, skipping
the path escaping fails the path test, sending declared parameters in the body fails the body
test, dropping the status from a failed call fails the error test, and treating an inline HTML
answer as plain text fails the card test.
"""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlsplit

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.listener import json_answer, text_answer
from harness.tool_calls import run_tool

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

TOOL_SERVERS_CONFIG = ("/api/v1/configs/tool_servers", "/api/v1/configs/tool_servers")
SERVER_ID = "pets"
TOOL_IDS = [f"server:{SERVER_ID}"]
PREFIX = "/pets-api"
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)
CARD = "<html><body><h1>Rex</h1></body></html>"


def _operation(operation_id: str, **extra) -> dict:
    return {"operationId": operation_id, "responses": {"200": {"description": "ok"}}, **extra}


def _query(name: str, schema: dict, **extra) -> dict:
    return {"name": name, "in": "query", "schema": schema, **extra}


PET_ID = {"name": "pet_id", "in": "path", "required": True, "schema": {"type": "string"}}

SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Pets", "version": "1"},
    "components": {
        "schemas": {
            "NewPet": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "description": "The pet's name"},
                    "age": {"type": "integer"},
                },
            }
        }
    },
    "paths": {
        "/pets": {
            "get": _operation(
                "list_pets",
                summary="List the pets.",
                parameters=[
                    _query("limit", {"type": "integer", "description": "At most this many"}),
                    _query("tag", {"type": "array", "items": {"type": "string"}}),
                    _query("status", {"type": "string", "enum": ["available", "sold"]}),
                ],
            ),
            "post": _operation(
                "create_pet",
                description="Add a pet.",
                parameters=[_query("dry_run", {"type": "boolean"})],
                requestBody={
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/NewPet"}}
                    }
                },
            ),
        },
        "/pets/{pet_id}": {
            "parameters": [PET_ID, _query("lang", {"type": "string"}, description="Language")],
            "get": _operation("get_pet"),
            "delete": _operation("delete_pet"),
        },
        "/pets/{pet_id}/photo": {"parameters": [PET_ID], "get": _operation("get_photo")},
        "/pets/{pet_id}/card": {"parameters": [PET_ID], "get": _operation("get_card")},
        "/report": {"get": _operation("get_report")},
        "/missing": {"get": _operation("get_missing")},
    },
}


@pytest.fixture
def pets_server(admin, preserve, listener):
    """The pets API on the listener, added by the admin as a tool server."""
    listener.route("GET", f"{PREFIX}/openapi.json", json_answer(SPEC))
    listener.route("GET", f"{PREFIX}/pets", json_answer([{"id": "1", "name": "Rex"}]))
    listener.route("POST", f"{PREFIX}/pets", json_answer({"id": "2", "created": True}))
    listener.route("GET", f"{PREFIX}/pets/a%20b%2Fc", json_answer({"id": "a b/c", "name": "Odd"}))
    listener.route("DELETE", f"{PREFIX}/pets/1", text_answer("database down", "text/plain", 500))
    listener.route("GET", f"{PREFIX}/pets/1/photo", (200, {"Content-Type": "image/png"}, PNG))
    card_headers = {"Content-Type": "text/html", "Content-Disposition": "inline"}
    listener.route("GET", f"{PREFIX}/pets/1/card", (200, card_headers, CARD.encode()))
    listener.route("GET", f"{PREFIX}/report", text_answer("3 pets, all happy", "text/plain"))
    listener.route("GET", f"{PREFIX}/missing", json_answer({"detail": "no such thing"}, 404))
    connection = {
        "url": f"{listener.base_url}{PREFIX}",
        "path": "openapi.json",
        "auth_type": "none",
        "key": "",
        "config": {"enable": True},
        "info": {"id": SERVER_ID, "name": "Pets"},
    }
    preserve(TOOL_SERVERS_CONFIG)
    with admin.client() as client:
        saved = client.post(TOOL_SERVERS_CONFIG[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
        assert saved.status_code == 200, saved.text
    return listener


@pytest.fixture
def owner(make_user):
    return make_user(role="admin")


def _offered_tools(upstream) -> dict[str, dict]:
    tools = upstream.chat_requests()[0]["tools"]
    return {tool["function"]["name"]: tool["function"] for tool in tools}


def _call(owner, upstream, name: str, arguments: dict) -> str:
    with owner.client() as client:
        return run_tool(client, upstream, name, arguments, tool_ids=TOOL_IDS)


def test_every_operation_is_offered_with_its_parameters(owner, upstream, pets_server):
    _call(owner, upstream, "get_report", {})

    offered = _offered_tools(upstream)
    get_pet = offered["get_pet"]["parameters"]
    assert set(get_pet["properties"]) == {"pet_id", "lang"}
    assert get_pet["required"] == ["pet_id"]
    assert get_pet["properties"]["lang"]["description"] == "Language"
    list_pets = offered["list_pets"]
    assert list_pets["description"] == "List the pets."
    assert list_pets["parameters"]["properties"]["tag"] == {
        "type": "array",
        "description": "",
        "items": {"type": "string"},
    }
    assert list_pets["parameters"]["properties"]["status"]["description"].endswith(
        "Possible values: available, sold"
    )
    create_pet = offered["create_pet"]["parameters"]
    assert set(create_pet["properties"]) == {"dry_run", "name", "age"}
    assert create_pet["required"] == ["name"]
    assert create_pet["properties"]["name"]["description"] == "The pet's name"


def test_a_path_parameter_is_escaped_into_the_url(owner, upstream, pets_server):
    result = _call(owner, upstream, "get_pet", {"pet_id": "a b/c", "lang": "de"})

    assert json.loads(result) == {"id": "a b/c", "name": "Odd"}
    [sent] = pets_server.requests_to(f"{PREFIX}/pets/a%20b%2Fc")
    assert parse_qs(urlsplit(sent.path).query) == {"lang": ["de"]}


def test_query_parameters_are_encoded_and_empty_ones_left_out(owner, upstream, pets_server):
    arguments = {"limit": 5, "status": "", "tag": "good boy", "unknown": "dropped"}

    result = _call(owner, upstream, "list_pets", arguments)

    assert json.loads(result) == {"results": [{"id": "1", "name": "Rex"}]}
    [sent] = pets_server.requests_to(f"{PREFIX}/pets")
    assert parse_qs(urlsplit(sent.path).query) == {"limit": ["5"], "tag": ["good boy"]}


def test_a_body_carries_only_its_own_fields(owner, upstream, pets_server):
    result = _call(owner, upstream, "create_pet", {"name": "Bo", "age": 2, "dry_run": True})

    assert json.loads(result) == {"id": "2", "created": True}
    [sent] = [
        entry for entry in pets_server.requests_to(f"{PREFIX}/pets") if entry.method == "POST"
    ]
    assert sent.json() == {"name": "Bo", "age": 2}
    assert parse_qs(urlsplit(sent.path).query) == {"dry_run": ["True"]}


def test_a_plain_text_answer_reaches_the_model_as_is(owner, upstream, pets_server):
    assert _call(owner, upstream, "get_report", {}) == "3 pets, all happy"


@pytest.mark.parametrize(
    "name,arguments,expected",
    [
        pytest.param("delete_pet", {"pet_id": "1"}, "HTTP error 500: database down", id="500-text"),
        pytest.param(
            "get_missing", {}, 'HTTP error 404: {"detail": "no such thing"}', id="404-json"
        ),
    ],
)
def test_a_failed_call_tells_the_model_the_status(
    owner, upstream, pets_server, name, arguments, expected
):
    upstream.queue(reply.tool_call(name, arguments), reply.text("that failed"))
    with owner.client() as client:
        _, message = ask(client, f"use {name}", tool_ids=TOOL_IDS)

    sent_back = upstream.chat_requests()[-1]["messages"][-1]
    assert sent_back["role"] == "tool"
    assert json.loads(sent_back["content"]) == {"error": expected}
    [result] = [item for item in message["output"] if item["type"] == "function_call_output"]
    assert result["status"] == "failed"


def test_an_image_answer_is_shown_to_the_model(owner, upstream, pets_server):
    upstream.queue(reply.tool_call("get_photo", {"pet_id": "1"}), reply.text("a cute dog"))
    with owner.client() as client:
        _, message = ask(client, "show me", tool_ids=TOOL_IDS)

    [result] = [item for item in message["output"] if item["type"] == "function_call_output"]
    texts = [part["text"] for part in result["output"] if part["type"] == "input_text"]
    assert texts == ["get_photo: Image file read successfully."]
    [image] = [part for part in result["output"] if part["type"] == "input_image"]
    assert image["image_url"]
    sent_back = json.dumps(upstream.chat_requests()[-1]["messages"])
    assert "get_photo: Image file read successfully." in sent_back
    assert "image_url" in sent_back


def test_an_inline_html_answer_becomes_a_card_for_the_user(owner, upstream, pets_server):
    upstream.queue(reply.tool_call("get_card", {"pet_id": "1"}), reply.text("here is Rex"))
    with owner.client() as client:
        _, message = ask(client, "card please", tool_ids=TOOL_IDS)

    [result] = [item for item in message["output"] if item["type"] == "function_call_output"]
    assert result["embeds"] == [CARD]
    told = json.loads(upstream.chat_requests()[-1]["messages"][-1]["content"])
    assert told["code"] == "ui_component"
    assert CARD not in json.dumps(upstream.chat_requests()[-1])
