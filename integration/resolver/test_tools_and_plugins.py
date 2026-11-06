"""Journey: tool and plugin code reaching outside services by host name, under both resolvers.

`AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER` also sets the default resolver for every aiohttp session, plugin
code's included. Each test names a local service by `localhost`, a hosts-file name for `::1` alone
or a hosts-file name for another local address (`localhost` alone where the fake binds only
127.0.0.1) and expects the same outcome under both resolvers. An admin imports a Tool and a Function
from a URL. The code interpreter's Jupyter engine runs a cell for a chat's code block, for the
`execute_code` tool and for the code execution endpoint. A Pipelines server added as an OpenAI
connection is listed, has pipelines uploaded, added, deleted and configured, filters a chat's
request and reply and answers a chat as a pipe. A Tool, a pipe and a filter written as plugin
authors write them open their own `aiohttp.ClientSession()` to a service by name during a chat. A
name that does not resolve fails each of those with the same status and message, the resolver's own
wording aside; that wording, in a pipe's error, proves plugin code gets the resolver the flag picks.

Discriminates: on dev 176d31d1d, a backend copy whose `env.py` installs a resolver that fails every
lookup when the flag is on turns every c-ares run of a by-name test red and leaves every threaded
run green; the same resolver installed for the flag off does the reverse. A failure test stays green
under a failing resolver by design, and a resolver that takes 20 seconds to refuse an unknown name
turns it red. Its c-ares runs skip, naming the reason, on a machine whose DNS server keeps c-ares
from refusing an unknown name at once (a cached reply with a stale EDNS cookie, c-ares issues 1081
and 1271). Dropping the flag's branch from `env.py` fails the threaded half of the plugin test,
reading the flag as always off its c-ares half.
"""

from __future__ import annotations

import json

import pytest

from harness import upstream as reply
from harness.backends import wait_until
from harness.chat import ask
from harness.code_interpreter import fake_jupyter
from harness.host_names import (
    C_ARES_REASONS,
    FAILS_WITHIN,
    UNRESOLVABLE,
    by_name,
    name_forms,
    resolver_reason,
    serving_by_name,
    timed,
    without_resolver_reason,
)
from harness.listener import json_answer, text_answer
from harness.plugins import installed_function
from harness.python_tools import python_tool
from harness.second_provider import OPENAI_CONFIG, sse
from harness.tool_calls import without_hidden_calls
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

CODE_EXECUTION_CONFIG = "/api/v1/configs/code_execution"
PIPELINES = "/api/v1/pipelines"
TAG_FORMAT = '<code_interpreter type="code" lang="python">'
UNRESOLVABLE_URL = f"http://{UNRESOLVABLE}:9099"
OUTLET_WAIT = 30.0

TOOL_SOURCE = '''
class Tools:
    def label(self) -> str:
        """Say which tool this is."""
        return "imported tool"
'''
FUNCTION_SOURCE = """
class Pipe:
    def pipe(self, body: dict) -> str:
        return "imported function"
"""


# --- importing a Tool or a Function from a URL --------------------------------------------


@pytest.mark.parametrize("name_form", name_forms())
@pytest.mark.parametrize("kind,source", [("tools", TOOL_SOURCE), ("functions", FUNCTION_SOURCE)])
def test_a_tool_and_a_function_are_imported_from_a_url_by_name(
    resolver, name_form, kind, source, resolving_admin
):
    with serving_by_name(name_form) as server, resolving_admin.client() as client:
        server.route("GET", f"/{kind}/named_plugin.py", text_answer(source, "text/x-python"))
        imported = client.post(
            f"/api/v1/{kind}/load/url", json={"url": f"{server.base_url}/{kind}/named_plugin.py"}
        )

    assert imported.status_code == 200, imported.text
    assert imported.json() == {"name": "named_plugin", "content": source}
    assert len(server.requests_to(f"/{kind}/named_plugin.py")) == 1


@pytest.mark.usefixtures("refuses_unknown_names")
@pytest.mark.parametrize(
    "kind,detail",
    [("tools", "Error fetching tool"), ("functions", "Error fetching function")],
)
def test_importing_from_an_unresolvable_url_fails_the_same_way(
    resolver, kind, detail, resolving_admin
):
    with resolving_admin.client() as client:
        imported, seconds = timed(
            client.post,
            f"/api/v1/{kind}/load/url",
            json={"url": f"http://{UNRESOLVABLE}:8000/{kind}/named_plugin.py"},
        )

    assert (imported.status_code, imported.json()) == (500, {"detail": f"[ERROR: {detail}]"})
    assert seconds < FAILS_WITHIN, f"the lookup failed only after {seconds:.1f}s"


# --- the code interpreter's Jupyter engine ------------------------------------------------


def _use_kernel(client, url: str) -> None:
    current = client.get(CODE_EXECUTION_CONFIG).json()
    on_kernel = {
        **current,
        "ENABLE_CODE_EXECUTION": True,
        "CODE_EXECUTION_ENGINE": "jupyter",
        "CODE_EXECUTION_JUPYTER_URL": url,
        "CODE_EXECUTION_JUPYTER_AUTH": "",
        "ENABLE_CODE_INTERPRETER": True,
        "CODE_INTERPRETER_ENGINE": "jupyter",
        "CODE_INTERPRETER_JUPYTER_URL": url,
        "CODE_INTERPRETER_JUPYTER_AUTH": "",
    }
    client.post(CODE_EXECUTION_CONFIG, json=on_kernel).raise_for_status()


def _run_in_chat(instance, client, path: str, code: str) -> dict:
    """The stored reply after the model ran `code` through the tag or the execute_code tool."""
    if path == "tag":
        model_reply = reply.text(f"{TAG_FORMAT}\n{code}\n</code_interpreter>")
        function_calling = "legacy"
    else:
        model_reply = reply.tool_call("execute_code", {"code": code})
        function_calling = "native"
    instance.upstream.queue(model_reply, reply.text("done"))
    _, message = ask(
        client,
        "run it",
        features={"code_interpreter": True},
        params={"function_calling": function_calling},
    )
    return message


@pytest.mark.parametrize("path", ["tag", "tool", "endpoint"])
def test_the_jupyter_engine_runs_a_cell_by_name(
    resolver, path, resolving_instance, resolving_admin, preserve
):
    preserve((CODE_EXECUTION_CONFIG, CODE_EXECUTION_CONFIG), on=resolving_instance)
    code = "print('sum is', 6 * 7)"
    with fake_jupyter() as kernel, resolving_admin.client() as client:
        _use_kernel(client, by_name(kernel.base_url, "localhost"))
        if path == "endpoint":
            executed = client.post("/api/v1/utils/code/execute", json={"code": code})
            outcome = executed.text
        else:
            outcome = json.dumps(_run_in_chat(resolving_instance, client, path, code))

    assert "sum is 42" in outcome, outcome
    assert kernel.executed_cells()[-1].endswith(code)


def _stderr_of(message: dict) -> str:
    """What the cell's engine reported on stderr, from the code block or the tool's result."""
    [outcome] = [
        item
        for item in without_hidden_calls(message["output"])
        if item["type"] in ("open_webui:code_interpreter", "function_call_output")
    ]
    if outcome["type"] == "open_webui:code_interpreter":
        return outcome["output"]["stderr"]
    return json.loads(outcome["output"][0]["text"])["stderr"]


@pytest.mark.usefixtures("refuses_unknown_names")
@pytest.mark.parametrize("path", ["tag", "tool", "endpoint"])
def test_an_unresolvable_jupyter_engine_reports_the_same_error(
    resolver, path, resolving_instance, resolving_admin, preserve
):
    preserve((CODE_EXECUTION_CONFIG, CODE_EXECUTION_CONFIG), on=resolving_instance)
    with resolving_admin.client() as client:
        _use_kernel(client, f"http://{UNRESOLVABLE}:8888")
        if path == "endpoint":
            executed, seconds = timed(
                client.post, "/api/v1/utils/code/execute", json={"code": "print(1)"}
            )
            outcome = executed.json()["stderr"]
        else:
            stored, seconds = timed(_run_in_chat, resolving_instance, client, path, "print(1)")
            outcome = _stderr_of(stored)

    assert without_resolver_reason(outcome) == (
        f"Error: Cannot connect to host {UNRESOLVABLE}:8888 ssl:default [<resolver reason>]"
    )
    assert seconds < FAILS_WITHIN, f"the lookup failed only after {seconds:.1f}s"


# --- a Pipelines server as an OpenAI connection -------------------------------------------

FILTER_ID = "named_filter"
PIPE_ID = "named_pipe"
PIPELINES_KEY = "0p3n-w3bu!"
PIPELINE_MODELS = {
    "data": [
        {
            "id": FILTER_ID,
            "name": "Named filter",
            "object": "model",
            "pipeline": {"type": "filter", "pipelines": ["*"], "priority": 0, "valves": True},
        },
        {"id": PIPE_ID, "name": "Named pipe", "object": "model", "pipeline": {"type": "pipe"}},
    ],
    "pipelines": True,
}


def _connect_pipelines(client, url: str) -> int:
    """Add the Pipelines server as an OpenAI connection; its index in the connection list."""
    connections = client.get(OPENAI_CONFIG[0]).json()
    index = len(connections["OPENAI_API_BASE_URLS"])
    updated = {
        **connections,
        "OPENAI_API_BASE_URLS": [*connections["OPENAI_API_BASE_URLS"], url],
        "OPENAI_API_KEYS": [*connections["OPENAI_API_KEYS"], PIPELINES_KEY],
        "OPENAI_API_CONFIGS": {
            **connections["OPENAI_API_CONFIGS"],
            str(index): {"enable": True, "model_ids": []},
        },
    }
    client.post(OPENAI_CONFIG[1], json=updated).raise_for_status()
    return index


def _inlet(request):
    body = request.json()["body"]
    body["messages"][-1]["content"] += " [seen by the filter]"
    return json_answer(body)


def _outlet(request):
    return json_answer(request.json()["body"])


def _serve_pipelines(server) -> None:
    server.route("GET", "/models", json_answer(PIPELINE_MODELS))
    server.route("GET", "/pipelines", json_answer({"data": [{"id": FILTER_ID, "valves": True}]}))
    server.route("POST", "/pipelines/upload", json_answer({"status": True, "id": "uploaded"}))
    server.route("POST", "/pipelines/add", json_answer({"status": True, "id": "added"}))
    server.route("DELETE", "/pipelines/delete", json_answer({"status": True, "id": "gone"}))
    server.route("GET", f"/{FILTER_ID}/valves", json_answer({"priority": 0}))
    server.route("GET", f"/{FILTER_ID}/valves/spec", json_answer({"properties": {"priority": {}}}))
    server.route("POST", f"/{FILTER_ID}/valves/update", json_answer({"priority": 3}))
    server.route("POST", f"/{FILTER_ID}/filter/inlet", _inlet)
    server.route("POST", f"/{FILTER_ID}/filter/outlet", _outlet)
    server.route("POST", f"/{PIPE_ID}/filter/inlet", _outlet)
    server.route("POST", f"/{PIPE_ID}/filter/outlet", _outlet)
    server.route("POST", "/chat/completions", lambda _request: sse({"content": "piped by name"}))


@pytest.mark.parametrize("name_form", name_forms())
def test_a_pipelines_server_is_managed_by_name(
    resolver, name_form, resolving_instance, resolving_admin, preserve
):
    preserve(OPENAI_CONFIG, on=resolving_instance)
    with serving_by_name(name_form) as server, resolving_admin.client() as client:
        _serve_pipelines(server)
        index = _connect_pipelines(client, server.base_url)
        params = {"urlIdx": index}
        listed = client.get(f"{PIPELINES}/list")
        pipelines = client.get(f"{PIPELINES}/", params=params)
        uploaded = client.post(
            f"{PIPELINES}/upload",
            data={"urlIdx": str(index)},
            files={"file": ("named_filter.py", b"class Pipeline: pass", "text/x-python")},
        )
        added = client.post(
            f"{PIPELINES}/add", json={"url": "https://example.com/p.py", "urlIdx": index}
        )
        deleted = client.request(
            "DELETE", f"{PIPELINES}/delete", json={"id": FILTER_ID, "urlIdx": index}
        )
        valves = client.get(f"{PIPELINES}/{FILTER_ID}/valves", params=params)
        spec = client.get(f"{PIPELINES}/{FILTER_ID}/valves/spec", params=params)
        updated = client.post(
            f"{PIPELINES}/{FILTER_ID}/valves/update", params=params, json={"priority": 3}
        )

    assert listed.json()["data"] == [{"url": server.base_url, "idx": index}]
    assert pipelines.json() == {"data": [{"id": FILTER_ID, "valves": True}]}
    assert uploaded.json() == {"status": True, "id": "uploaded"}, uploaded.text
    assert added.json() == {"status": True, "id": "added"}, added.text
    assert deleted.json() == {"status": True, "id": "gone"}, deleted.text
    assert valves.json() == {"priority": 0}
    assert spec.json() == {"properties": {"priority": {}}}
    assert updated.json() == {"priority": 3}
    [received] = server.requests_to("/pipelines/upload")
    assert b"class Pipeline: pass" in received.body
    assert json.loads(server.requests_to("/pipelines/delete")[0].body) == {"id": FILTER_ID}
    assert received.headers["Authorization"] == f"Bearer {PIPELINES_KEY}"


@pytest.mark.parametrize("name_form", name_forms())
def test_a_pipelines_filter_and_pipe_serve_chats_by_name(
    resolver, name_form, resolving_instance, resolving_admin, preserve
):
    preserve(OPENAI_CONFIG, on=resolving_instance)
    upstream = resolving_instance.upstream
    with serving_by_name(name_form) as server, resolving_admin.client() as client:
        _serve_pipelines(server)
        _connect_pipelines(client, server.base_url)
        client.get("/api/models").raise_for_status()
        upstream.queue(reply.text("answered by the provider"))
        _, filtered = ask(client, "hello there", model=MOCK_MODEL_ID)
        _, piped = ask(client, "hello pipe", model=PIPE_ID)
        # the outlet runs once the reply is stored as done
        outlets_called = wait_until(
            lambda: len(server.requests_to(f"/{FILTER_ID}/filter/outlet")) == 2, OUTLET_WAIT
        )

    assert filtered["content"] == "answered by the provider", filtered
    sent = reply.conversation(upstream.chat_requests()[-1])[-1]["content"]
    assert sent == "hello there [seen by the filter]"
    assert piped["content"] == "piped by name", piped
    assert len(server.requests_to(f"/{FILTER_ID}/filter/inlet")) == 2
    outlets = len(server.requests_to(f"/{FILTER_ID}/filter/outlet"))
    assert outlets_called, f"the filter's outlet ran {outlets} times, not once per chat"
    [pipe_request] = server.requests_to("/chat/completions")
    piped_prompt = reply.conversation(pipe_request.json())[-1]["content"]
    assert piped_prompt == "hello pipe [seen by the filter]"


@pytest.mark.usefixtures("refuses_unknown_names")
@pytest.mark.parametrize(
    "method,path",
    [
        pytest.param("GET", "/", id="list"),
        pytest.param("GET", f"/{FILTER_ID}/valves", id="valves"),
        pytest.param("GET", f"/{FILTER_ID}/valves/spec", id="valves-spec"),
        pytest.param("POST", f"/{FILTER_ID}/valves/update", id="valves-update"),
        pytest.param("POST", "/add", id="add"),
        pytest.param("DELETE", "/delete", id="delete"),
        pytest.param("POST", "/upload", id="upload"),
    ],
)
def test_an_unresolvable_pipelines_server_fails_every_call_the_same_way(
    resolver, method, path, resolving_instance, resolving_admin, preserve
):
    preserve(OPENAI_CONFIG, on=resolving_instance)
    with resolving_admin.client() as client:
        index = _connect_pipelines(client, UNRESOLVABLE_URL)
        body = {"urlIdx": index, "url": "https://example.com/p.py", "id": FILTER_ID}
        if path == "/upload":
            request = {
                "data": {"urlIdx": str(index)},
                "files": {"file": ("p.py", b"class Pipeline: pass", "text/x-python")},
            }
        elif method == "GET" or "valves" in path:
            request = {"params": {"urlIdx": index}, "json": {"priority": 3}}
            request = request if method == "POST" else {"params": request["params"]}
        else:
            request = {"json": body}
        answer, seconds = timed(client.request, method, f"{PIPELINES}{path}", **request)

    assert (answer.status_code, answer.json()) == (404, {"detail": "Pipeline not found"})
    assert seconds < FAILS_WITHIN, f"the lookup failed only after {seconds:.1f}s"


# --- plugin code opening its own session --------------------------------------------------

PLUGIN_TOOL = '''
import aiohttp


class Tools:
    async def lookup(self, name: str) -> str:
        """
        Look a pet up in the pet service.
        :param name: The pet's name.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get("__URL__/pets", params={"name": name}) as response:
                return await response.text()
'''
PLUGIN_PIPE = """
import aiohttp


class Pipe:
    async def pipe(self, body: dict) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get("__URL__/greeting") as response:
                return await response.text()
"""
PLUGIN_FILTER = """
import aiohttp


class Filter:
    async def inlet(self, body: dict) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.get("__URL__/context") as response:
                note = await response.text()
        body["messages"][-1]["content"] += " " + note
        return body
"""


def _serve_plugin_service(server) -> None:
    server.route("GET", "/pets", json_answer({"found": "Rex"}))
    server.route("GET", "/greeting", text_answer("greeted by name", "text/plain"))
    server.route("GET", "/context", text_answer("[context by name]", "text/plain"))


@pytest.mark.parametrize("name_form", name_forms())
def test_a_tool_a_pipe_and_a_filter_reach_a_service_by_name(
    resolver, name_form, resolving_instance, resolving_admin
):
    upstream = resolving_instance.upstream
    with serving_by_name(name_form) as server, resolving_admin.client() as client:
        _serve_plugin_service(server)
        url = server.base_url
        with (
            python_tool(resolving_admin, PLUGIN_TOOL.replace("__URL__", url)) as tool_id,
            installed_function(resolving_admin, PLUGIN_PIPE.replace("__URL__", url)) as pipe_id,
            installed_function(
                resolving_admin, PLUGIN_FILTER.replace("__URL__", url), is_global=True
            ),
        ):
            client.get("/api/models").raise_for_status()  # registers the pipe's model
            upstream.queue(reply.tool_call("lookup", {"name": "Rex"}), reply.text("looked up"))
            _, toolled = ask(client, "find Rex", tool_ids=[tool_id])
            _, piped = ask(client, "greet me", model=pipe_id)

    sent_back = reply.conversation(upstream.chat_requests()[-1])
    assert toolled["content"] == "looked up", toolled
    assert json.loads(sent_back[-1]["content"]) == {"found": "Rex"}
    assert piped["content"] == "greeted by name", piped
    asked = reply.conversation(upstream.chat_requests()[0])[-1]["content"]
    assert asked == "find Rex [context by name]"
    assert len(server.requests_to("/pets")) == 1
    assert len(server.requests_to("/greeting")) == 1
    assert len(server.requests_to("/context")) == 2


PLUGIN_ERROR = f"Cannot connect to host {UNRESOLVABLE}:8000 ssl:default [<resolver reason>]"


@pytest.mark.usefixtures("refuses_unknown_names")
def test_a_tool_reaching_an_unresolvable_name_tells_the_model_the_same_error(
    resolver, resolving_instance, resolving_admin
):
    source = PLUGIN_TOOL.replace("__URL__", f"http://{UNRESOLVABLE}:8000")
    upstream = resolving_instance.upstream
    with python_tool(resolving_admin, source) as tool_id, resolving_admin.client() as client:
        upstream.queue(reply.tool_call("lookup", {"name": "Rex"}), reply.text("gave up"))
        (_, message), seconds = timed(ask, client, "find Rex", tool_ids=[tool_id])

    told = json.loads(reply.conversation(upstream.chat_requests()[-1])[-1]["content"])
    assert without_resolver_reason(told["error"]) == PLUGIN_ERROR
    assert message["content"] == "gave up", message
    assert seconds < FAILS_WITHIN, f"the lookup failed only after {seconds:.1f}s"


@pytest.mark.usefixtures("refuses_unknown_names")
def test_a_pipe_reaching_an_unresolvable_name_fails_the_chat_the_same_way(
    resolver, resolving_admin
):
    source = PLUGIN_PIPE.replace("__URL__", f"http://{UNRESOLVABLE}:8000")
    with installed_function(resolving_admin, source) as pipe_id, resolving_admin.client() as client:
        client.get("/api/models").raise_for_status()  # registers the pipe's model
        (_, message), seconds = timed(ask, client, "greet me", model=pipe_id)

    assert without_resolver_reason(message["error"]["content"]["detail"]) == PLUGIN_ERROR
    assert seconds < FAILS_WITHIN, f"the lookup failed only after {seconds:.1f}s"


@pytest.mark.usefixtures("refuses_unknown_names")
def test_a_filter_reaching_an_unresolvable_name_fails_the_chat_the_same_way(
    resolver, resolving_instance, resolving_admin
):
    source = PLUGIN_FILTER.replace("__URL__", f"http://{UNRESOLVABLE}:8000")
    resolving_instance.upstream.queue(reply.text("unfiltered"))
    with (
        installed_function(resolving_admin, source, is_global=True),
        resolving_admin.client() as client,
    ):
        (_, message), seconds = timed(ask, client, "find Rex")

    assert without_resolver_reason(message["error"]["content"]) == PLUGIN_ERROR
    assert seconds < FAILS_WITHIN, f"the lookup failed only after {seconds:.1f}s"


def test_plugin_code_resolves_names_with_the_resolver_the_flag_picks(resolver, resolving_admin):
    """Plugin code builds its own sessions, so only its error's wording shows which resolver ran."""
    source = PLUGIN_PIPE.replace("__URL__", f"http://{UNRESOLVABLE}:8000")
    with installed_function(resolving_admin, source) as pipe_id, resolving_admin.client() as client:
        client.get("/api/models").raise_for_status()  # registers the pipe's model
        _, message = ask(client, "greet me", model=pipe_id)

    reason = resolver_reason(message["error"]["content"]["detail"])
    assert reason, f"the error names no resolver reason: {message['error']}"
    if resolver == "c-ares":
        assert reason in C_ARES_REASONS, f"the flag is on but plugins used the OS: {reason}"
    else:
        assert reason not in C_ARES_REASONS, f"the flag is off but plugins used c-ares: {reason}"
