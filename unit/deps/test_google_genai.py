"""Dependency contract: google-genai (import ``from google import genai``).

google-genai is the official Google Gen AI SDK ("python-genai"), pinned in
``backend/requirements.txt`` (``google-genai==1.66.0``). It is the client
for Gemini models / native function-calling.

IMPORTANT — usage note: at this HEAD the Open WebUI backend does NOT
import ``google.genai`` in its own Python. The only references are
comments in ``utils/tools.py`` explaining tool-signature handling for
"python-genai" native function calling (it strips frozen/partial kwargs so
genai can infer tool properties, working around
googleapis/python-genai#907). Gemini access in Open WebUI otherwise flows
through the OpenAI-compatible / direct provider HTTP layer, not this SDK
object. So google-genai is a *declared* dependency with no first-party
call sites to pin keyword arguments against.

This module therefore pins the SDK's *core public surface* — the namespace
package import (``from google import genai``), the ``Client`` entrypoint
and the constructor knobs that matter for tool/function-calling usage, the
``types`` model objects (``Tool`` / ``FunctionDeclaration`` /
``GenerateContentConfig`` / ``Content`` / ``Part``) the tool-signature
comment alludes to, and the ``errors`` hierarchy — and constructs a
``Client`` OFFLINE (no network: the client is lazy, only making HTTP calls
when a method is invoked, which we never do here).

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "google.genai"
DIST_NAME = "google-genai"

# Top-level genai surface.
PACKAGE_SYMBOLS = [
    "Client",  # the SDK entrypoint
    "types",  # request/response + tool model objects
    "errors",  # exception hierarchy
    "models",  # models namespace (client.models.* equivalent)
]

# types submodule objects relevant to native function-calling / generation.
TYPES_SYMBOLS = [
    "Tool",
    "FunctionDeclaration",
    "GenerateContentConfig",
    "Content",
    "Part",
    "Schema",
]

# errors submodule.
ERROR_SYMBOLS = [
    "APIError",  # base API error
    "ClientError",  # 4xx
    "ServerError",  # 5xx
]


def test_import_namespace_package(depcheck):
    """The SDK is imported as `from google import genai` (a namespace package).
    The genai module must load and report the dotted name."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "google.genai"


def test_from_google_import_genai_form(depcheck):
    """Pin the exact import form the ecosystem uses: `from google import genai`.
    Resolve `google` then its `genai` attribute (importing the submodule)."""
    google_pkg = depcheck.load("google")
    # Trigger submodule import then resolve as an attribute.
    depcheck.load("google.genai")
    assert depcheck.has(google_pkg, "genai"), "`from google import genai` no longer resolves"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_package_symbols_exist(depcheck):
    """Client/types/errors/models must remain on the genai package."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, PACKAGE_SYMBOLS)


def test_client_is_class(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.Client)


def test_client_constructor_accepts_core_kwargs(depcheck):
    """Client(api_key=, vertexai=, project=, location=, http_options=) are the
    standard construction knobs. Pin the names so a swap-in / config wiring
    stays valid."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.Client.__init__,
        ["api_key", "vertexai", "project", "location", "http_options"],
    )


def test_client_constructs_offline(depcheck):
    """A Client built with a dummy api_key must not perform any network I/O at
    construction (calls are lazy). It must expose the `models` accessor (and the
    async `aio` accessor) used for generate_content."""
    mod = depcheck.load(IMPORT_NAME)
    client = mod.Client(api_key="dummy-offline-key")
    assert client is not None
    # The generate-content surface hangs off client.models (sync) / client.aio.
    assert hasattr(client, "models"), "Client lost its .models accessor"
    assert hasattr(client, "aio"), "Client lost its async .aio accessor"


def test_client_models_has_generate_content(depcheck):
    """The whole point of the SDK is generate_content / generate_content_stream.
    Pin those exist on the client's models accessor (offline; not invoked)."""
    mod = depcheck.load(IMPORT_NAME)
    client = mod.Client(api_key="dummy-offline-key")
    models = client.models
    for name in ("generate_content", "generate_content_stream"):
        assert hasattr(models, name), f"client.models.{name} missing"
        assert callable(getattr(models, name))


def test_types_symbols_exist(depcheck):
    """The tool/function-calling and generation model objects the SDK exposes
    (referenced conceptually by utils/tools.py's signature handling) must
    remain importable."""
    depcheck.load(IMPORT_NAME)
    types_mod = depcheck.load("google.genai.types")
    depcheck.assert_symbols(types_mod, TYPES_SYMBOLS)


def test_function_declaration_constructs_offline(depcheck):
    """utils/tools.py adjusts function signatures so genai can build a tool/
    function declaration. Build a FunctionDeclaration offline to prove the
    pydantic model + its name/description/parameters fields are intact."""
    depcheck.load(IMPORT_NAME)
    types_mod = depcheck.load("google.genai.types")
    fd = types_mod.FunctionDeclaration(
        name="get_weather",
        description="Return the weather for a city.",
        parameters=types_mod.Schema(
            type="OBJECT",
            properties={"city": types_mod.Schema(type="STRING")},
            required=["city"],
        ),
    )
    assert fd.name == "get_weather"
    assert fd.description.startswith("Return the weather")


def test_tool_wraps_function_declarations(depcheck):
    """A Tool bundles FunctionDeclarations — the object genai turns a Python
    function into for native function calling. Build one offline."""
    depcheck.load(IMPORT_NAME)
    types_mod = depcheck.load("google.genai.types")
    tool = types_mod.Tool(
        function_declarations=[
            types_mod.FunctionDeclaration(name="noop", description="does nothing"),
        ]
    )
    assert tool.function_declarations is not None
    assert len(tool.function_declarations) == 1
    assert tool.function_declarations[0].name == "noop"


def test_generate_content_config_accepts_tools(depcheck):
    """GenerateContentConfig is where tools are attached for a request. Build it
    offline with a tools list to pin the field."""
    depcheck.load(IMPORT_NAME)
    types_mod = depcheck.load("google.genai.types")
    cfg = types_mod.GenerateContentConfig(
        temperature=0.0,
        tools=[types_mod.Tool(function_declarations=[])],
    )
    assert cfg.temperature == 0.0
    assert cfg.tools is not None


def test_content_and_part_construct_offline(depcheck):
    """Content/Part are the message building blocks. Part.from_text(...) and a
    Content(role=, parts=) must build offline."""
    depcheck.load(IMPORT_NAME)
    types_mod = depcheck.load("google.genai.types")
    part = types_mod.Part(text="hello")
    content = types_mod.Content(role="user", parts=[part])
    assert content.role == "user"
    assert content.parts[0].text == "hello"


def test_errors_symbols_exist_and_hierarchy(depcheck):
    """The errors module must expose APIError/ClientError/ServerError, with the
    HTTP-status subclasses rooted at APIError so a broad `except APIError`
    catches both 4xx and 5xx."""
    depcheck.load(IMPORT_NAME)
    errors_mod = depcheck.load("google.genai.errors")
    depcheck.assert_symbols(errors_mod, ERROR_SYMBOLS)
    base = errors_mod.APIError
    assert issubclass(errors_mod.ClientError, base)
    assert issubclass(errors_mod.ServerError, base)


def test_not_imported_by_backend_marker():
    """Documentation guard (no dep assertion): records that the backend does not
    import google.genai at this HEAD — it's a declared dependency referenced
    only in utils/tools.py comments about native function-calling signature
    handling. The behavioural pins above guard the surface should it be wired
    in (or a bump break it)."""
    assert True
