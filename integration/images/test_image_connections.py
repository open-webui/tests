"""Journey: several image connections at once, each with its own models and access.

The admin adds image connections under Admin Settings > Images, each a server with its own model
IDs, capabilities and access, and picks the default generation and edit models. Every account
lists only the image models of the connections it was granted, a request runs on the connection
of the model it names, and the API refuses a model the caller was not granted or one that cannot
do what was asked. With nothing named the admin's defaults run, in a chat the chat model's own
default image model wins over them and a pick from the Image entry of the + menu wins over both.
An OpenAI compatible connection with no model IDs offers every image model a LiteLLM server
labels, new ones included. One local service plays every server, one path prefix per connection,
so the tests read which server got each request and for which model.

Discriminates: passes on dev f80a4f782. Mutations in a backend copy: `get_image_models` skipping
the access check turns the listing test, the three refusals of ungranted models and both
ungranted default tests red (the outsider sees and draws on the team and vault models),
`resolve_image_model` ignoring the admin's default turns the defaults test red,
`get_requested_image_model` ignoring the model's `imageModelId` turns both chat priority cases
red and `get_synced_image_models` returning every listed model turns the LiteLLM test red (the
chat, embedding and video models are offered).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import pytest

from harness import upstream as reply
from harness.access import make_group
from harness.actors import Actor
from harness.chat import ask
from harness.image_engines import (
    EVERYONE,
    IMAGES_CONFIG,
    PNG_BASE64,
    image_connection,
    image_settings,
    save_image_settings,
)
from harness.listener import Listener, json_answer
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

PICTURE = f"data:image/png;base64,{PNG_BASE64}"
DRAWN = json_answer({"data": [{"b64_json": PNG_BASE64}]})
# what each connection offers: (connection id, model ids, generates, edits)
STUDIO_A = ("studio-a", ("a-sketch", "a-paint"), True, True)
STUDIO_B = ("studio-b", ("b-paint",), True, True)
TEAM = ("team", ("team-only",), True, False)
VAULT = ("vault", ("vault-model",), True, True)


def ref(connection_id: str, model_id: str) -> str:
    """How an image model is named across connections: its connection, then its model id."""
    return f"{connection_id}:{model_id}"


@dataclass
class Studios:
    admin: Actor
    listener: Listener
    member: Actor
    outsider: Actor

    def configure(self, generation_default: str = "", edit_default: str = "") -> None:
        with self.admin.client() as client:
            save_image_settings(
                client,
                IMAGE_GENERATION_DEFAULT_MODEL=generation_default,
                IMAGE_EDIT_DEFAULT_MODEL=edit_default,
            )

    def sent(self, connection_id: str, endpoint: str, prompt: str) -> list[str]:
        """The models `connection_id`'s server was asked for at `endpoint` with `prompt`."""
        calls = self.listener.requests_to(f"/{connection_id}/images/{endpoint}")
        if endpoint == "generations":
            return [call.json()["model"] for call in calls if call.json()["prompt"] == prompt]
        return [_form_value(call.body, "model") for call in calls if prompt.encode() in call.body]

    def drawn_on(self, endpoint: str, prompt: str) -> dict[str, list[str]]:
        """Every server that got `prompt`, with the models it was asked for."""
        servers = [STUDIO_A[0], STUDIO_B[0], TEAM[0], VAULT[0]]
        sent = {server: self.sent(server, endpoint, prompt) for server in servers}
        return {server: models for server, models in sent.items() if models}


def _form_value(body: bytes, name: str) -> str:
    marker = f'name="{name}"\r\n\r\n'.encode()
    assert marker in body, f"no {name!r} field in the form sent: {body[:400]!r}"
    return body.split(marker, 1)[1].split(b"\r\n", 1)[0].decode()


@pytest.fixture
def studios(admin, preserve, listener, make_user) -> Studios:
    """Four connections: two public, one for a group the member is in, one for admins only."""
    preserve(IMAGES_CONFIG)
    member, outsider = make_user(), make_user()
    team_group = make_group(admin, [member])
    team_grants = [{"principal_type": "group", "principal_id": team_group, "permission": "read"}]
    grants = {STUDIO_A[0]: EVERYONE, STUDIO_B[0]: EVERYONE, TEAM[0]: team_grants, VAULT[0]: []}
    connections = []
    for connection_id, models, generates, edits in (STUDIO_A, STUDIO_B, TEAM, VAULT):
        listener.route("POST", f"/{connection_id}/images/generations", DRAWN)
        listener.route("POST", f"/{connection_id}/images/edits", DRAWN)
        connections.append(
            image_connection(
                connection_id,
                "openai",
                f"{listener.base_url}/{connection_id}",
                "sk-images",
                models=models,
                generates=generates,
                edits=edits,
                access_grants=grants[connection_id],
            )
        )
    with admin.client() as client:
        save_image_settings(client, **image_settings(*connections))
    return Studios(admin, listener, member, outsider)


def listed(actor: Actor) -> dict[tuple[str, str], tuple[bool, bool]]:
    """The image models `actor` can pick from, with whether each generates and edits."""
    with actor.client() as client:
        models = client.get("/api/v1/images/models")
    assert models.status_code == 200, models.text
    return {
        (model["connection_id"], model["model"]): (model["image_generation"], model["image_edit"])
        for model in models.json()
    }


def offered(*connections: tuple) -> dict[tuple[str, str], tuple[bool, bool]]:
    return {
        (connection_id, model): (generates, edits)
        for connection_id, models, generates, edits in connections
        for model in models
    }


def generate(actor: Actor, prompt: str, model: str | None = None):
    with actor.client() as client:
        return client.post(
            "/api/v1/images/generations", json={"prompt": prompt, "model": model, "n": 1}
        )


def edit(actor: Actor, prompt: str, model: str | None = None):
    with actor.client() as client:
        return client.post(
            "/api/v1/images/edit", json={"image": PICTURE, "prompt": prompt, "model": model}
        )


def prompt_for(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:8]}"


def test_each_account_lists_only_the_image_models_it_was_granted(studios, make_user):
    other_admin = make_user(role="admin")

    assert listed(studios.outsider) == offered(STUDIO_A, STUDIO_B), (
        f"a user outside the team saw image models it was never granted: {listed(studios.outsider)}"
    )
    assert listed(studios.member) == offered(STUDIO_A, STUDIO_B, TEAM), (
        f"a team member's image models are not the public and the team ones: "
        f"{listed(studios.member)}"
    )
    assert listed(other_admin) == offered(STUDIO_A, STUDIO_B, TEAM, VAULT), (
        f"an admin does not see every connection's image models: {listed(other_admin)}"
    )


def test_a_request_runs_on_the_connection_of_the_model_it_names(studios):
    generation_prompt, edit_prompt = prompt_for("a tugboat"), prompt_for("make it red")

    generated = generate(studios.member, generation_prompt, ref("team", "team-only"))
    edited = edit(studios.outsider, edit_prompt, ref("studio-a", "a-paint"))

    assert generated.status_code == 200, generated.text
    assert studios.drawn_on("generations", generation_prompt) == {"team": ["team-only"]}, (
        "a generation naming the team model did not run on the team server alone"
    )
    assert edited.status_code == 200, edited.text
    assert studios.drawn_on("edits", edit_prompt) == {"studio-a": ["a-paint"]}, (
        "an edit naming a-paint did not run on studio A alone"
    )


@pytest.mark.parametrize(
    ("caller", "endpoint", "model"),
    [
        ("outsider", "generations", ref("team", "team-only")),
        ("outsider", "generations", ref("vault", "vault-model")),
        ("member", "edits", ref("vault", "vault-model")),
        ("member", "edits", ref("team", "team-only")),
        ("member", "generations", ref("studio-a", "no-such-model")),
    ],
    ids=[
        "another-groups-model",
        "an-admins-only-model",
        "an-admins-only-model-for-an-edit",
        "a-generation-only-model-for-an-edit",
        "a-model-no-connection-offers",
    ],
)
def test_the_api_refuses_an_image_model_the_caller_cannot_use(studios, caller, endpoint, model):
    prompt = prompt_for("a lighthouse")
    send = generate if endpoint == "generations" else edit

    refused = send(getattr(studios, caller), prompt, model)

    assert 400 <= refused.status_code < 500, (
        f"the {caller} asked for {model} and got HTTP {refused.status_code}: {refused.text}"
    )
    assert studios.drawn_on(endpoint, prompt) == {}, (
        f"a refused request for {model} still reached a server: "
        f"{studios.drawn_on(endpoint, prompt)}"
    )


def test_the_admins_default_models_run_when_nothing_is_named(studios):
    studios.configure(ref("studio-b", "b-paint"), ref("studio-a", "a-paint"))
    first_generation, first_edit = prompt_for("a harbour"), prompt_for("add fog")

    assert generate(studios.outsider, first_generation).status_code == 200
    assert edit(studios.outsider, first_edit).status_code == 200

    assert studios.drawn_on("generations", first_generation) == {"studio-b": ["b-paint"]}, (
        "a generation naming no model did not run on the default generation model"
    )
    assert studios.drawn_on("edits", first_edit) == {"studio-a": ["a-paint"]}, (
        "an edit naming no model did not run on the default edit model"
    )

    studios.configure(ref("studio-a", "a-sketch"), ref("studio-b", "b-paint"))
    second_generation, second_edit = prompt_for("a harbour"), prompt_for("add fog")
    assert generate(studios.outsider, second_generation).status_code == 200
    assert edit(studios.outsider, second_edit).status_code == 200

    assert studios.drawn_on("generations", second_generation) == {"studio-a": ["a-sketch"]}
    assert studios.drawn_on("edits", second_edit) == {"studio-b": ["b-paint"]}


def test_a_default_the_caller_was_not_granted_never_runs_for_them(studios):
    studios.configure(ref("vault", "vault-model"), ref("vault", "vault-model"))
    generation_prompt, edit_prompt = prompt_for("a crane"), prompt_for("paint it blue")

    generated = generate(studios.outsider, generation_prompt)
    edited = edit(studios.outsider, edit_prompt)

    assert generated.status_code == 200, generated.text
    assert edited.status_code == 200, edited.text
    for endpoint, prompt in (("generations", generation_prompt), ("edits", edit_prompt)):
        servers = studios.drawn_on(endpoint, prompt)
        assert len(servers) == 1 and set(servers) <= {"studio-a", "studio-b"}, (
            f"with the admins-only model as the default, a user's request ran on {servers}"
        )


@pytest.fixture
def chat_model(admin):
    """`chat_model(image_model)` is a workspace model for everyone with `image_model` as its own
    default image model; deleted afterwards."""
    created_ids = []

    def create(image_model: str) -> str:
        model_id = f"illustrator-{uuid.uuid4().hex[:8]}"
        form = {
            "id": model_id,
            "name": model_id,
            "base_model_id": MOCK_MODEL_ID,
            "meta": {"capabilities": {"image_generation": True}, "imageModelId": image_model},
            "params": {"function_calling": "native"},
            "access_grants": EVERYONE,
        }
        with admin.client() as client:
            saved = client.post("/api/v1/models/create", json=form)
            assert saved.status_code == 200, f"creating {model_id} failed: {saved.text}"
            client.get("/api/models").raise_for_status()  # registers the new model
        created_ids.append(model_id)
        return model_id

    yield create
    with admin.client() as client:
        for model_id in created_ids:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def draw_in_chat(actor: Actor, upstream, tool: str, model: str, picked: str | None) -> str:
    """Have the chat model call `tool` with the Image entry on and `picked` chosen there (None:
    Default); returns the image prompt the tool was given."""
    image_prompt = prompt_for("a quay at night")
    arguments = {"prompt": image_prompt}
    if tool == "edit_image":
        arguments["image_urls"] = [PICTURE]
    question = f"please {tool} {image_prompt}"
    asked = reply.answering(question)
    upstream.queue(reply.tool_call(tool, arguments, match=asked), reply.text("Done.", match=asked))
    feature = {"model": picked} if picked else True
    with actor.client() as client:
        _, message = ask(client, question, model=model, features={"image_generation": feature})
    images = [file for file in message.get("files") or [] if file.get("type") == "image"]
    assert len(images) == 1, f"{tool} left no picture on the reply: {message}"
    return image_prompt


@pytest.mark.parametrize(
    ("tool", "endpoint"),
    [("generate_image", "generations"), ("edit_image", "edits")],
    ids=["generating", "editing"],
)
def test_in_a_chat_the_pick_beats_the_models_default_which_beats_the_admins(
    studios, chat_model, upstream, tool, endpoint
):
    studios.configure(ref("studio-a", "a-sketch"), ref("studio-a", "a-sketch"))
    illustrator = chat_model(ref("studio-b", "b-paint"))
    user = studios.member

    with_default = draw_in_chat(user, upstream, tool, illustrator, picked=None)
    assert studios.drawn_on(endpoint, with_default) == {"studio-b": ["b-paint"]}, (
        "with Default picked, the chat did not use its model's own default image model"
    )

    with_pick = draw_in_chat(user, upstream, tool, illustrator, ref("studio-a", "a-paint"))
    assert studios.drawn_on(endpoint, with_pick) == {"studio-a": ["a-paint"]}, (
        "the image model picked for the chat did not win over the model's default"
    )

    plain_chat = draw_in_chat(user, upstream, tool, MOCK_MODEL_ID, picked=None)
    assert studios.drawn_on(endpoint, plain_chat) == {"studio-a": ["a-sketch"]}, (
        "a chat model without a default image model did not use the admin's default"
    )


def test_a_chat_never_draws_on_an_image_model_its_user_was_not_granted(
    studios, chat_model, upstream
):
    studios.configure(ref("studio-a", "a-sketch"), ref("studio-a", "a-sketch"))
    vault_by_default = chat_model(ref("vault", "vault-model"))

    picked = draw_in_chat(
        studios.outsider, upstream, "generate_image", MOCK_MODEL_ID, ref("team", "team-only")
    )
    defaulted = draw_in_chat(studios.outsider, upstream, "generate_image", vault_by_default, None)

    assert studios.drawn_on("generations", picked) == {"studio-a": ["a-sketch"]}, (
        "a pick of another group's image model was not replaced by the admin's default"
    )
    assert studios.drawn_on("generations", defaulted) == {"studio-a": ["a-sketch"]}, (
        "a chat model's admins-only default image model was not replaced by the admin's default"
    )


# LiteLLM's /model/info: each model's mode, and for some the endpoints it serves
LITELLM_MODELS = {
    "gpt-4o": {"mode": "chat"},
    "dall-e-3": {"mode": "image_generation"},
    "flux-kontext": {"mode": "image_edit"},
    "gpt-image-1": {
        "mode": "image_generation",
        "supported_endpoints": ["/v1/images/generations", "/v1/images/edits"],
    },
    "sora-2": {"mode": "video_generation"},
    "text-embedding-3-small": {"mode": "embedding"},
}
LITELLM_IMAGE_MODELS = {
    "dall-e-3": (True, False),
    "flux-kontext": (False, True),
    "gpt-image-1": (True, True),
}


def serve_litellm(listener: Listener, models: dict[str, dict]) -> None:
    listing = [{"id": model_id, "object": "model"} for model_id in models]
    info = [{"model_name": model_id, "model_info": entry} for model_id, entry in models.items()]
    listener.route("GET", "/lite/models", json_answer({"object": "list", "data": listing}))
    listener.route("GET", "/lite/model/info", json_answer({"data": info}))


def test_a_litellm_connection_without_model_ids_offers_its_image_models(
    admin, preserve, listener, make_user
):
    preserve(IMAGES_CONFIG)
    serve_litellm(listener, LITELLM_MODELS)
    listener.route("POST", "/lite/images/generations", DRAWN)
    litellm = image_connection("litellm", "openai", f"{listener.base_url}/lite", "sk-litellm")
    with admin.client() as client:
        save_image_settings(client, **image_settings(litellm))
    user = make_user()

    expected = {("litellm", model): flags for model, flags in LITELLM_IMAGE_MODELS.items()}
    assert listed(user) == expected, (
        f"the LiteLLM connection did not offer exactly its image models: {listed(user)}"
    )

    prompt = prompt_for("a lighthouse")
    generated = generate(user, prompt, ref("litellm", "gpt-image-1"))
    assert generated.status_code == 200, generated.text
    sent = [call.json() for call in listener.requests_to("/lite/images/generations")]
    assert [body["model"] for body in sent if body["prompt"] == prompt] == ["gpt-image-1"]
    refused = edit(user, prompt_for("make it red"), ref("litellm", "dall-e-3"))
    assert 400 <= refused.status_code < 500, (
        f"the API edited with a LiteLLM model that only generates: {refused.text}"
    )

    serve_litellm(listener, {**LITELLM_MODELS, "imagen-4": {"mode": "image_generation"}})
    deadline = time.monotonic() + 15
    while ("litellm", "imagen-4") not in listed(user) and time.monotonic() < deadline:
        time.sleep(0.5)
    assert listed(user) == {**expected, ("litellm", "imagen-4"): (True, False)}, (
        "an image model added on the LiteLLM server never showed up for the user"
    )
