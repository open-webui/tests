"""Regression: an ordinary image-generation request must not switch the
instance-wide Automatic1111 checkpoint.

open-webui 0.11.0 fix `8becf9443` (#27244): the automatic1111 branch of
`image_generations` called `set_image_model` whenever the request carried a
`model` field. `set_image_model` is not request-scoped: it persists
`image_generation.model` to the global config and POSTs the new
`sd_model_checkpoint` to the shared Automatic1111 server, which holds one
checkpoint for the whole instance. Any non-admin with the image-generation
permission could therefore repoint the image model for every user by adding a
`model` to a normal generation request. The fix gates that call on
`user.role == 'admin'`.

0.11.2 `aeb126b95` changed `upload_image` to return a file descriptor dict that
`image_generations` appends verbatim, so the stub and the return assertion follow that
shape; the admin gate itself is unchanged.

Discriminates: passes on v0.11.0 through v0.11.3, fails on v0.10.2 (a non-admin's
`model` still reaches `set_image_model`, so the checkpoint POST and the config write
happen).
"""

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

BASE_URL = "http://a1111.local:7860"
CONFIGURED_CHECKPOINT = "configured.safetensors"
ATTACKER_CHECKPOINT = "attacker-choice.safetensors"
# 0.11.2 `aeb126b95` made `upload_image` return the file descriptor the route appends verbatim.
UPLOADED_IMAGE = {
    "id": "f1",
    "url": "/file/f1",
    "name": "generated-image.png",
    "content_type": "image/png",
}


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    async def json(self, content_type=None):
        return self._payload

    def raise_for_status(self):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class FakeAutomatic1111:
    """Stands in for the shared Automatic1111 server. Records every call so the
    test can assert on the upstream traffic rather than on the route's return."""

    def __init__(self):
        self.checkpoint = CONFIGURED_CHECKPOINT
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, None))
        return FakeResponse({"sd_model_checkpoint": self.checkpoint})

    def post(self, url, json=None, **kwargs):
        self.calls.append(("POST", url, json))
        if url.endswith("/sdapi/v1/options"):
            self.checkpoint = json["sd_model_checkpoint"]
            return FakeResponse({})
        return FakeResponse({"images": ["ZmFrZQ=="], "info": "{}"})

    @property
    def checkpoint_switches(self):
        return [
            body
            for method, url, body in self.calls
            if method == "POST" and url.endswith("/sdapi/v1/options")
        ]

    @property
    def txt2img_payload(self):
        bodies = [
            body
            for method, url, body in self.calls
            if method == "POST" and url.endswith("/sdapi/v1/txt2img")
        ]
        assert len(bodies) == 1, f"expected exactly one txt2img call, got {len(bodies)}"
        return bodies[0]


def _image_config(engine="automatic1111", size="512x512", steps=20):
    return SimpleNamespace(
        IMAGE_GENERATION_ENGINE=engine,
        IMAGE_GENERATION_MODEL=CONFIGURED_CHECKPOINT,
        IMAGE_SIZE=size,
        IMAGE_STEPS=steps,
        AUTOMATIC1111_BASE_URL=BASE_URL,
        AUTOMATIC1111_API_AUTH=None,
        AUTOMATIC1111_PARAMS=None,
    )


def _user(role):
    return SimpleNamespace(id=f"{role}-1", role=role, email=f"{role}@example.com", name=role)


async def _generate(images_module, form_data, user, engine="automatic1111"):
    """Run the real `image_generations` against a fake Automatic1111 and a
    stubbed config store. Returns the fake server so callers can inspect it."""
    server = FakeAutomatic1111()
    upsert = AsyncMock()
    import open_webui.models.config as config_module

    with (
        patch.object(
            images_module, "get_image_config", AsyncMock(return_value=_image_config(engine))
        ),
        patch.object(images_module, "get_session", AsyncMock(return_value=server)),
        patch.object(
            images_module, "get_image_data", AsyncMock(return_value=(b"fake", "image/png"))
        ),
        patch.object(
            images_module,
            "upload_image",
            AsyncMock(return_value=(SimpleNamespace(id="f1"), UPLOADED_IMAGE)),
        ),
        patch.object(config_module.Config, "upsert", upsert),
    ):
        result = await images_module.image_generations(SimpleNamespace(), form_data, user=user)
    return server, upsert, result


@pytest.mark.asyncio
async def test_non_admin_model_override_does_not_switch_the_checkpoint(owui_module):
    images = owui_module("open_webui.routers.images")
    form = images.CreateImageForm(prompt="a cat", model=ATTACKER_CHECKPOINT)

    server, upsert, _ = await _generate(images, form, _user("user"))

    assert server.checkpoint_switches == [], (
        "a non-admin generation request carrying a model repointed the shared "
        f"Automatic1111 checkpoint for the whole instance (#27244): {server.checkpoint_switches}"
    )
    assert server.checkpoint == CONFIGURED_CHECKPOINT
    assert upsert.await_count == 0, (
        "a non-admin generation request persisted a new instance-wide image "
        f"model to the global config (#27244): {upsert.await_args_list}"
    )


@pytest.mark.asyncio
async def test_admin_model_override_still_switches_the_checkpoint(owui_module):
    images = owui_module("open_webui.routers.images")
    form = images.CreateImageForm(prompt="a cat", model="admin-choice.safetensors")

    server, upsert, _ = await _generate(images, form, _user("admin"))

    assert server.checkpoint_switches == [{"sd_model_checkpoint": "admin-choice.safetensors"}], (
        "an admin lost per-request model switching on Automatic1111; the fix for "
        "#27244 was only meant to gate the switch, not remove it"
    )
    assert upsert.await_args_list[0].args[0] == {
        "image_generation.model": "admin-choice.safetensors"
    }


# Broad: no request payload from a non-admin may mutate any instance-wide image
# setting, and the settings surface itself stays admin-only.


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["automatic1111", ""])
async def test_non_admin_request_mutates_no_global_image_setting(owui_module, engine):
    """Every field a non-admin can put on the payload, at once."""
    images = owui_module("open_webui.routers.images")
    form = images.CreateImageForm(
        prompt="a cat",
        model=ATTACKER_CHECKPOINT,
        size="1024x1024",
        n=2,
        steps=7,
        negative_prompt="blurry",
    )

    server, upsert, _ = await _generate(images, form, _user("user"), engine=engine)

    assert upsert.await_count == 0, (
        "a non-admin image request wrote instance-wide configuration; every key "
        f"in IMAGE_CONFIG_KEYS is admin-managed only (#27244): {upsert.await_args_list}"
    )
    assert server.checkpoint_switches == []


def test_image_settings_routes_require_an_admin(owui_module):
    """The routes that read or write the instance-wide image settings must keep
    their admin dependency, so the config surface has one gate."""
    images = owui_module("open_webui.routers.images")
    admin_only_paths = {"/config", "/config/update", "/config/url/verify"}

    assert admin_only_paths <= {route.path for route in images.router.routes}
    for route in images.router.routes:
        if route.path not in admin_only_paths:
            continue
        dependencies = [
            parameter.default.dependency
            for parameter in inspect.signature(route.endpoint).parameters.values()
            if hasattr(parameter.default, "dependency")
        ]
        assert images.get_admin_user in dependencies, (
            f"{route.path} exposes the instance-wide image settings without an "
            "admin dependency, so any verified user could read or change them"
        )


# Nearby: the ordinary non-admin path is untouched by the gate.


@pytest.mark.asyncio
async def test_non_admin_without_override_generates_on_configured_checkpoint(owui_module):
    images = owui_module("open_webui.routers.images")
    form = images.CreateImageForm(prompt="a cat")

    server, upsert, result = await _generate(images, form, _user("user"))

    assert result == [UPLOADED_IMAGE]
    assert server.checkpoint == CONFIGURED_CHECKPOINT
    assert upsert.await_count == 0


@pytest.mark.asyncio
async def test_non_admin_per_request_parameters_are_still_honoured(owui_module):
    """Size, steps, batch size, prompt and negative prompt are genuinely
    per-request, so the gate must not swallow them."""
    images = owui_module("open_webui.routers.images")
    form = images.CreateImageForm(
        prompt="a cat",
        model=ATTACKER_CHECKPOINT,
        size="768x1024",
        n=3,
        steps=7,
        negative_prompt="blurry",
    )

    server, _, _ = await _generate(images, form, _user("user"))

    assert server.txt2img_payload == {
        "prompt": "a cat",
        "batch_size": 3,
        "width": 768,
        "height": 1024,
        "steps": 7,
        "negative_prompt": "blurry",
    }


@pytest.mark.asyncio
async def test_non_admin_falls_back_to_the_configured_size_and_steps(owui_module):
    images = owui_module("open_webui.routers.images")
    form = images.CreateImageForm(prompt="a cat")

    server, _, _ = await _generate(images, form, _user("user"))

    payload = server.txt2img_payload
    assert (payload["width"], payload["height"], payload["steps"]) == (512, 512, 20)
