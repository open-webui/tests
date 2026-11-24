"""A model's new image only showed after a page reload, #34703.

Every view loads a model's image from one fixed address per model, so after saving a new image, or
resetting it to the default logo, the page kept the copy it had already loaded: the router model's
row in Admin Settings > Models and, for a workspace model, the Workspace > Models list and the
model selector showed the old image until a reload.

Discriminates: fails on dev d3e87f8b1 (the rows keep the red image after the save), passes on the
fix bc501425f (branch fix/model-image-refresh, PR for #34703).
"""

from __future__ import annotations

import base64
import re
import struct
import uuid
import zlib

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.router_models import (
    EVERYONE_READS,
    ROUTER_MODELS_CONFIG,
    default_routes,
    save_router_model,
)
from harness.upstream import MOCK_MODEL_ID
from utils.model_selector import model_options

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

RED = (255, 0, 0)
BLUE = (0, 0, 255)

# the colour of the image's middle pixel, or null until it has loaded
MIDDLE_PIXEL = """(image) => {
    if (!image.complete || !image.naturalWidth) return null;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 1;
    const context = canvas.getContext('2d');
    context.drawImage(image, image.naturalWidth / 2, image.naturalHeight / 2, 1, 1, 0, 0, 1, 1);
    return Array.from(context.getImageData(0, 0, 1, 1).data.slice(0, 3));
}"""


def _png(colour: tuple[int, int, int], size: int = 8) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    rows = b"".join(b"\x00" + bytes(colour) * size for _ in range(size))
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def _data_uri(colour: tuple[int, int, int]) -> str:
    return "data:image/png;base64," + base64.b64encode(_png(colour)).decode()


def _is_near(pixel: list[int] | None, colour: tuple[int, int, int]) -> bool:
    return pixel is not None and all(
        abs(shown - wanted) < 40 for shown, wanted in zip(pixel, colour)
    )


def _expect_image(image: Locator, colour: tuple[int, int, int], shown: bool = True) -> None:
    """Wait until the image shows `colour`, or with shown=False any other colour."""
    expect(image).to_be_visible()
    pixel = None
    for _ in range(50):
        pixel = image.evaluate(MIDDLE_PIXEL)
        if pixel is not None and _is_near(pixel, colour) == shown:
            return
        image.page.wait_for_timeout(200)
    wanted = f"{'' if shown else 'anything but '}{colour}"
    raise AssertionError(f"the model's image shows {pixel}, expected {wanted}")


@pytest.fixture
def router_model(admin, preserve) -> tuple[str, str]:
    """A public router model with a red image."""
    preserve(ROUTER_MODELS_CONFIG)
    router_id = f"router-{uuid.uuid4().hex[:8]}"
    name = f"Router {router_id[-8:]}"
    with admin.client() as client:
        save_router_model(
            client,
            router_id,
            name,
            routes=default_routes(MOCK_MODEL_ID),
            access_grants=[EVERYONE_READS],
            meta={"profile_image_url": _data_uri(RED)},
        )
    return router_id, name


@pytest.fixture
def red_model(admin):
    """A public workspace model with a red image, deleted afterwards."""
    suffix = uuid.uuid4().hex[:8]
    form = {
        "id": f"painted-{suffix}",
        "name": f"Painted {suffix}",
        "base_model_id": MOCK_MODEL_ID,
        "meta": {"profile_image_url": _data_uri(RED)},
        "params": {},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        yield form
        client.post("/api/v1/models/model/delete", json={"id": form["id"]})


def _router_rows(page: Page, router_id: str, name: str) -> list[Locator]:
    """The router model's row in the model list and in the Router Models panel, opened if closed."""
    settings = page.get_by_role("dialog")
    listed = settings.locator("#model-list > div").filter(has_text=name)
    expect(listed).to_have_count(1)
    panel_toggle = settings.get_by_role("button", name="Router Models", expanded=False)
    if panel_toggle.count():
        panel_toggle.click()
    return [listed, settings.get_by_role("button").filter(has_text=router_id)]


def test_a_router_model_reset_to_the_default_image_shows_it_in_its_rows_at_once(
    page_for, make_user, router_model
):
    router_id, name = router_model
    page: Page = page_for(make_user(role="admin"), path="/admin/settings/models")
    for row in _router_rows(page, router_id, name):
        _expect_image(row.locator("img").first, RED)

    _router_rows(page, router_id, name)[1].click()
    editor = page.get_by_role("dialog").filter(
        has=page.get_by_role("button", name="Upload profile image")
    )
    editor.get_by_role("button", name="Upload profile image").hover()
    editor.get_by_role("button", name="Reset", exact=True).click()
    with page.expect_response(lambda response: "/models/model/access/update" in response.url):
        editor.get_by_role("button", name="Save", exact=True).click()
    expect(editor).to_be_hidden()

    for row in _router_rows(page, router_id, name):
        _expect_image(row.locator("img").first, RED, shown=False)


def test_a_new_workspace_model_image_shows_in_the_list_and_the_selector_at_once(
    page_for, make_user, red_model
):
    page: Page = page_for(make_user(role="admin"), path="/workspace/models")
    listed = page.locator(f"#model-item-{red_model['id']}")
    _expect_image(listed.locator("img").first, RED)

    listed.locator("img").first.click()
    editor = page.get_by_role("main")
    expect(editor.get_by_placeholder("Model Name")).to_have_value(red_model["name"])
    with page.expect_file_chooser() as chooser:
        editor.get_by_role("button", name="Upload profile image").click()
    chooser.value.set_files(
        files=[{"name": "blue.png", "mimeType": "image/png", "buffer": _png(BLUE)}]
    )
    _expect_image(editor.get_by_role("img", name="model profile").first, BLUE)
    editor.get_by_role("button", name="Save & Update").click()
    expect(page).to_have_url(re.compile(r"/workspace/models/?$"))

    _expect_image(listed.locator("img").first, BLUE)

    page.get_by_role("navigation", name="Chat history").get_by_role("link", name="New Chat").click()
    _expect_image(model_options(page, red_model["name"]).locator("img").first, BLUE)
