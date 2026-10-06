"""Journey: each image engine an admin can pick in Admin Settings > Images, set up there and used.

The admin sets an engine up in the Images tab against a local stand-in, then a user turns on
Image in the chat and the model calls `generate_image` or `edit_image`. Automatic1111: the
checkpoint picked from the engine's own list is switched to on save, and the size, steps, auth
string and extra parameters reach `txt2img`. ComfyUI: an uploaded workflow and its node mapping
put the prompt, checkpoint, size, steps and a seed into the queued graph, and the picture is
still in the chat after a reload. Gemini: the generateContent method picked in the tab is the
endpoint called. OpenAI: the size, API version and extra parameters reach the request. Editing:
a picture the user attaches is what the edit engine receives (OpenAI set up in the tab's Edit
Image section with its own model and size, Gemini and ComfyUI with its image node), and the
edited picture shows in the reply, also after a reload. The Image switch, its permission and the
OpenAI engine's own setup are in e2e/chat/test_image_generation.py; Image Prompt Generation only
runs with legacy function calling and is left out.

Discriminates: passes on the dev ebc6add67 build. In a backend copy whose `_apply_workflow_nodes`
skips the width and height nodes the ComfyUI test fails, whose Automatic1111 branch drops
`IMAGE_STEPS` the Automatic1111 test fails, whose Gemini branch always calls `:predict` the Gemini
test fails, whose `load_url_image` hands the engines the file id in place of the picture the
three edit tests fail, and whose OpenAI generation leaves out `IMAGES_OPENAI_API_PARAMS` the
OpenAI test fails.
"""

from __future__ import annotations

import base64
import json
import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.channel_chat import serve_openai_images
from harness.comfyui import (
    CHECKPOINTS,
    IMAGE_NODE,
    PNG,
    PROMPT_NODE,
    FakeComfyUI,
    comfyui_settings,
    serving,
    workflow,
)
from harness.image_engines import (
    CHECKPOINT,
    GEMINI_API_KEY,
    GEMINI_IMAGE_MODEL,
    IMAGES_CONFIG,
    PNG_BASE64,
    checkpoint_switches,
    gemini_calls,
    save_image_settings,
    serve_automatic1111,
    serve_gemini,
)
from harness.listener import json_answer
from utils.chat_ui import REPLY_TIMEOUT_MS, chat_input, expect_reply, replies, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

FILE_CONTENT_URL = re.compile(r"/api/v1/files/[^/]+/content$")
# a 2x2 blue PNG, the picture a user attaches for an edit
ATTACHED_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGNgYGD4z8DAwMDAAAAPAAH/JqQZAAAAAElFTkSuQmCC"
)
# a 1x1 green PNG, what the OpenAI stand-in answers for an edit
EDITED_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC"
)
# a text-to-image graph in ComfyUI's API format; the harness stand-in reports node 9's image
TEXT_TO_IMAGE = {
    "3": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 20, "model": ["4", 0]}},
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ""}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1, "height": 1, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
}


@pytest.fixture
def images_restored(preserve) -> None:
    preserve(IMAGES_CONFIG)


@pytest.fixture
def comfyui() -> FakeComfyUI:
    fake = FakeComfyUI()
    with serving(fake):
        yield fake


def image_settings(page: Page) -> Locator:
    """Admin Settings > Images, freshly loaded from the server."""
    page.goto("/admin/settings/images")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("switch", name="Image Generation")).to_be_visible()
    return settings


def switch_on(settings: Locator, name: str) -> None:
    switch = settings.get_by_role("switch", name=name, exact=True)
    if switch.get_attribute("aria-checked") != "true":
        switch.click()
    expect(switch).to_be_checked()


def save(page: Page, settings: Locator) -> None:
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def pick_engine(settings: Locator, engine: str, editing: bool = False) -> None:
    """Pick `engine` in Create Image's engine select, or in Edit Image's."""
    selects = settings.get_by_role("combobox", name="Select Engine")
    (selects.last if editing else selects.first).select_option(label=engine)


def open_image_chat(page: Page) -> None:
    """A new chat with Image turned on under Integrations."""
    page.goto("/")
    expect(chat_input(page)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    page.get_by_label("Integrations").click()
    menu = page.get_by_role("menu")
    menu.get_by_role("button", name="Image", exact=True).click()
    page.keyboard.press("Escape")


def draw(page: Page, upstream, prompt: str) -> str:
    """Ask for `prompt` to be drawn; returns the reply text."""
    request = f"please draw {prompt}"
    answer = f"Here is {prompt}."
    upstream.queue(
        reply.tool_call("generate_image", {"prompt": prompt}, match=reply.answering(request)),
        reply.text(answer, match=reply.answering(request)),
    )
    send(page, request)
    expect_reply(page, answer)
    return answer


def attach_picture(page: Page) -> str:
    """Attach `ATTACHED_PNG` through Upload Files; returns the stored file's id."""
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="More", exact=True).last.click()
    with page.expect_response(_is_file_upload) as uploaded:
        with page.expect_file_chooser() as chooser:
            page.get_by_role("menu").get_by_role("button", name="Upload Files").click()
        chooser.value.set_files(
            {"name": "harbour.png", "mimeType": "image/png", "buffer": ATTACHED_PNG}
        )
    expect(page.get_by_role("button", name="Show image preview")).to_be_visible()
    return uploaded.value.json()["id"]


def _is_file_upload(response) -> bool:
    return response.request.method == "POST" and re.search(r"/api/v1/files/?(\?|$)", response.url)


def edit_attached(page: Page, upstream, file_id: str, change: str) -> str:
    """Ask for the attached picture to be changed; returns the reply text."""
    answer = f"Edited: {change}."
    upstream.queue(
        reply.tool_call(
            "edit_image",
            {"prompt": change, "image_urls": [file_id]},
            match=reply.answering(change),
        ),
        reply.text(answer, match=reply.answering(change)),
    )
    send(page, change)
    expect_reply(page, answer)
    return answer


def offered_attachment(upstream, change: str) -> str:
    """The user message the model got with the change, attachments included."""
    asked = [body for body in upstream.chat_requests() if reply.answering(change)(body)]
    users = [entry for entry in asked[0]["messages"] if entry["role"] == "user"]
    return json.dumps(users[-1]["content"])


def reply_image(page: Page) -> Locator:
    return replies(page).last.get_by_role("img")


def expect_stored_picture(page: Page) -> str:
    expect(reply_image(page)).to_be_visible()
    source = reply_image(page).get_attribute("src")
    assert FILE_CONTENT_URL.search(source), f"the picture is not a stored file: {source}"
    return source


def stored_bytes(actor, source: str) -> bytes:
    with actor.client() as client:
        fetched = client.get(source)
    assert fetched.status_code == 200, fetched.text
    return fetched.content


def form_field(body: bytes, name: str) -> str:
    """The value of the multipart form field `name` in `body`."""
    found = re.search(rb'name="' + name.encode() + rb'"\r\n(?:[^\r\n]+\r\n)*\r\n([^\r\n]*)', body)
    assert found, f"no {name!r} field in the form"
    return found.group(1).decode()


def test_automatic1111_set_up_in_the_tab_draws_on_its_checkpoint_size_and_steps(
    page_for, make_user, images_restored, listener, upstream
):
    serve_automatic1111(listener)
    listener.route(
        "GET",
        "/sdapi/v1/sd-models",
        json_answer([{"title": name, "model_name": name} for name in ("sketch", CHECKPOINT)]),
    )
    admin_page = page_for(make_user(role="admin"))
    settings = image_settings(admin_page)
    switch_on(settings, "Image Generation")
    pick_engine(settings, "Automatic1111")
    settings.get_by_placeholder("Enter URL (e.g. http://127.0.0.1:7860/)").fill(listener.base_url)
    settings.get_by_role("button", name="verify connection").click()
    expect(admin_page.get_by_text("Server connection verified")).to_be_visible()
    settings.get_by_placeholder("Enter api auth string (e.g. username:password)").fill(
        "painter:easel"
    )
    settings.get_by_placeholder("Enter additional parameters in JSON format").fill(
        '{"cfg_scale": 5}'
    )
    model_box = settings.get_by_role("combobox", name="Select a model").first
    model_box.fill("sketch")
    settings.get_by_placeholder("Enter Image Size (e.g. 512x512)").first.fill("640x384")
    settings.get_by_placeholder("Enter Number of Steps (e.g. 50)").fill("12")
    save(admin_page, settings)

    assert {"sd_model_checkpoint": "sketch"} in checkpoint_switches(listener)
    page = page_for(make_user())
    open_image_chat(page)
    draw(page, upstream, "a lighthouse at dusk")

    expect_stored_picture(page)
    [generation] = listener.requests_to("/sdapi/v1/txt2img")
    sent = generation.json()
    assert sent["prompt"] == "a lighthouse at dusk", sent
    assert (sent["width"], sent["height"], sent["steps"]) == (640, 384, 12), sent
    assert sent["cfg_scale"] == 5, sent
    expected_auth = "Basic " + base64.b64encode(b"painter:easel").decode()
    assert generation.headers.get("authorization") == expected_auth


def test_a_comfyui_workflow_and_its_node_mapping_from_the_tab_shape_the_queued_graph(
    page_for, make_user, images_restored, comfyui, upstream
):
    admin_page = page_for(make_user(role="admin"))
    settings = image_settings(admin_page)
    switch_on(settings, "Image Generation")
    pick_engine(settings, "ComfyUI")
    settings.get_by_placeholder("Enter URL (e.g. http://127.0.0.1:7860/)").first.fill(
        comfyui.base_url
    )
    settings.get_by_role("button", name="verify connection").first.click()
    expect(admin_page.get_by_text("Server connection verified")).to_be_visible()
    with admin_page.expect_file_chooser() as chooser:
        settings.get_by_role("button", name="Click here to upload a workflow.json file.").click()
    chooser.value.set_files(
        {
            "name": "workflow.json",
            "mimeType": "application/json",
            "buffer": json.dumps(TEXT_TO_IMAGE).encode(),
        }
    )
    mapping = {"prompt*": "6", "model": "4", "width": "5", "height": "5", "steps": "3", "seed": "3"}
    for node_type, node_ids in mapping.items():
        node_ids_box(settings, node_type).fill(node_ids)
    settings.get_by_role("combobox", name="Select a model").first.fill(CHECKPOINTS[1])
    settings.get_by_placeholder("Enter Image Size (e.g. 512x512)").first.fill("768x320")
    settings.get_by_placeholder("Enter Number of Steps (e.g. 50)").fill("9")
    save(admin_page, settings)

    person = make_user()
    page = page_for(person)
    open_image_chat(page)
    draw(page, upstream, "a harbour in fog")

    source = expect_stored_picture(page)
    [queued] = comfyui.queued
    graph = queued["prompt"]
    assert graph["6"]["inputs"]["text"] == "a harbour in fog", graph
    assert graph["4"]["inputs"]["ckpt_name"] == CHECKPOINTS[1], graph
    size = (graph["5"]["inputs"]["width"], graph["5"]["inputs"]["height"])
    assert size == (768, 320), graph
    assert graph["3"]["inputs"]["steps"] == 9, graph
    assert graph["3"]["inputs"]["seed"] != 0, "the seed node never got a seed"
    assert stored_bytes(person, source) == PNG

    page.reload()
    expect_reply(page, "Here is a harbour in fog.")
    expect(reply_image(page)).to_have_attribute("src", source)


def node_ids_box(settings: Locator, node_type: str) -> Locator:
    """The Node Ids box in the workflow node mapping row labelled `node_type`."""
    label = settings.get_by_text(node_type, exact=True).first
    return label.locator("xpath=../..").get_by_placeholder("Node Ids")


def test_the_gemini_endpoint_method_picked_in_the_tab_is_the_one_called(
    page_for, make_user, images_restored, listener, upstream
):
    serve_gemini(listener)
    admin_page = page_for(make_user(role="admin"))
    settings = image_settings(admin_page)
    switch_on(settings, "Image Generation")
    pick_engine(settings, "Gemini")
    settings.get_by_role("combobox", name="Select a model").first.fill(GEMINI_IMAGE_MODEL)
    settings.get_by_role("textbox", name="API Base URL").first.fill(listener.base_url)
    settings.get_by_role("textbox", name="API Key").first.fill(GEMINI_API_KEY)
    settings.get_by_role("combobox", name="Select Method").select_option("generateContent")
    save(admin_page, settings)

    page = page_for(make_user())
    open_image_chat(page)
    draw(page, upstream, "a kestrel on a post")

    expect_stored_picture(page)
    [call] = gemini_calls(listener, GEMINI_IMAGE_MODEL, "generateContent")
    assert call.json()["contents"][0]["parts"][0]["text"] == "a kestrel on a post"
    assert gemini_calls(listener, GEMINI_IMAGE_MODEL, "predict") == []


def test_the_openai_size_version_and_extra_parameters_reach_the_engine(
    page_for, make_user, admin, images_restored, listener, upstream
):
    with admin.client() as client:
        save_image_settings(client, **serve_openai_images(listener))
    admin_page = page_for(make_user(role="admin"))
    settings = image_settings(admin_page)
    settings.get_by_role("combobox", name="Select a model").first.fill("gpt-image-1")
    settings.get_by_placeholder("Enter Image Size (e.g. 512x512)").first.fill("1024x1536")
    settings.get_by_role("textbox", name="API Version").first.fill("2025-04-01")
    settings.get_by_placeholder("Enter additional parameters in JSON format").fill(
        '{"quality": "high"}'
    )
    save(admin_page, settings)

    page = page_for(make_user())
    open_image_chat(page)
    draw(page, upstream, "a tall red buoy")

    expect_stored_picture(page)
    [generation] = listener.requests_to("/images/generations")
    sent = generation.json()
    assert (sent["model"], sent["size"], sent["quality"]) == ("gpt-image-1", "1024x1536", "high")
    assert generation.path.endswith("?api-version=2025-04-01"), generation.path


def test_an_attached_picture_is_edited_by_the_openai_edit_engine_set_up_in_the_tab(
    page_for, make_user, admin, images_restored, listener, upstream
):
    listener.route(
        "POST", "/images/edits", json_answer({"data": [{"b64_json": EDITED_PNG_BASE64}]})
    )
    with admin.client() as client:
        save_image_settings(client, **serve_automatic1111(listener), ENABLE_IMAGE_EDIT=False)
    admin_page = page_for(make_user(role="admin"))
    settings = image_settings(admin_page)
    switch_on(settings, "Image Edit")
    pick_engine(settings, "Default (Open AI)", editing=True)
    settings.get_by_role("combobox", name="Select a model").last.fill("gpt-image-1")
    settings.get_by_placeholder("Enter Image Size (e.g. 512x512)").last.fill("256x256")
    settings.get_by_role("textbox", name="API Base URL").last.fill(listener.base_url)
    settings.get_by_role("textbox", name="API Key").last.fill("sk-edits")
    save(admin_page, settings)

    person = make_user()
    page = page_for(person)
    open_image_chat(page)
    file_id = attach_picture(page)
    change = f"make the harbour green {uuid.uuid4().hex[:6]}"
    answer = edit_attached(page, upstream, file_id, change)

    assert file_id in offered_attachment(upstream, change), "the model was not told of the picture"
    [edit] = listener.requests_to("/images/edits")
    assert ATTACHED_PNG in edit.body, "the edit engine never got the attached picture"
    assert edit.headers.get("Authorization") == "Bearer sk-edits"
    assert (form_field(edit.body, "model"), form_field(edit.body, "size")) == (
        "gpt-image-1",
        "256x256",
    )
    assert form_field(edit.body, "prompt") == change
    source = expect_stored_picture(page)
    assert stored_bytes(person, source) == base64.b64decode(EDITED_PNG_BASE64)

    page.reload()
    expect_reply(page, answer)
    expect(reply_image(page)).to_have_attribute("src", source)


def test_an_attached_picture_is_edited_through_gemini(
    page_for, make_user, admin, images_restored, listener, upstream
):
    with admin.client() as client:
        save_image_settings(client, **serve_gemini(listener))
    person = make_user()
    page = page_for(person)
    open_image_chat(page)
    file_id = attach_picture(page)
    edit_attached(page, upstream, file_id, f"add a gull {uuid.uuid4().hex[:6]}")

    [edit] = gemini_calls(listener, GEMINI_IMAGE_MODEL, "generateContent")
    parts = edit.json()["contents"][0]["parts"]
    sent_pictures = [part["inline_data"]["data"] for part in parts if "inline_data" in part]
    assert [base64.b64decode(picture) for picture in sent_pictures] == [ATTACHED_PNG], parts
    source = expect_stored_picture(page)
    assert stored_bytes(person, source) == base64.b64decode(PNG_BASE64)


def test_an_attached_picture_is_edited_through_the_comfyui_image_node(
    page_for, make_user, admin, images_restored, comfyui, upstream
):
    with admin.client() as client:
        save_image_settings(client, **comfyui_settings(comfyui.base_url, workflow()))
    person = make_user()
    page = page_for(person)
    open_image_chat(page)
    file_id = attach_picture(page)
    change = f"paint it at night {uuid.uuid4().hex[:6]}"
    edit_attached(page, upstream, file_id, change)

    [queued] = comfyui.queued
    graph = queued["prompt"]
    assert graph[IMAGE_NODE["node_ids"][0]]["inputs"]["image"] == "input.png", graph
    assert graph[PROMPT_NODE["node_ids"][0]]["inputs"]["text"] == change, graph
    assert comfyui.uploads == [ATTACHED_PNG], "ComfyUI never got the attached picture"
    source = expect_stored_picture(page)
    assert stored_bytes(person, source) == PNG
