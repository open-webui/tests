"""Journey: the Documents, Web Search, Audio and Images tabs of the admin settings take effect.

Each test changes a setting the way an admin does, saves it, reloads the tab to see it stuck and
then uses it from a chat. Documents: a custom RAG template wraps the text of a file uploaded to
the chat. Web Search: an external engine on a local stand-in answers the model's search. Audio:
an OpenAI-compatible speech engine on a local stand-in reads a reply aloud. Images: the Gemini
engine on a local stand-in draws the picture the model asks for. Every test works as a fresh
admin and puts the settings back afterwards.

Discriminates: passes on dev ac00d40e3; in a backend copy, with `/api/v1/retrieval/config/update`
ignoring `RAG_TEMPLATE` the documents test fails (the template never sticks), with it ignoring
`WEB_SEARCH_ENGINE` the web search test fails (the engine is unset after the reload), with
`/api/v1/audio/config/update` dropping the TTS voice the audio test fails (the voice is back to
`alloy`), and with `/api/v1/images/config/update` dropping the Gemini API key the images test
fails (the engine refuses the request and no image shows).
"""

from __future__ import annotations

import json
import time
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.audio_engine import (
    AUDIO_NAMESPACE,
    CONFIG_IMPORT,
    SPEECH_MODEL,
    VOICE,
    serve_audio_engine,
)
from harness.image_engines import GEMINI_API_KEY, IMAGEN_MODEL, IMAGES_CONFIG, serve_gemini
from harness.listener import json_answer
from harness.web_retrieval import RETRIEVAL_CONFIG
from utils.chat_ui import chat_input, conversation, expect_reply, last_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def admin_page(make_user, page_for) -> Page:
    """A fresh admin's page, so nothing here touches the shared admin's own settings."""
    return page_for(make_user(role="admin"))


def _open_tab(page: Page, tab: str) -> Locator:
    """The admin settings on `tab`, freshly loaded from the server."""
    page.goto(f"/admin/settings/{tab}")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("tab", selected=True)).to_be_visible()
    return settings


def _save(page: Page, settings: Locator) -> None:
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def _turn_on(page: Page, integration: str) -> None:
    page.get_by_label("Integrations").click()
    page.get_by_role("menu").get_by_role("button", name=integration).click()
    page.keyboard.press("Escape")


def test_a_saved_rag_template_wraps_an_uploaded_file(admin_page, preserve, upstream):
    preserve(RETRIEVAL_CONFIG)
    marker = f"Answer from these notes only, batch {uuid.uuid4().hex[:6]}."
    settings = _open_tab(admin_page, "documents")
    template = settings.get_by_role(
        "textbox", name="Leave empty to use the default prompt, or enter a custom prompt"
    )
    template.fill(f"{marker}\n<context>{{{{CONTEXT}}}}</context>")
    _save(admin_page, settings)

    settings = _open_tab(admin_page, "documents")
    expect(template).to_have_value(f"{marker}\n<context>{{{{CONTEXT}}}}</context>")

    question = "which shelf holds the atlas?"
    upstream.queue(reply.text("The top shelf.", match=reply.answering(question)))
    admin_page.goto("/")
    expect(chat_input(admin_page)).to_be_visible()
    admin_page.get_by_role("main").get_by_label("More").click()
    with admin_page.expect_file_chooser() as chooser:
        admin_page.get_by_role("menu").get_by_role("button", name="Upload Files").click()
    chooser.value.set_files(
        {
            "name": "shelves.txt",
            "mimeType": "text/plain",
            "buffer": b"The atlas is on the top shelf.",
        }
    )
    expect(admin_page.get_by_role("button", name="shelves.txt")).to_be_visible()
    send(admin_page, question)
    expect_reply(admin_page, "The top shelf.")

    sent = json.dumps(next(filter(reply.answering(question), upstream.chat_requests())))
    assert marker in sent, "the saved RAG template was not used"
    assert "The atlas is on the top shelf." in sent


def test_a_saved_search_engine_answers_the_models_web_search(
    admin_page, preserve, listener, upstream
):
    preserve(RETRIEVAL_CONFIG)
    listener.route(
        "POST",
        "/search",
        json_answer(
            [
                {
                    "link": "https://birds.example/kestrels",
                    "title": "Kestrel census",
                    "snippet": "Kestrels number 412 in the county.",
                }
            ]
        ),
    )
    settings = _open_tab(admin_page, "web")
    settings.get_by_role("switch", name="Web Search", exact=True).click()
    settings.get_by_role("combobox", name="Select a engine").first.select_option("external")
    settings.get_by_role("textbox", name="Enter External Web Search URL").fill(
        f"{listener.base_url}/search"
    )
    settings.get_by_role("textbox", name="Enter External Web Search API Key").fill("search-key")
    settings.get_by_role("switch", name="Bypass Web Loader").click()
    _save(admin_page, settings)

    settings = _open_tab(admin_page, "web")
    expect(settings.get_by_role("switch", name="Web Search", exact=True)).to_be_checked()
    expect(settings.get_by_role("combobox", name="Select a engine").first).to_have_value("external")

    question = "how many kestrels are there?"
    upstream.queue(
        reply.tool_call("search_web", {"query": "kestrel count"}, match=reply.answering(question)),
        reply.text("There are 412 kestrels.", match=reply.answering(question)),
    )
    admin_page.goto("/")
    expect(chat_input(admin_page)).to_be_visible()
    _turn_on(admin_page, "Web Search")
    send(admin_page, question)
    expect_reply(admin_page, "There are 412 kestrels.")

    searches = listener.requests_to("/search")
    assert [search.json()["query"] for search in searches] == ["kestrel count"]
    assert searches[0].headers["Authorization"] == "Bearer search-key"
    tool_results = [
        entry["content"]
        for entry in upstream.chat_requests()[-1]["messages"]
        if entry["role"] == "tool"
    ]
    assert any("Kestrels number 412" in result for result in tool_results), tool_results


@pytest.fixture
def audio_restored(admin):
    """Puts the audio settings back through the config import, as `harness.audio_engine` does."""
    with admin.client() as client:
        snapshot = client.get(AUDIO_NAMESPACE)
        snapshot.raise_for_status()
    yield
    with admin.client() as client:
        restored = client.post(CONFIG_IMPORT, json={"config": snapshot.json()})
    assert restored.status_code == 200, f"restoring the audio settings failed: {restored.text}"


def _first_speech_request(engine, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if requests := engine.speech_requests():
            return requests[0]
        time.sleep(0.2)
    raise AssertionError("the speech engine was never asked to read the reply")


def test_a_saved_speech_engine_reads_a_reply_aloud(admin_page, audio_restored, listener, upstream):
    engine = serve_audio_engine(listener)
    settings = _open_tab(admin_page, "audio")
    # local Whisper would load a model on save, which an offline instance cannot
    settings.get_by_role("combobox", name="Select an engine").select_option(label="OpenAI")
    settings.get_by_role("combobox", name="Select a mode", exact=True).select_option(label="OpenAI")
    for base_url_box in settings.get_by_role("textbox", name="API Base URL").all():
        base_url_box.fill(engine.base_url)
    for key_box in settings.get_by_role("textbox", name="API Key").all():
        key_box.fill("sk-audio")
    settings.get_by_role("combobox", name="Select a voice").fill(VOICE)
    settings.get_by_role("combobox", name="Select a model").last.fill(SPEECH_MODEL)
    _save(admin_page, settings)

    settings = _open_tab(admin_page, "audio")
    expect(settings.get_by_role("combobox", name="Select a mode", exact=True)).to_have_value(
        "openai"
    )
    expect(settings.get_by_role("combobox", name="Select a voice")).to_have_value(VOICE)

    spoken = f"The ferry leaves at noon, pier {uuid.uuid4().hex[:4]}."
    upstream.queue(reply.text(spoken, match=reply.answering("when does the ferry leave?")))
    admin_page.goto("/")
    send(admin_page, "when does the ferry leave?")
    expect_reply(admin_page, spoken)
    last_reply(admin_page).hover()
    conversation(admin_page).get_by_role("button", name="Read Aloud").last.click()

    request = _first_speech_request(engine)
    assert (request["voice"], request["model"]) == (VOICE, SPEECH_MODEL), request
    assert request["input"] in spoken


def test_a_saved_image_engine_draws_the_models_picture(admin_page, preserve, listener, upstream):
    preserve(IMAGES_CONFIG)
    serve_gemini(listener)
    settings = _open_tab(admin_page, "images")
    settings.get_by_role("switch", name="Image Generation").click()
    settings.get_by_role("combobox", name="Select Engine").first.select_option(label="Gemini")
    settings.get_by_role("combobox", name="Select a model").fill(IMAGEN_MODEL)
    # the first pair is image generation's, the second image editing's
    settings.get_by_role("textbox", name="API Base URL").first.fill(listener.base_url)
    settings.get_by_role("textbox", name="API Key").first.fill(GEMINI_API_KEY)
    _save(admin_page, settings)

    settings = _open_tab(admin_page, "images")
    expect(settings.get_by_role("switch", name="Image Generation")).to_be_checked()
    expect(settings.get_by_role("combobox", name="Select Engine").first).to_have_value("gemini")

    question = "draw a red square"
    upstream.queue(
        reply.tool_call(
            "generate_image", {"prompt": "a red square"}, match=reply.answering(question)
        ),
        reply.text("Here is your red square.", match=reply.answering(question)),
    )
    admin_page.goto("/")
    expect(chat_input(admin_page)).to_be_visible()
    _turn_on(admin_page, "Image")
    send(admin_page, question)
    expect_reply(admin_page, "Here is your red square.")
    expect(last_reply(admin_page).get_by_role("img")).to_be_visible()
    assert listener.requests_to(f"/models/{IMAGEN_MODEL}:predict")
