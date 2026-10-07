"""Journey: the chat's model selector downloads Ollama models and shows and ejects loaded ones.

An admin who searches the selector for a tag Ollama does not have can download it from there: the
selector shows the progress, the model is offered as soon as the download finishes and an error
Ollama streams back is shown. A model in Ollama's running list (`/api/ps`) carries a green dot
whose tooltip says when Ollama will unload it. An admin hovering it gets an Eject button: Ollama
is asked to unload the model at once, a toast confirms it and the dot is gone once the list
refreshes. A model that is not loaded offers no Eject, a refused unload shows Ollama's error and
keeps the dot, and a regular user sees the dot without the button.

Discriminates: passes on dev ebc6add67; in a frontend copy, with the list not refreshed after a
download the download test fails, with stream errors ignored the download error test fails, with
the dot shown for every model the not-loaded test fails, with the Eject button offered to every
role the regular-user test fails, with the list not refreshed after an unload the eject test
fails (the dot stays) and with success announced after a refused unload the refused unload test
fails.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.listener import json_answer
from harness.ollama_provider import OLLAMA_CONFIG, connect_ollama, ndjson, serve_ollama
from utils.chat_ui import chat_input
from utils.model_selector import SELECTOR_BUTTON, model_options

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

LOADED_MODEL = "llama3:latest"
IDLE_MODEL = "mistral:7b"
SHARED_MODEL = "loaded-for-everyone:latest"
DOWNLOADED_MODEL = "qwen3:0.6b"


@pytest.fixture
def ollama(admin, preserve, listener):
    preserve(OLLAMA_CONFIG)
    server = serve_ollama(listener, LOADED_MODEL, IDLE_MODEL)
    server.loaded.append(LOADED_MODEL)
    with admin.client() as client:
        connect_ollama(client, listener)
    yield server
    server.release_pulls()


def searched(page: Page, name: str) -> Locator:
    expect(chat_input(page)).to_be_visible()
    option = model_options(page, name)
    expect(option).to_have_count(1)
    return option


def running_status(option: Locator) -> str | None:
    """The tooltip of the option's loaded dot, or None when the option has no dot."""
    return option.evaluate(
        "(option) => [...option.querySelectorAll('*')]"
        ".map((element) => element._tippy?.props.content)"
        ".find((content) => /^(Unloads|Loaded)/.test(content ?? '')) ?? null"
    )


def eject_button(option: Locator) -> Locator:
    return option.get_by_role("button", name="Eject model")


def download_from_selector(page: Page, tag: str) -> None:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name=SELECTOR_BUTTON).click()
    page.get_by_role("textbox", name="Search In Models").fill(tag)
    page.get_by_role("option", name=f'Download "{tag}"').click()


def test_a_model_downloaded_from_the_selector_is_offered_at_once(page_for, make_user, ollama):
    ollama.hold_pulls()
    page = page_for(make_user(role="admin"))

    download_from_selector(page, DOWNLOADED_MODEL)

    expect(page.get_by_text("50%")).to_be_visible()
    page.wait_for_timeout(1500)  # outlasts the server's one-second cache of the model list
    ollama.release_pulls()
    downloaded = f"Model '{DOWNLOADED_MODEL}' has been successfully downloaded."
    expect(page.get_by_text(downloaded)).to_be_visible()
    assert [sent["name"] for sent in ollama.sent("/api/pull")] == [DOWNLOADED_MODEL]
    expect(model_options(page, DOWNLOADED_MODEL)).to_have_count(1)


def test_a_download_ollama_refuses_shows_its_error(page_for, make_user, ollama, listener):
    refusal = "pull model manifest: file does not exist"
    listener.route("POST", "/api/pull", ndjson({"status": "pulling manifest"}, {"error": refusal}))
    page = page_for(make_user(role="admin"))

    download_from_selector(page, "nonexistent:latest")

    expect(page.get_by_text(refusal)).to_be_visible()
    expect(page.get_by_text("Download canceled")).to_be_visible()
    assert page.get_by_text("has been successfully downloaded").count() == 0


def test_a_loaded_model_shows_when_it_unloads_and_eject_frees_it(page_for, make_user, ollama):
    page = page_for(make_user(role="admin"))
    option = searched(page, LOADED_MODEL)
    assert re.fullmatch(r"Unloads in .+", running_status(option) or ""), running_status(option)

    option.hover()
    eject_button(option).click()

    expect(page.get_by_text("Model unloaded successfully")).to_be_visible()
    assert ollama.sent("/api/generate") == [{"model": LOADED_MODEL, "keep_alive": 0, "prompt": ""}]
    expect(eject_button(option)).to_have_count(0)
    assert running_status(option) is None


def test_a_model_that_is_not_loaded_has_no_dot_and_no_eject(page_for, make_user, ollama):
    page = page_for(make_user(role="admin"))

    option = searched(page, IDLE_MODEL)
    option.hover()

    assert running_status(option) is None
    expect(eject_button(option)).to_have_count(0)


def test_an_unload_ollama_refuses_shows_its_error_and_keeps_the_model_loaded(
    page_for, make_user, ollama, listener
):
    listener.route("POST", "/api/generate", json_answer({"error": "server busy"}, status=500))
    page = page_for(make_user(role="admin"))
    option = searched(page, LOADED_MODEL)

    option.hover()
    eject_button(option).click()

    expect(page.get_by_text(re.compile("Error unloading model"))).to_be_visible()
    page.wait_for_timeout(1000)  # a success toast would follow the error at once
    assert page.get_by_text("Model unloaded successfully").count() == 0
    assert ollama.loaded == [LOADED_MODEL]
    page.reload()
    assert running_status(searched(page, LOADED_MODEL)) is not None


def test_a_regular_user_sees_the_dot_but_cannot_eject(page_for, make_user, admin, ollama):
    # a model of its own, as the read grant outlives the test
    ollama.models.append(SHARED_MODEL)
    ollama.loaded.append(SHARED_MODEL)
    with admin.client() as client:
        client.get("/api/models").raise_for_status()  # registers the connection's models
        shared = client.post(
            "/api/v1/models/model/access/update",
            json={
                "id": SHARED_MODEL,
                "name": SHARED_MODEL,
                "access_grants": [
                    {"principal_type": "user", "principal_id": "*", "permission": "read"}
                ],
            },
        )
    assert shared.status_code == 200, shared.text
    page = page_for(make_user())
    option = searched(page, SHARED_MODEL)

    option.hover()

    assert running_status(option) is not None
    expect(eject_button(option)).to_have_count(0)
