"""Journey: an admin manages the models of Ollama servers from Admin Settings > Models > Manage.

The Manage dialog of the models page lists every Ollama connection in a picker and works on the one
picked: it pulls a model with a progress bar (a pasted `ollama pull ...` command pulls the bare tag,
a tag already downloading is not pulled twice), updates every model the server has (and stops when
cancelled), deletes a model after a confirmation and creates one from a JSON recipe. Each action is
checked on both sides: what the admin sees (the progress, the toasts, the lists and the chat's model
selector) and what the Ollama stand-in was sent. Errors from the server reach the admin as a toast,
and an action on one server leaves the other alone.

Two tests stay red on dev until the dialog is fixed. A create whose stream reports download
progress (Ollama pulls a base it does not have first) shows no progress: the progress block reads
`createModelTag`, a variable the dialog lost in 419005a57, and throws "createModelTag is not
defined" (open-webui/open-webui#32001). A create Ollama refuses outright (a 400 such as "neither
'from' or 'files' was specified") shows nothing at all: the dialog only reads a response that
succeeded, clears the form and gives no error (open-webui/open-webui#32002).

Discriminates: passes on dev ebc6add67 apart from the two red tests; in frontend copies, with the
pasted command no longer trimmed the pull test fails, with the picker's choice not passed on the
second-server and unreachable-server tests fail, with stream errors ignored the pull error test
fails, with Update All stopping after the first model the update test fails, with its Cancel doing
nothing the cancel test fails, with the queue check dropped the already-downloading test fails,
with the confirmation skipped the delete test fails, with the recipe sent without its name the
create test fails and with a recipe that is not JSON sent as empty the JSON test fails; with the
progress block reading the typed name and a refused create shown as an error, both red tests pass.
"""

from __future__ import annotations

import json
import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.listener import json_answer, listening
from harness.ollama_provider import OLLAMA_CONFIG, ndjson, ndjson_stream, serve_ollama
from utils.chat_ui import chat_input
from utils.model_selector import model_options
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

BASE_MODEL = "llama3:latest"
PULLED_MODEL = "qwen3:0.6b"
DOWNLOADED = f"Model '{PULLED_MODEL}' has been successfully downloaded."
TAG_PLACEHOLDER = "Enter model tag (e.g. mistral:7b)"
NEW_MODEL_PLACEHOLDER = "Enter model tag (e.g. my-modelfile)"
NO_CREATE_PROGRESS = (
    'the create showed no progress: "createModelTag is not defined" (open-webui/open-webui#32001)'
)


def _connect(client, *urls: str) -> None:
    current = client.get(OLLAMA_CONFIG[0])
    current.raise_for_status()
    saved = client.post(
        OLLAMA_CONFIG[1],
        json={
            **current.json(),
            "ENABLE_OLLAMA_API": True,
            "OLLAMA_BASE_URLS": list(urls),
            "OLLAMA_API_CONFIGS": {},
        },
    )
    assert saved.status_code == 200, saved.text


@pytest.fixture
def operator(make_user):
    """An admin of its own, so the browser session and its download pool are the test's alone."""
    return make_user(role="admin")


@pytest.fixture
def ollama(admin, preserve, listener):
    preserve(OLLAMA_CONFIG)
    server = serve_ollama(listener, BASE_MODEL)
    with admin.client() as client:
        _connect(client, listener.base_url)
    yield server
    server.release_pulls()


@pytest.fixture
def two_servers(admin, preserve, listener):
    """Two Ollama stand-ins, each with a model of its own, the first one listed first."""
    preserve(OLLAMA_CONFIG)
    with listening() as second_listener:
        first = serve_ollama(listener, BASE_MODEL)
        second = serve_ollama(second_listener, "mistral:7b")
        with admin.client() as client:
            _connect(client, listener.base_url, second_listener.base_url)
        yield first, second


def manage_dialog(page: Page) -> Locator:
    page.goto("/admin/settings/models")
    return choose_manage(page)


def choose_manage(page: Page) -> Locator:
    """Opens the Manage dialog from the models page already showing, without a reload."""
    page.get_by_role("dialog").get_by_role("button", name="Actions").first.click()
    page.get_by_role("menu").get_by_role("button", name="Manage", exact=True).click()
    return page.get_by_role("dialog").filter(has_text="Manage Models")


def open_manage_models(page: Page) -> Locator:
    dialog = manage_dialog(page)
    expect(dialog.get_by_text("Pull a model from Ollama.com")).to_be_visible()
    return dialog


def delete_choices(dialog: Locator) -> Locator:
    return dialog.get_by_role("combobox", name="Select a model")


def pull(dialog: Locator, tag: str) -> None:
    dialog.get_by_placeholder(TAG_PLACEHOLDER).fill(tag)
    tooltip_button(dialog, "Pull Model").click()


def create(dialog: Locator, name: str, recipe: dict | str) -> None:
    dialog.get_by_placeholder(NEW_MODEL_PLACEHOLDER).fill(name)
    body = recipe if isinstance(recipe, str) else json.dumps(recipe)
    dialog.get_by_role("textbox").last.fill(body)
    dialog.get_by_role("button", name="Create Model").click()


def offered_in_chat(page: Page, name: str) -> Locator:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    return model_options(page, name)


def test_a_pulled_model_shows_its_progress_and_is_offered_in_chat(page_for, operator, ollama):
    ollama.hold_pulls()
    page = page_for(operator)
    dialog = open_manage_models(page)

    pull(dialog, f"ollama pull {PULLED_MODEL}")

    expect(dialog.get_by_text(PULLED_MODEL, exact=True)).to_be_visible()
    expect(dialog.get_by_text("50%")).to_be_visible()
    expect(dialog.get_by_text("sha256:0f")).to_be_visible()
    ollama.release_pulls()
    expect(page.get_by_text(DOWNLOADED)).to_be_visible()
    expect(dialog.get_by_text("50%")).to_have_count(0)
    assert [sent["name"] for sent in ollama.sent("/api/pull")] == [PULLED_MODEL]
    expect(offered_in_chat(page, PULLED_MODEL)).to_have_count(1)


def test_a_pull_ollama_refuses_shows_its_error(page_for, operator, ollama, listener):
    refusal = "pull model manifest: file does not exist"
    listener.route("POST", "/api/pull", ndjson({"status": "pulling manifest"}, {"error": refusal}))
    page = page_for(operator)
    dialog = open_manage_models(page)

    pull(dialog, "nonexistent:latest")

    expect(page.get_by_text(refusal)).to_be_visible()
    expect(page.get_by_text("Download canceled")).to_be_visible()
    assert page.get_by_text("has been successfully downloaded").count() == 0
    assert ollama.models == [BASE_MODEL]


def test_update_all_models_pulls_every_model_the_server_has(page_for, operator, ollama):
    ollama.models.append("mistral:7b")
    page = page_for(operator)
    dialog = open_manage_models(page)

    tooltip_button(dialog, "Update All Models").click()

    expect(page.get_by_text("All models are up to date")).to_be_visible()
    assert [sent["name"] for sent in ollama.sent("/api/pull")] == [BASE_MODEL, "mistral:7b"]


def test_cancelling_update_all_models_stops_after_the_model_in_flight(page_for, operator, ollama):
    ollama.models.append("mistral:7b")
    ollama.hold_pulls()
    page = page_for(operator)
    dialog = open_manage_models(page)

    tooltip_button(dialog, "Update All Models").click()
    expect(dialog.get_by_text(f'Updating "{BASE_MODEL}" (50%)')).to_be_visible()
    tooltip_button(dialog, "Cancel").click()

    expect(page.get_by_text("Model update cancelled")).to_be_visible()
    expect(dialog.get_by_text("Updating")).to_have_count(0)
    assert [sent["name"] for sent in ollama.sent("/api/pull")] == [BASE_MODEL]


def test_a_tag_already_downloading_is_not_pulled_twice(page_for, operator, ollama):
    ollama.hold_pulls()
    page = page_for(operator)
    dialog = open_manage_models(page)
    pull(dialog, PULLED_MODEL)
    expect(dialog.get_by_text("50%")).to_be_visible()

    page.keyboard.press("Escape")
    expect(dialog).to_be_hidden()
    dialog = choose_manage(page)
    pull(dialog, PULLED_MODEL)

    queued = f"Model '{PULLED_MODEL}' is already in queue for downloading."
    expect(page.get_by_text(queued)).to_be_visible()
    assert len(ollama.sent("/api/pull")) == 1


def test_deleting_a_model_asks_first_and_removes_it_everywhere(page_for, operator, ollama):
    ollama.models.append("mistral:7b")
    page = page_for(operator)
    dialog = open_manage_models(page)

    delete_choices(dialog).select_option(BASE_MODEL)
    dialog.get_by_role("button", name="Delete Model").click()
    confirm = page.get_by_role("dialog", name="Confirm your action")
    expect(confirm).to_be_visible()
    assert ollama.sent("/api/delete") == []
    confirm.get_by_role("button", name="Confirm").click()

    expect(page.get_by_text(f"Deleted {BASE_MODEL}")).to_be_visible()
    expect(delete_choices(dialog).locator("option", has_text=BASE_MODEL)).to_have_count(0)
    expect(delete_choices(dialog).locator("option", has_text="mistral:7b")).to_have_count(1)
    assert ollama.sent("/api/delete") == [{"model": BASE_MODEL}]
    expect(offered_in_chat(page, BASE_MODEL)).to_have_count(0)


def test_a_model_created_from_a_recipe_is_offered_in_chat(page_for, operator, ollama):
    recipe = {"from": BASE_MODEL, "system": "Answer in one line."}
    page = page_for(operator)
    dialog = open_manage_models(page)

    create(dialog, "brief", recipe)

    expect(page.get_by_text("success", exact=True)).to_be_visible()
    assert ollama.sent("/api/create") == [{"model": "brief", **recipe}]
    expect(dialog.get_by_placeholder(NEW_MODEL_PLACEHOLDER)).to_have_value("")
    expect(offered_in_chat(page, "brief:latest")).to_have_count(1)


def test_a_create_that_pulls_its_base_shows_the_download_progress(
    page_for, operator, ollama, listener
):
    # red on dev: the progress block throws "createModelTag is not defined"
    progress = {"status": "pulling 0f1e2d", "digest": "sha256:0f1e2d", "total": 100}
    lines = [{"status": "pulling manifest"}] + [{**progress, "completed": 40}] * 60
    listener.route("POST", "/api/create", lambda _request: ndjson_stream(lines))
    page = page_for(operator)
    dialog = open_manage_models(page)

    create(dialog, "tiny", {"from": "tinyllama:latest"})

    expect(dialog.get_by_text("40%"), NO_CREATE_PROGRESS).to_be_visible()
    expect(dialog.get_by_text("sha256:0f1e2d"), NO_CREATE_PROGRESS).to_be_visible()


def test_a_recipe_that_is_not_json_is_refused_before_ollama_is_asked(page_for, operator, ollama):
    page = page_for(operator)
    dialog = open_manage_models(page)

    create(dialog, "brief", "FROM llama3")

    expect(page.get_by_text("SyntaxError")).to_be_visible()
    assert ollama.sent("/api/create") == []


def test_a_create_ollama_refuses_shows_its_error(page_for, operator, ollama, listener):
    # red on dev: a refused create clears the form and shows no error
    refusal = "neither 'from' or 'files' was specified"
    listener.route("POST", "/api/create", json_answer({"error": refusal}, status=400))
    page = page_for(operator)
    dialog = open_manage_models(page)

    create(dialog, "orphan", {"system": "Answer in one line."})

    expect(
        page.get_by_text(refusal),
        "a refused create cleared the form and showed no error (open-webui/open-webui#32002)",
    ).to_be_visible()


def test_the_picked_server_is_the_one_managed(page_for, operator, two_servers):
    first, second = two_servers
    page = page_for(operator)
    dialog = open_manage_models(page)
    servers = dialog.get_by_role("combobox", name="Select an Ollama instance")
    expect(servers.locator("option")).to_have_text(
        [first.listener.base_url, second.listener.base_url]
    )
    expect(delete_choices(dialog).locator("option", has_text=BASE_MODEL)).to_have_count(1)

    servers.select_option(second.listener.base_url)
    expect(delete_choices(dialog).locator("option", has_text="mistral:7b")).to_have_count(1)
    expect(delete_choices(dialog).locator("option", has_text=BASE_MODEL)).to_have_count(0)
    pull(dialog, PULLED_MODEL)

    expect(page.get_by_text(DOWNLOADED)).to_be_visible()
    assert [sent["name"] for sent in second.sent("/api/pull")] == [PULLED_MODEL]
    assert first.sent("/api/pull") == []
    assert PULLED_MODEL in second.models and PULLED_MODEL not in first.models


def test_a_server_that_cannot_be_reached_is_reported_and_the_other_still_works(
    page_for, operator, admin, preserve, listener
):
    preserve(OLLAMA_CONFIG)
    working = serve_ollama(listener, BASE_MODEL)
    unreachable = "http://127.0.0.1:9"
    with admin.client() as client:
        _connect(client, unreachable, listener.base_url)
    page = page_for(operator)
    dialog = manage_dialog(page)

    expect(page.get_by_text(re.compile("Cannot connect to host 127.0.0.1:9"))).to_be_visible()
    expect(dialog.get_by_text("Failed to fetch models")).to_be_visible()
    expect(dialog.get_by_text("Pull a model from Ollama.com")).to_have_count(0)
    dialog.get_by_role("combobox", name="Select an Ollama instance").select_option(
        working.listener.base_url
    )
    expect(dialog.get_by_text("Pull a model from Ollama.com")).to_be_visible()
    expect(delete_choices(dialog).locator("option", has_text=BASE_MODEL)).to_have_count(1)
