"""Journey: managing the models of a llama.cpp or LM Studio connection through Open WebUI.

The admin's model management reads a runner's catalog, downloads, loads, unloads and deletes a
model, and the model selector shows which models are loaded. llama.cpp is reached at the root
of its `/v1` connection URL with the connection's prefix taken off the model name, and LM Studio
unloads by instance id through a download job it reports on. The model selector's unload goes
to llama.cpp's own unload as well.

Discriminates: in a backend copy, `get_model_management_root_url` keeping the `/v1` suffix,
`get_model_management_payload` sending LM Studio the model name to unload and the unified
`/api/models/unload` answering 400 for llama.cpp each turn tests red.
"""

from __future__ import annotations

import pytest

from harness.model_runners import (
    DOWNLOAD_JOB,
    connect_runner,
    serve_llama_cpp,
    serve_lm_studio,
)
from harness.second_provider import OPENAI_CONFIG

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

LLAMA_MODEL = "qwen3-0.6b"
STUDIO_MODEL = "google/gemma-3-1b"


@pytest.fixture
def admin_client(admin, preserve):
    preserve(OPENAI_CONFIG)
    with admin.client() as client:
        yield client


def _loaded_flags(client) -> dict[str, bool]:
    listed = client.get("/api/models")
    assert listed.status_code == 200, listed.text
    return {model["id"]: model.get("loaded") for model in listed.json()["data"]}


def test_a_llama_cpp_model_is_loaded_and_unloaded_by_its_bare_name(admin_client, listener):
    runner = serve_llama_cpp(listener, LLAMA_MODEL)
    index = connect_runner(admin_client, runner, prefix_id="local")
    prefixed = f"local.{LLAMA_MODEL}"

    catalog = admin_client.get(f"/openai/models/{index}/catalog")
    loaded = admin_client.post(f"/openai/models/{index}/load", json={"model": prefixed})
    flags_while_loaded = _loaded_flags(admin_client)
    unloaded = admin_client.post(f"/openai/models/{index}/unload", json={"model": prefixed})

    assert catalog.status_code == 200, catalog.text
    assert catalog.json()["data"][0]["status"] == {"value": "unloaded"}
    assert loaded.status_code == 200 and loaded.json() == {"success": True}, loaded.text
    assert unloaded.status_code == 200 and unloaded.json() == {"success": True}, unloaded.text
    assert runner.sent("/models/load") == [{"model": LLAMA_MODEL}]
    assert runner.sent("/models/unload") == [{"model": LLAMA_MODEL}]
    assert flags_while_loaded[prefixed] is True
    assert _loaded_flags(admin_client)[prefixed] is False


def test_a_llama_cpp_model_is_downloaded_and_deleted(admin_client, listener):
    runner = serve_llama_cpp(listener, LLAMA_MODEL)
    index = connect_runner(admin_client, runner)

    downloaded = admin_client.post(f"/openai/models/{index}/download", json={"model": "tiny"})
    deleted = admin_client.delete(f"/openai/models/{index}", params={"model": LLAMA_MODEL})
    events = admin_client.get(f"/openai/models/{index}/sse")

    assert downloaded.status_code == 200, downloaded.text
    assert deleted.status_code == 200, deleted.text
    assert runner.sent("/models") == [{"model": "tiny"}]
    assert listener.requests_to("/models")[-1].path.endswith(f"?model={LLAMA_MODEL}")
    assert runner.models == ["tiny"]
    assert events.status_code == 200 and '"tiny"' in events.text, events.text
    assert "tiny" in _loaded_flags(admin_client)


def test_the_model_selector_unloads_a_llama_cpp_model(admin_client, listener):
    runner = serve_llama_cpp(listener, LLAMA_MODEL)
    runner.loaded.append(LLAMA_MODEL)
    connect_runner(admin_client, runner)
    assert _loaded_flags(admin_client)[LLAMA_MODEL] is True

    unloaded = admin_client.post("/api/models/unload", json={"model": LLAMA_MODEL})

    assert unloaded.status_code == 200, unloaded.text
    assert runner.sent("/models/unload") == [{"model": LLAMA_MODEL}]
    assert runner.loaded == []


def test_an_lm_studio_model_is_downloaded_loaded_and_unloaded(admin_client, listener):
    runner = serve_lm_studio(listener)
    index = connect_runner(admin_client, runner)

    started = admin_client.post(f"/openai/models/{index}/download", json={"model": STUDIO_MODEL})
    status = admin_client.get(f"/openai/models/{index}/download/status/{DOWNLOAD_JOB}")
    catalog = admin_client.get(f"/openai/models/{index}/catalog")
    loaded = admin_client.post(f"/openai/models/{index}/load", json={"model": STUDIO_MODEL})
    unloaded = admin_client.post(
        f"/openai/models/{index}/unload", json={"model": STUDIO_MODEL, "instance_id": STUDIO_MODEL}
    )

    assert started.json() == {"job_id": DOWNLOAD_JOB, "status": "downloading"}, started.text
    assert status.json()["status"] == "completed", status.text
    assert [model["key"] for model in catalog.json()["models"]] == [STUDIO_MODEL]
    assert loaded.json()["instance_id"] == STUDIO_MODEL, loaded.text
    assert unloaded.status_code == 200, unloaded.text
    assert runner.sent("/api/v1/models/unload") == [{"instance_id": STUDIO_MODEL}]
    assert runner.loaded == []


def test_a_connection_without_model_management_is_refused(admin_client, upstream):
    refused = admin_client.post("/openai/models/0/load", json={"model": "mock-model"})

    assert refused.status_code == 400, refused.text
    assert "does not support model management" in refused.json()["detail"]
    assert not upstream.requests_to("/load")
