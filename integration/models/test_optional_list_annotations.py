"""Optional list fields refused `null` and accepted `[null]`.

Fixed in open-webui 0.11.3 by `9962d122c` (`ModelsConfigForm.MODEL_ORDER_LIST`,
`WebConfig.WEB_SEARCH_DOMAIN_FILTER_LIST`), `b6d505522` (`ModelForm.access_grants`), `e96b6464b`
(`PromptForm.access_grants`, `ToolForm.access_grants`) and `873fb741c` (`PromptForm.tags`). Each
field was written `list[X | None] = None`, a list whose elements may be null, when it meant
`list[X] | None = None`. So a form sending `null` got a 422 and one sending `[null]` got through
to code that expects real elements.

Twin of unit/models/test_optional_list_annotations.py, which keeps the module-wide annotation sweep.

Discriminates: passes on upstream dev `bbfa876af`; with each annotation back on `list[X | None]`
every `test_null_is_accepted` and `test_a_null_element_is_refused` case and both config tests
fail, while the real-list and still-required tests pass.
"""

from __future__ import annotations

import uuid
from typing import Callable, Iterator

import httpx
import pytest

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
MODELS_CONFIG = ("/api/v1/configs/models", "/api/v1/configs/models")


def _suffix() -> str:
    return uuid.uuid4().hex[:8]


def _model_form() -> dict:
    return {"id": f"optional-list-{_suffix()}", "name": "M", "meta": {}, "params": {}}


def _prompt_form() -> dict:
    return {"command": f"optional-list-{_suffix()}", "name": "P", "content": "body"}


def _tool_form() -> dict:
    code = "class Tools:\n    pass\n"
    return {"id": f"optional_list_{_suffix()}", "name": "T", "content": code, "meta": {}}


def _delete_model(client: httpx.Client, created: dict) -> None:
    client.post("/api/v1/models/model/delete", json={"id": created["id"]})


def _delete_prompt(client: httpx.Client, created: dict) -> None:
    client.delete(f"/api/v1/prompts/id/{created['id']}/delete")


def _delete_tool(client: httpx.Client, created: dict) -> None:
    client.delete(f"/api/v1/tools/id/{created['id']}/delete")


# create endpoint, a fresh form, the optional list field, how to remove what was created
CREATE_FORMS = {
    "ModelForm.access_grants": (
        "/api/v1/models/create",
        _model_form,
        "access_grants",
        _delete_model,
    ),
    "PromptForm.access_grants": (
        "/api/v1/prompts/create",
        _prompt_form,
        "access_grants",
        _delete_prompt,
    ),
    "PromptForm.tags": ("/api/v1/prompts/create", _prompt_form, "tags", _delete_prompt),
    "ToolForm.access_grants": ("/api/v1/tools/create", _tool_form, "access_grants", _delete_tool),
}


def _real_list(field: str, owner_id: str) -> list:
    if field == "tags":
        return ["alpha"]
    return [{"principal_type": "user", "principal_id": owner_id, "permission": "read"}]


@pytest.fixture
def admin_client(admin) -> Iterator[httpx.Client]:
    with admin.client() as client:
        yield client


@pytest.fixture
def create(admin_client) -> Iterator[Callable[[str, object], httpx.Response]]:
    """`create(case, value)` posts a fresh form with the field set to `value`; created rows go."""
    cleanups: list[tuple[Callable, dict]] = []

    def post(case: str, value) -> httpx.Response:
        path, form, field, cleanup = CREATE_FORMS[case]
        response = admin_client.post(path, json={**form(), field: value})
        if response.status_code == 200:
            cleanups.append((cleanup, response.json()))
        return response

    yield post
    for cleanup, created in cleanups:
        cleanup(admin_client, created)


def _refused_element(response: httpx.Response) -> list:
    assert response.status_code == 422, f"HTTP {response.status_code} {response.text}"
    return [error["loc"][-2:] for error in response.json()["detail"]]


@pytest.mark.parametrize("case", CREATE_FORMS)
def test_null_is_accepted(create, case):
    response = create(case, None)
    assert response.status_code == 200, f"{case}: HTTP {response.status_code} {response.text}"


@pytest.mark.parametrize("case", CREATE_FORMS)
def test_a_null_element_is_refused(create, case):
    field = CREATE_FORMS[case][2]
    assert _refused_element(create(case, [None])) == [[field, 0]]


@pytest.mark.parametrize("case", CREATE_FORMS)
def test_a_real_list_is_still_accepted(create, admin, case):
    field = CREATE_FORMS[case][2]
    response = create(case, _real_list(field, admin.id))
    assert response.status_code == 200, f"{case}: HTTP {response.status_code} {response.text}"


def test_model_order_list_accepts_null_and_refuses_a_null_element(admin_client, preserve):
    preserve(MODELS_CONFIG)
    current = admin_client.get("/api/v1/configs/models").json()

    cleared = admin_client.post(
        "/api/v1/configs/models", json={**current, "MODEL_ORDER_LIST": None}
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["MODEL_ORDER_LIST"] is None

    with_null = {**current, "MODEL_ORDER_LIST": [None]}
    refused = admin_client.post("/api/v1/configs/models", json=with_null)
    assert _refused_element(refused) == [["MODEL_ORDER_LIST", 0]]


def test_model_order_list_is_still_required(admin_client, preserve):
    preserve(MODELS_CONFIG)
    current = admin_client.get("/api/v1/configs/models").json()
    current.pop("MODEL_ORDER_LIST")

    assert admin_client.post("/api/v1/configs/models", json=current).status_code == 422


def test_web_search_domain_filter_accepts_null_and_refuses_a_null_element(admin_client, preserve):
    preserve(RETRIEVAL_CONFIG)
    web = admin_client.get("/api/v1/retrieval/config").json()["web"]

    cleared = admin_client.post(
        "/api/v1/retrieval/config/update",
        json={"web": {**web, "WEB_SEARCH_DOMAIN_FILTER_LIST": None}},
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["web"]["WEB_SEARCH_DOMAIN_FILTER_LIST"] is None

    refused = admin_client.post(
        "/api/v1/retrieval/config/update",
        json={"web": {**web, "WEB_SEARCH_DOMAIN_FILTER_LIST": [None]}},
    )
    assert _refused_element(refused) == [["WEB_SEARCH_DOMAIN_FILTER_LIST", 0]]
