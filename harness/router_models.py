"""Router models, saved by the admin the way Admin Settings > Models > Router Models saves one.

The panel stores the router model in the router settings, then its access on a model row of
the same id, which makes the saving admin the row's owner.
"""

from __future__ import annotations

import httpx

ROUTER_MODELS_CONFIG = ("/api/v1/configs/router_models", "/api/v1/configs/router_models")
EVERYONE_READS = {"principal_type": "user", "principal_id": "*", "permission": "read"}


def default_routes(*model_ids: str) -> list[dict]:
    """Routes without a pattern or a condition: the models the router picks from by strategy."""
    return [{"model_id": model_id} for model_id in model_ids]


def save_router_model(
    client: httpx.Client,
    model_id: str,
    name: str,
    routes: list[dict],
    access_grants: list[dict],
    strategy: str = "random",
    meta: dict | None = None,
) -> None:
    """Add a router model next to the ones already saved and set its access, as the panel does."""
    current = client.get(ROUTER_MODELS_CONFIG[0])
    assert current.status_code == 200, current.text
    router_model = {
        "id": model_id,
        "name": name,
        "meta": {**(meta or {}), "routes": routes, "strategy": strategy},
    }
    saved = client.post(
        ROUTER_MODELS_CONFIG[1],
        json={"ROUTER_MODELS": [*current.json()["ROUTER_MODELS"], router_model]},
    )
    assert saved.status_code == 200, f"saving router model {name} failed: {saved.text}"
    access = client.post(
        "/api/v1/models/model/access/update",
        json={"id": model_id, "name": name, "access_grants": access_grants},
    )
    assert access.status_code == 200, f"setting the access of {name} failed: {access.text}"
