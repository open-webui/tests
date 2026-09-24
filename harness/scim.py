"""SCIM 2.0 provisioning against an instance booted with SCIM switched on.

`SCIM_ENV` switches SCIM on with `SCIM_TOKEN` as its bearer token; pass it to `instance_with`
(modules that pass the same env share one instance). `scim_client(instance)` is the identity
provider's client for `/api/v1/scim/v2`, and `provision(client)` creates an account the way a
directory sync does and returns its SCIM resource.
"""

from __future__ import annotations

import uuid

import httpx

from harness.instance import LaunchedInstance

SCIM_TOKEN = "scim-test-token-0123456789"
SCIM_ENV = {"ENABLE_SCIM": "true", "SCIM_TOKEN": SCIM_TOKEN, "SCIM_AUTH_PROVIDER": "oidc"}
PATCH_SCHEMA = "urn:ietf:params:scim:api:messages:2.0:PatchOp"
USER_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:User"


def scim_client(instance: LaunchedInstance) -> httpx.Client:
    return httpx.Client(
        base_url=f"{instance.base_url}/api/v1/scim/v2",
        headers={"Authorization": f"Bearer {SCIM_TOKEN}"},
        timeout=60.0,
    )


def provision(client: httpx.Client, display_name: str = "Directory User") -> dict:
    """A new account pushed by the directory, with an externalId of its own."""
    suffix = uuid.uuid4().hex[:8]
    email = f"directory-{suffix}@example.com"
    created = client.post(
        "/Users",
        json={
            "schemas": [USER_SCHEMA],
            "userName": email,
            "externalId": f"ext-{suffix}",
            "displayName": display_name,
            "emails": [{"value": email, "primary": True}],
            "active": True,
        },
    )
    assert created.status_code == 201, f"provisioning failed: {created.text}"
    return created.json()


def patch(client: httpx.Client, user_id: str, *operations: dict) -> httpx.Response:
    return client.patch(
        f"/Users/{user_id}", json={"schemas": [PATCH_SCHEMA], "Operations": list(operations)}
    )
