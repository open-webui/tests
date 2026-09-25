"""Journey: exporting the settings and importing them back, the way the admin settings page does.

The export is every stored setting by key. Importing it after a setting has changed puts the
exported value back, both in a fresh export and on the route that serves the setting. An import
whose body is not `{"config": {key: value}}` is refused before anything is stored, so the
settings read the same after it as before.

Discriminates: in a backend copy, `import_config` not calling `Config.upsert` turns
`test_importing_an_export_puts_a_changed_setting_back` red, and `import_config` upserting the
raw request body before validating it turns that test and every case of
`test_a_malformed_import_is_refused_and_changes_nothing` red (the junk is stored).
"""

from __future__ import annotations

import httpx
import pytest

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

EXPORT, IMPORT, BANNERS = (
    "/api/v1/configs/export",
    "/api/v1/configs/import",
    "/api/v1/configs/banners",
)

CHANGED_BANNER = {
    "id": "export-import",
    "type": "info",
    "title": "Changed after the export",
    "content": "set between the export and the import",
    "dismissible": True,
    "timestamp": 1767225600,
}


def _export(client: httpx.Client) -> dict:
    exported = client.get(EXPORT)
    assert exported.status_code == 200, exported.text
    return exported.json()


@pytest.fixture
def admin_client(admin):
    """The admin's client; settings a test left changed are imported back as exported."""
    with admin.client() as client:
        snapshot = _export(client)
        yield client
        if _export(client) != snapshot:
            restored = client.post(IMPORT, json={"config": snapshot})
            assert restored.status_code == 200, restored.text


def test_importing_an_export_puts_a_changed_setting_back(admin_client):
    exported = _export(admin_client)
    changed = admin_client.post(BANNERS, json={"banners": [CHANGED_BANNER]})
    assert changed.status_code == 200, changed.text
    assert _export(admin_client) != exported

    imported = admin_client.post(IMPORT, json={"config": exported})

    assert imported.status_code == 200, imported.text
    assert _export(admin_client) == exported
    banners = admin_client.get(BANNERS)
    assert banners.status_code == 200, banners.text
    assert banners.json() == exported["ui.banners"]


def _unwrapped(exported: dict) -> dict:
    return {**exported, "ui.banners": [CHANGED_BANNER]}


def _pairs(exported: dict) -> dict:
    return {"config": [["ui.banners", [CHANGED_BANNER]]]}


def _text(exported: dict) -> dict:
    return {"config": "ui.banners=changed"}


@pytest.mark.parametrize(
    "malformed",
    [_unwrapped, _pairs, _text],
    ids=["export-without-config-wrapper", "config-as-pairs", "config-as-text"],
)
def test_a_malformed_import_is_refused_and_changes_nothing(admin_client, malformed):
    exported = _export(admin_client)

    refused = admin_client.post(IMPORT, json=malformed(exported))

    assert refused.status_code == 422, refused.text
    assert _export(admin_client) == exported
