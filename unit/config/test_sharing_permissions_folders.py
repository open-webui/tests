"""Regression: a default user permission without a schema field cannot be saved.

open-webui 0.11.0 fix `3cf3f8e18` (PR #27296, issue #27120): `sharing.folders` was in
`DEFAULT_USER_PERMISSIONS` but had no field on `SharingPermissions`, so the default and group
permission endpoints dropped it on every save. The integration and browser twins pin that flag;
this audit covers every section and every key added later, which neither can see because both
read the permissions back through the same schema.

Discriminates: passes on bbfa876af, fails with `folders` removed from `SharingPermissions`
(`sharing.folders` is reported missing).
"""

import pytest

pytestmark = pytest.mark.regression


def test_every_default_permission_has_a_schema_field(owui_module):
    defaults = owui_module("open_webui.config").DEFAULT_USER_PERMISSIONS
    sections = owui_module("open_webui.routers.users").UserPermissions.model_fields

    missing = []
    for section, flags in defaults.items():
        if section not in sections:
            missing.append(section)
            continue
        fields = sections[section].annotation.model_fields
        saved_names = {field.alias or name for name, field in fields.items()}
        missing += [f"{section}.{key}" for key in flags if key not in saved_names]

    assert missing == [], (
        f"default permissions {missing} have no field on UserPermissions, so an admin's save "
        "silently drops them (#27120)"
    )
