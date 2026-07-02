"""Regression for open-webui 0.10.2 fix `54f31c630`.

Editing a workspace model wiped its `base_model_id` whenever the edit form
didn't explicitly carry that field. `base_model_id` is the actual upstream
model a workspace entry proxies to; clearing it silently detaches the entry
from its backing model (the model layer then treats a null `base_model_id` as
a *base* model, so it disappears from the workspace `/models` list, which
filters on `base_model_id != None`).

Mechanism. The clobber lives entirely in the router endpoint
`update_model_by_id` (backend/open_webui/routers/models.py). `ModelForm` gives
`base_model_id` a default of `None`, so an edit payload that omits the field
still produces a form whose `base_model_id` is `None`. The model-layer writer
`Models.update_model_by_id` then does `model.model_dump(...)` and writes EVERY
column unconditionally (`update(Model).values(**data)`), so that defaulted
`None` overwrites the stored value. The model layer is identical between the
buggy and fixed versions — only the router changed — so the preservation has
to be, and is, asserted at the router.

Fix (commit 54f31c630): before handing the form to the model layer, the
router restores the existing value when the caller didn't set it explicitly::

    if 'base_model_id' not in form_data.model_fields_set:
        form_data.base_model_id = model.base_model_id

Pydantic's `model_fields_set` is the load-bearing signal: it holds only the
fields the incoming payload actually supplied, so it distinguishes "omitted
(preserve)" from "explicitly sent" — including an explicit `null` (a real
detach, which stays in the set and is honored).

Why a source audit for the guard. Tripping the real clobber needs a running
app: a populated `app.state`, a DB, an authenticated write-capable session and
the FastAPI dependency graph, none of which this offline suite has. The guard
is a single scoped edit to one function and the fixed/buggy bodies are
textually distinct, so a body-scoped read discriminates cleanly. To keep that
audit from being a coincidental string match, the behavioral tests below prove
— against the real `ModelForm` — that the mechanism the guard relies on is
exactly as described: an omitted `base_model_id` is absent from
`model_fields_set` and defaults to `None`, so without the guard the model layer
would write that `None`.

Discriminates: passes on dev/0.10.2 (guard present), fails on 0.10.1 (no guard).
"""

from __future__ import annotations

import re
import sys

import pytest

pytestmark = pytest.mark.regression

FUNC = "update_model_by_id"


def _function_body(open_webui_backend) -> str:
    """Return the source of the router's `update_model_by_id` only.

    Sliced from its `async def` to the next top-level `@router`/`async def`/
    `def` so the audit is confined to the endpoint the fix touched and can't be
    fooled by an identical guard elsewhere in the module (the model-layer method
    shares the name).
    """
    src = (open_webui_backend / "open_webui" / "routers" / "models.py").read_text(encoding="utf-8")
    start = re.search(rf"^async def {re.escape(FUNC)}\b", src, re.MULTILINE)
    assert start is not None, f"{FUNC} not found in routers/models.py — source moved?"

    rest = src[start.end() :]
    nxt = re.search(r"^(?:@router\b|async def |def )", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def _load_model_form(open_webui_backend):
    """Import the real `ModelForm` from the checkout under test.

    Don't evict a cached `open_webui.models.models`: re-importing re-runs the
    SQLAlchemy `class Model(Base)` declaration against a metadata that may
    already hold the `model` table ("Table 'model' is already defined"). We
    only need the stable `ModelForm` pydantic class, so a plain import is both
    sufficient and re-import-safe.
    """
    if str(open_webui_backend) not in sys.path:
        sys.path.insert(0, str(open_webui_backend))
    try:
        from open_webui.models.models import ModelForm
    except Exception as e:  # pragma: no cover - env-dependent
        pytest.skip(f"Could not import open_webui.models.models: {e}")
    return ModelForm


# --- source audit: the discriminating guard lives in the router -------------


def test_router_preserves_base_model_id_when_not_submitted(open_webui_backend):
    """The fix: when the edit form didn't set `base_model_id`, the router must
    restore it from the existing model instead of letting the `None` default
    reach the writer. Requires both the `model_fields_set` guard AND the
    assignment back to the stored value in the endpoint body."""
    body = _function_body(open_webui_backend)
    guards_membership = re.search(
        r"['\"]base_model_id['\"]\s+not\s+in\s+form_data\.model_fields_set", body
    )
    restores_value = re.search(r"form_data\.base_model_id\s*=\s*model\.base_model_id", body)
    assert guards_membership and restores_value, (
        "update_model_by_id no longer preserves base_model_id for edit forms "
        "that omit it; workspace model edits regress to 54f31c630, silently "
        "clearing base_model_id and detaching the model from its upstream. "
        f"model_fields_set guard: {bool(guards_membership)}, "
        f"restore assignment: {bool(restores_value)}"
    )


# --- behavioral: prove the mechanism the guard depends on is real -----------


def test_omitted_base_model_id_is_absent_from_fields_set(open_webui_backend):
    """An edit payload without `base_model_id` yields a form where the field is
    NOT in `model_fields_set` and defaults to `None` — the exact precondition
    the guard keys on, and the value that clobbers the DB without it."""
    ModelForm = _load_model_form(open_webui_backend)
    form = ModelForm(
        **{"id": "my-model", "name": "My Model", "meta": {}, "params": {}, "access_grants": []}
    )
    assert "base_model_id" not in form.model_fields_set
    assert form.base_model_id is None


def test_explicit_base_model_id_stays_in_fields_set(open_webui_backend):
    """A payload that DOES send `base_model_id` keeps it in `model_fields_set`,
    so the guard leaves it alone — an explicit change (incl. an explicit detach)
    is honored, not overwritten by the preserved value."""
    ModelForm = _load_model_form(open_webui_backend)
    form = ModelForm(
        **{
            "id": "my-model",
            "name": "My Model",
            "meta": {},
            "params": {},
            "access_grants": [],
            "base_model_id": "gpt-4o",
        }
    )
    assert "base_model_id" in form.model_fields_set
    assert form.base_model_id == "gpt-4o"


def test_guard_logic_restores_existing_value(open_webui_backend):
    """Replicate the fix's two lines to show the outcome: an omitted field is
    restored to the stored upstream id, whereas the unguarded path (what 0.10.1
    ran) would send the defaulted `None` to the writer and wipe it."""
    ModelForm = _load_model_form(open_webui_backend)
    existing_base_model_id = "gpt-4o"

    form = ModelForm(
        **{"id": "my-model", "name": "Renamed", "meta": {}, "params": {}, "access_grants": []}
    )
    # Unguarded (0.10.1): the value the model layer would have written.
    assert form.base_model_id is None

    # Guarded (54f31c630): restore because it wasn't explicitly submitted.
    if "base_model_id" not in form.model_fields_set:
        form.base_model_id = existing_base_model_id
    assert form.base_model_id == existing_base_model_id
