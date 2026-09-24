"""Sweep: no pydantic field that defaults to `None` may refuse `None`.

The broad layer of the 0.11.3 fixes `9962d122c`, `b6d505522`, `e96b6464b`, `8ed548769` and
`873fb741c`, which rewrote `list[X | None] = None` as `list[X] | None = None`. The written form
refuses the default it declares, so a client sending `null` got a 422. The request forms are
pinned over HTTP by integration/models/test_optional_list_annotations.py; this sweep also covers
the response and stats models (`PromptModel.tags`, `MessageStats.tags`) and the next field that
repeats the mistake.

Discriminates: passes on upstream dev `bbfa876af`; with `ModelForm.access_grants`,
`PromptForm.tags`/`access_grants`, `PromptModel.tags`, `ToolForm.access_grants` and
`MessageStats.tags` back on `list[X | None] = None`, the models, prompts, tools and chats sweeps
fail.
"""

from __future__ import annotations

import importlib

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

pytestmark = pytest.mark.regression

SWEPT_MODULES = (
    "open_webui.models.models",
    "open_webui.models.prompts",
    "open_webui.models.tools",
    "open_webui.models.chats",
    "open_webui.routers.configs",
    "open_webui.routers.retrieval",
)


def _import(owui_module, name: str):
    owui_module("open_webui")  # puts the checkout on sys.path
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        pytest.fail(f"{name} is gone; retarget the sweep at the module that now declares its forms")


def _fields_defaulting_to_none(module):
    for declared in vars(module).values():
        is_local_model = (
            isinstance(declared, type)
            and issubclass(declared, BaseModel)
            and declared.__module__ == module.__name__
        )
        if not is_local_model:
            continue
        for name, field in declared.model_fields.items():
            if not field.is_required() and field.default_factory is None and field.default is None:
                yield f"{declared.__name__}.{name}", field.annotation


def _accepts_none(annotation) -> bool:
    try:
        TypeAdapter(annotation).validate_python(None)
    except ValidationError:
        return False
    return True


@pytest.mark.parametrize("module_name", SWEPT_MODULES)
def test_every_field_defaulting_to_none_accepts_none(owui_module, module_name):
    fields = dict(_fields_defaulting_to_none(_import(owui_module, module_name)))
    assert fields, f"{module_name} declares no field defaulting to None; the sweep saw nothing"

    offenders = [
        f"{name}: {annotation}"
        for name, annotation in fields.items()
        if not _accepts_none(annotation)
    ]
    assert not offenders, (
        f"{module_name} declares a None default its annotation refuses: {offenders}"
    )
