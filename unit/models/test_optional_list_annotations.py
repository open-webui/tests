"""Regression: `list[X | None] = None` annotations that meant `list[X] | None = None`.

Fixed in Open WebUI 0.11.3 by commits 9962d122c (ModelsConfigForm.MODEL_ORDER_LIST,
WebConfig.WEB_SEARCH_DOMAIN_FILTER_LIST), b6d505522 (ModelForm.access_grants), e96b6464b
(PromptForm.access_grants, ToolForm.access_grants), 8ed548769 (MessageStats.tags) and 873fb741c
(PromptModel.tags, PromptForm.tags). The written form declares a required list whose ELEMENTS may
be null, with an invalid `None` default, so pydantic rejected a plain `null` payload with
`list_type` and happily accepted `[null]` into the list.

Discriminates: passes on v0.11.3, fails on v0.11.1 (v0.11.1 rejects None for these fields and
accepts a list containing a null element).
"""

from __future__ import annotations

import types
import typing

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

pytestmark = pytest.mark.regression


@pytest.fixture(scope="module")
def models_module(owui_module):
    return owui_module("open_webui.models.models")


@pytest.fixture(scope="module")
def prompts_module(owui_module):
    return owui_module("open_webui.models.prompts")


@pytest.fixture(scope="module")
def tools_module(owui_module):
    return owui_module("open_webui.models.tools")


@pytest.fixture(scope="module")
def chats_module(owui_module):
    return owui_module("open_webui.models.chats")


@pytest.fixture(scope="module")
def configs_router_module(owui_module):
    return owui_module("open_webui.routers.configs")


@pytest.fixture(scope="module")
def retrieval_router_module(owui_module):
    return owui_module("open_webui.routers.retrieval")


GRANT = {"subject_type": "user", "subject_id": "u1"}


def model_form_case(models_module):
    base = {"id": "m1", "name": "M", "meta": {}, "params": {}}
    return models_module.ModelForm, base, "access_grants", [GRANT]


def prompt_form_grants_case(prompts_module):
    base = {"command": "/c", "name": "P", "content": "body"}
    return prompts_module.PromptForm, base, "access_grants", [GRANT]


def prompt_form_tags_case(prompts_module):
    base = {"command": "/c", "name": "P", "content": "body"}
    return prompts_module.PromptForm, base, "tags", ["alpha"]


def prompt_model_tags_case(prompts_module):
    base = {"command": "/c", "user_id": "u1", "name": "P", "content": "body"}
    return prompts_module.PromptModel, base, "tags", ["alpha"]


def tool_form_case(tools_module):
    base = {"id": "t1", "name": "T", "content": "code", "meta": {}}
    return tools_module.ToolForm, base, "access_grants", [GRANT]


def message_stats_tags_case(chats_module):
    base = {"id": "msg1", "role": "user", "content_length": 4}
    return chats_module.MessageStats, base, "tags", ["alpha"]


def models_config_form_case(configs_router_module):
    base = {"DEFAULT_MODELS": None, "DEFAULT_PINNED_MODELS": None}
    return configs_router_module.ModelsConfigForm, base, "MODEL_ORDER_LIST", ["gpt-4"]


def web_config_case(retrieval_router_module):
    return retrieval_router_module.WebConfig, {}, "WEB_SEARCH_DOMAIN_FILTER_LIST", ["example.com"]


ALL_CASES = [
    ("ModelForm.access_grants", model_form_case, "models_module"),
    ("PromptForm.access_grants", prompt_form_grants_case, "prompts_module"),
    ("PromptForm.tags", prompt_form_tags_case, "prompts_module"),
    ("PromptModel.tags", prompt_model_tags_case, "prompts_module"),
    ("ToolForm.access_grants", tool_form_case, "tools_module"),
    ("MessageStats.tags", message_stats_tags_case, "chats_module"),
    ("ModelsConfigForm.MODEL_ORDER_LIST", models_config_form_case, "configs_router_module"),
    ("WebConfig.WEB_SEARCH_DOMAIN_FILTER_LIST", web_config_case, "retrieval_router_module"),
]

ACCESS_GRANT_FORMS = [
    ("ModelForm", "models_module", "ModelForm"),
    ("PromptForm", "prompts_module", "PromptForm"),
    ("ToolForm", "tools_module", "ToolForm"),
]

SWEPT_MODULES = [
    "models_module",
    "prompts_module",
    "tools_module",
    "chats_module",
    "configs_router_module",
    "retrieval_router_module",
]


def build(request, case):
    _label, factory, fixture_name = case
    return factory(request.getfixturevalue(fixture_name))


def annotation_allows_null_list(annotation) -> bool:
    """True for `list[X] | None`, False for `list[X | None]`."""
    origin = typing.get_origin(annotation)
    if origin not in (typing.Union, types.UnionType):
        return False
    args = typing.get_args(annotation)
    if type(None) not in args:
        return False
    return all(type(None) not in typing.get_args(arg) for arg in args if arg is not type(None))


def fields_defaulting_to_none(module):
    """Every pydantic field declared in this module whose default is a literal None."""
    for attr in vars(module).values():
        if not (isinstance(attr, type) and issubclass(attr, BaseModel)):
            continue
        if attr.__module__ != module.__name__:
            continue
        for name, field in attr.model_fields.items():
            if field.is_required() or field.default is not None:
                continue
            if field.default_factory is not None:
                continue
            yield attr.__name__, name, field


# --------------------------------------------------------------------------------------
# narrow: each field accepts an explicit null and rejects a list holding a null element
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_explicit_none_is_accepted(request, case):
    form, base, field, _valid = build(request, case)
    assert getattr(form(**{**base, field: None}), field) is None


@pytest.mark.parametrize("case", ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_list_containing_null_element_is_rejected(request, case):
    form, base, field, _valid = build(request, case)
    with pytest.raises(ValidationError) as excinfo:
        form(**{**base, field: [None]})
    assert excinfo.value.errors()[0]["loc"][:2] == (field, 0)


# --------------------------------------------------------------------------------------
# broad: nothing that defaults to None may carry an annotation that refuses None
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", SWEPT_MODULES)
def test_none_default_fields_accept_none(request, fixture_name):
    module = request.getfixturevalue(fixture_name)
    offenders = [
        f"{cls}.{name}: {field.annotation}"
        for cls, name, field in fields_defaulting_to_none(module)
        if not _accepts_none(field.annotation)
    ]
    assert not offenders, f"{module.__name__} declares {offenders}"


def _accepts_none(annotation) -> bool:
    try:
        TypeAdapter(annotation).validate_python(None)
    except ValidationError:
        return False
    return True


@pytest.mark.parametrize(
    "label,fixture_name,form_name", ACCESS_GRANT_FORMS, ids=[c[0] for c in ACCESS_GRANT_FORMS]
)
def test_access_grants_annotation_is_optional_list(request, label, fixture_name, form_name):
    form = getattr(request.getfixturevalue(fixture_name), form_name)
    annotation = form.model_fields["access_grants"].annotation
    assert annotation_allows_null_list(annotation), f"{label} declares {annotation}"


# --------------------------------------------------------------------------------------
# nearby: valid element lists and omitted fields keep working on both refs
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_valid_list_still_validates(request, case):
    form, base, field, valid = build(request, case)
    assert getattr(form(**{**base, field: valid}), field) == valid


@pytest.mark.parametrize("case", ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_empty_list_still_validates(request, case):
    form, base, field, _valid = build(request, case)
    assert getattr(form(**{**base, field: []}), field) == []


@pytest.mark.parametrize("case", ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_wrong_element_type_still_rejected(request, case):
    form, base, field, valid = build(request, case)
    wrong = 123 if isinstance(valid[0], (str, dict)) else "x"
    with pytest.raises(ValidationError):
        form(**{**base, field: [wrong]})


def test_optional_fields_default_when_omitted(models_module, prompts_module, tools_module):
    assert models_module.ModelForm(id="m1", name="M", meta={}, params={}).access_grants is None
    assert prompts_module.PromptForm(command="/c", name="P", content="body").access_grants is None
    assert prompts_module.PromptForm(command="/c", name="P", content="body").tags is None
    assert tools_module.ToolForm(id="t1", name="T", content="code", meta={}).access_grants is None


def test_model_order_list_stays_required(configs_router_module):
    """The fix widened the type while leaving the field required."""
    with pytest.raises(ValidationError):
        configs_router_module.ModelsConfigForm(DEFAULT_MODELS=None, DEFAULT_PINNED_MODELS=None)


def test_web_search_domain_filter_defaults_to_empty_list(retrieval_router_module):
    assert retrieval_router_module.WebConfig().WEB_SEARCH_DOMAIN_FILTER_LIST == []
