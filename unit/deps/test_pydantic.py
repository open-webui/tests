"""Dependency contract: pydantic (v2).

pydantic is the validation/serialization backbone of the Open WebUI
backend: nearly every request body, response schema, settings object, and
ORM-to-schema mapping is a `pydantic.BaseModel`. The code lives squarely
on the pydantic v2 API — `model_dump()`/`model_dump_json()`,
`model_validate()`, `ConfigDict`, `field_validator`/`model_validator`,
`create_model`, `pydantic.fields.FieldInfo` — and would break loudly if a
bump (e.g. 2.12 -> 2.13, or a slip back toward v1) removed, renamed, or
quietly changed any of it.

This module pins both halves of that contract:
  - the API surface (symbol existence + callability + version is v2.x);
  - the behaviour the backend depends on, exercised offline against
    representative `BaseModel` subclasses that mirror real codebase
    patterns (UserModel-style `ConfigDict(from_attributes=True)`,
    `ConfigDict(extra="allow")` settings models, scim.py alias fields,
    tool-spec `create_model`, profile-image `field_validator`, the
    `_ensure_profile_image` `model_validator(mode="after")`, etc.).

Exemplar for the unit/deps/ pattern: symbol-existence checks (API
surface) + offline behavioural contracts (no network). Uses the
`depcheck` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import datetime

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "pydantic"
DIST_NAME = "pydantic"

# Top-level symbols the Open WebUI backend imports from `pydantic`.
# (Confirmed via `from pydantic import ...` across backend/open_webui.)
USED_SYMBOLS = [
    "BaseModel",
    "Field",
    "ConfigDict",
    "field_validator",
    "model_validator",
    "create_model",
    "ValidationError",
    "HttpUrl",
    # legacy v1-compat shim still imported by routers/ollama.py
    "validator",
    # pydantic.fields.FieldInfo — utils/tools.py builds tool-arg specs with it
    "fields.FieldInfo",
]

# Methods/attrs the codebase calls on a BaseModel instance/class.
MODEL_API = [
    "model_dump",
    "model_dump_json",
    "model_validate",
    "model_validate_json",
    "model_config",
    "model_fields",
]


# ---------------------------------------------------------------------------
# Local model fixtures mirroring real codebase patterns. Defined at module
# scope so many tests share them; all are offline and deterministic.
# ---------------------------------------------------------------------------


def _models(mod):
    """Build a small zoo of BaseModel subclasses mirroring backend idioms.

    Returns a dict so each test can pull only what it needs without paying
    for a session fixture (the classes are cheap to define).
    """
    BaseModel = mod.BaseModel
    ConfigDict = mod.ConfigDict
    Field = mod.Field
    field_validator = mod.field_validator
    model_validator = mod.model_validator

    DEFAULT_IMG = "/api/v1/users/{user_id}/profile/image"

    class UserSettings(BaseModel):
        # models/users.py: UserSettings — Optional dict default + extra allow.
        ui: dict | None = {}
        model_config = ConfigDict(extra="allow")

    class UserModel(BaseModel):
        # models/users.py: UserModel — the central from_attributes schema,
        # nested optional submodel, scalar defaults, and an after-validator
        # that fills in a default profile image.
        id: str
        email: str
        username: str | None = None
        role: str = "pending"
        name: str
        profile_image_url: str | None = None
        date_of_birth: datetime.date | None = None
        settings: UserSettings | None = None
        created_at: int

        model_config = ConfigDict(from_attributes=True)

        @model_validator(mode="after")
        def _ensure_profile_image(self) -> "UserModel":
            self.profile_image_url = self.profile_image_url or DEFAULT_IMG.format(user_id=self.id)
            return self

    class UpdateProfileForm(BaseModel):
        # models/users.py: a field_validator normalising one field.
        profile_image_url: str
        name: str
        bio: str | None = None

        @field_validator("profile_image_url")
        @classmethod
        def check_profile_image_url(cls, v: str) -> str:
            if not v:
                raise ValueError("profile_image_url must be non-empty")
            return v.strip()

    class TaskItem(BaseModel):
        # tools/builtin.py: Field(...) required + Field(default, description).
        content: str = Field(..., description="Task description.")
        status: str = Field("pending", description="Task status.")
        id: str | None = Field(None, description="Auto-generated if omitted.")

    class ToolModel(BaseModel):
        # models/tools.py: Field(default_factory=list) + extra allow combo.
        id: str
        access_grants: list[str] = Field(default_factory=list)
        model_config = ConfigDict(extra="allow")

    class AliasModel(BaseModel):
        # routers/scim.py: aliased field + populate_by_name.
        ref: str | None = Field(None, alias="$ref")
        model_config = ConfigDict(populate_by_name=True)

    return {
        "UserSettings": UserSettings,
        "UserModel": UserModel,
        "UpdateProfileForm": UpdateProfileForm,
        "TaskItem": TaskItem,
        "ToolModel": ToolModel,
        "AliasModel": AliasModel,
        "DEFAULT_IMG": DEFAULT_IMG,
    }


# ---------------------------------------------------------------------------
# API surface: import + symbol existence + version is v2.
# ---------------------------------------------------------------------------


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "pydantic"


def test_used_symbols_exist(depcheck):
    """Every top-level pydantic symbol the codebase imports must exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_is_pydantic_v2(depcheck):
    """The codebase is v2-only (model_dump, ConfigDict, etc.). Guard against
    a slip back to a v1 line, which would silently break the whole API."""
    mod = depcheck.load(IMPORT_NAME)
    version = getattr(mod, "VERSION", None)
    assert version is not None, "pydantic.VERSION missing"
    major = int(str(version).split(".")[0])
    assert major == 2, f"expected pydantic v2, got {version}"


def test_version_reported(depcheck):
    """Sanity: the installed distribution version is resolvable."""
    assert depcheck.dist_version(DIST_NAME) is not None


def test_basemodel_is_class(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.BaseModel, type)


def test_field_is_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "Field")


def test_configdict_constructs(depcheck):
    """ConfigDict(extra=, from_attributes=, populate_by_name=) is used widely;
    it must accept those keys and produce a mapping."""
    mod = depcheck.load(IMPORT_NAME)
    cfg = mod.ConfigDict(extra="allow", from_attributes=True, populate_by_name=True)
    assert cfg["extra"] == "allow"
    assert cfg["from_attributes"] is True
    assert cfg["populate_by_name"] is True


def test_validator_decorators_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "field_validator")
    depcheck.assert_callable(mod, "model_validator")


def test_create_model_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "create_model")


def test_fieldinfo_importable(depcheck):
    """utils/tools.py does `from pydantic.fields import FieldInfo`."""
    mod = depcheck.load(IMPORT_NAME)
    assert depcheck.has(mod, "fields.FieldInfo")
    fi = depcheck.resolve(mod, "fields.FieldInfo")
    assert isinstance(fi, type)


def test_validationerror_is_exception(depcheck):
    """Routers/models catch and surface pydantic.ValidationError."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.ValidationError, Exception)


def test_model_instance_api_present(depcheck):
    """A constructed model must expose the v2 method/attr surface the
    codebase calls (model_dump, model_validate, model_config, ...)."""
    mod = depcheck.load(IMPORT_NAME)
    m = _models(mod)["TaskItem"](content="x")
    names = set(dir(m))
    for attr in MODEL_API:
        assert attr in names, f"BaseModel.{attr} missing in this pydantic"
    assert callable(m.model_dump)
    assert callable(m.model_dump_json)
    assert callable(type(m).model_validate)


def test_legacy_v1_namespace_absent(depcheck):
    """v1 used `.dict()` / `.json()` / `parse_obj` on instances. The backend
    migrated off them; if a future pydantic re-introduced *only* the v1
    spelling and dropped model_dump, the v2 calls would break. Assert the
    v2 names exist (the positive guard); we do not forbid the legacy ones."""
    mod = depcheck.load(IMPORT_NAME)
    m = _models(mod)["TaskItem"](content="x")
    assert hasattr(m, "model_dump")
    assert hasattr(m, "model_dump_json")


# ---------------------------------------------------------------------------
# Behavioural contracts: construction + validation.
# ---------------------------------------------------------------------------


def test_basic_construction_and_attr_access(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["UserModel"]
    u = M(id="u1", email="a@b.c", name="Alice", created_at=1700000000)
    assert u.id == "u1"
    assert u.email == "a@b.c"
    assert u.name == "Alice"
    assert u.created_at == 1700000000


def test_field_defaults_applied(depcheck):
    """Scalar defaults (role='pending') and None defaults must hold."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["UserModel"]
    u = M(id="u1", email="a@b.c", name="Alice", created_at=1)
    assert u.role == "pending"
    assert u.username is None
    assert u.settings is None


def test_optional_none_handling(depcheck):
    """`x | None = None` fields accept None and an explicit value alike."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["UserModel"]
    u = M(id="u1", email="a@b.c", name="A", created_at=1, username=None)
    assert u.username is None
    u2 = M(id="u1", email="a@b.c", name="A", created_at=1, username="alice")
    assert u2.username == "alice"


def test_required_field_missing_raises(depcheck):
    """Omitting a required field (no default) raises ValidationError."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["UserModel"]
    with pytest.raises(mod.ValidationError):
        M(email="a@b.c", name="A", created_at=1)  # missing id


def test_type_coercion_and_rejection(depcheck):
    """int field coerces a numeric string but rejects a non-numeric one —
    the v2 strict-ish coercion the request models rely on."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["UserModel"]
    u = M(id="u1", email="a@b.c", name="A", created_at="1700000000")
    assert u.created_at == 1700000000
    assert isinstance(u.created_at, int)
    with pytest.raises(mod.ValidationError):
        M(id="u1", email="a@b.c", name="A", created_at="not-an-int")


def test_nested_model_construction(depcheck):
    """A nested submodel (UserModel.settings: UserSettings) is built from a
    dict on construction."""
    mod = depcheck.load(IMPORT_NAME)
    models = _models(mod)
    M = models["UserModel"]
    u = M(
        id="u1",
        email="a@b.c",
        name="A",
        created_at=1,
        settings={"ui": {"theme": "dark"}},
    )
    assert isinstance(u.settings, models["UserSettings"])
    assert u.settings.ui == {"theme": "dark"}


def test_date_field_parses_iso_string(depcheck):
    """date_of_birth: datetime.date parses an ISO string (request bodies send
    JSON strings)."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["UserModel"]
    u = M(
        id="u1",
        email="a@b.c",
        name="A",
        created_at=1,
        date_of_birth="1990-05-17",
    )
    assert u.date_of_birth == datetime.date(1990, 5, 17)


# ---------------------------------------------------------------------------
# Behavioural contracts: model_dump / model_dump_json shapes.
# ---------------------------------------------------------------------------


def test_model_dump_returns_plain_dict(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["TaskItem"]
    d = M(content="do it").model_dump()
    assert isinstance(d, dict)
    assert d == {"content": "do it", "status": "pending", "id": None}


def test_model_dump_exclude_none(depcheck):
    """`model_dump(exclude_none=True)` is used (e.g. users update form,
    builtin tasks). None-valued keys must drop out."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["TaskItem"]
    d = M(content="x").model_dump(exclude_none=True)
    assert "id" not in d
    assert d == {"content": "x", "status": "pending"}


def test_model_dump_include_subset(depcheck):
    """utils/audit.py: model_dump(include={...}) projects a field subset."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["UserModel"]
    u = M(id="u1", email="a@b.c", name="A", created_at=1)
    d = u.model_dump(include={"id", "email"})
    assert set(d.keys()) == {"id", "email"}


def test_model_dump_exclude_subset(depcheck):
    """models/tools.py: model_dump(exclude={'access_grants'})."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["ToolModel"]
    t = M(id="t1", access_grants=["g1"])
    d = t.model_dump(exclude={"access_grants"})
    assert "access_grants" not in d
    assert d["id"] == "t1"


def test_model_dump_mode_json_serializes_date(depcheck):
    """main.py/mcp client use model_dump(mode='json'); date must come out as
    an ISO string, not a datetime.date object."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["UserModel"]
    u = M(
        id="u1",
        email="a@b.c",
        name="A",
        created_at=1,
        date_of_birth=datetime.date(1990, 5, 17),
    )
    plain = u.model_dump()
    jsonish = u.model_dump(mode="json")
    assert plain["date_of_birth"] == datetime.date(1990, 5, 17)
    assert jsonish["date_of_birth"] == "1990-05-17"


def test_model_dump_json_is_string(depcheck):
    """functions.py: line.model_dump_json() must yield a JSON string."""
    import json

    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["TaskItem"]
    s = M(content="x").model_dump_json()
    assert isinstance(s, str)
    assert json.loads(s) == {"content": "x", "status": "pending", "id": None}


def test_model_dump_json_exclude_none(depcheck):
    """utils/oauth.py: user.model_dump_json(exclude_none=True)."""
    import json

    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["TaskItem"]
    s = M(content="x").model_dump_json(exclude_none=True)
    assert "id" not in json.loads(s)


def test_round_trip_dump_then_construct(depcheck):
    """models/chats.py does `Chat(**chat.model_dump())` — dump then splat
    back into a constructor must reproduce the model."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["UserModel"]
    u = M(id="u1", email="a@b.c", name="A", created_at=1)
    u2 = M(**u.model_dump())
    assert u2.model_dump() == u.model_dump()


# ---------------------------------------------------------------------------
# Behavioural contracts: model_validate / from_attributes (ORM mapping).
# ---------------------------------------------------------------------------


def test_model_validate_from_dict(depcheck):
    """OAuth/SCIM flows call Model.model_validate(<dict>)."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["UserModel"]
    u = M.model_validate({"id": "u1", "email": "a@b.c", "name": "A", "created_at": 1})
    assert u.id == "u1"


def test_model_validate_from_attributes(depcheck):
    """The dominant ORM pattern: UserModel.model_validate(<SQLAlchemy row>)
    with ConfigDict(from_attributes=True) reads attributes off an object."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["UserModel"]

    class Row:
        id = "u1"
        email = "a@b.c"
        username = None
        role = "user"
        name = "Alice"
        profile_image_url = "/img.png"
        date_of_birth = None
        settings = None
        created_at = 1700000000

    u = M.model_validate(Row())
    assert u.id == "u1"
    assert u.role == "user"
    assert u.name == "Alice"
    assert u.created_at == 1700000000


def test_from_attributes_requires_config(depcheck):
    """A model WITHOUT from_attributes must NOT silently accept an arbitrary
    object — guards against the config flag becoming a no-op."""
    mod = depcheck.load(IMPORT_NAME)
    BaseModel = mod.BaseModel

    class Plain(BaseModel):
        id: str

    class Obj:
        id = "x"

    with pytest.raises(mod.ValidationError):
        Plain.model_validate(Obj())


def test_model_validate_json(depcheck):
    """model_validate_json parses a JSON string into a model."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["TaskItem"]
    m = M.model_validate_json('{"content": "x", "status": "done"}')
    assert m.content == "x"
    assert m.status == "done"


# ---------------------------------------------------------------------------
# Behavioural contracts: ConfigDict(extra=...) behaviour.
# ---------------------------------------------------------------------------


def test_extra_allow_keeps_unknown_fields(depcheck):
    """ConfigDict(extra='allow') (UserSettings, ToolModel, many *Model
    classes): unknown keys are kept and surface in model_dump()."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["UserSettings"]
    s = M(ui={"a": 1}, custom_flag=True, other="z")
    assert s.custom_flag is True
    assert s.other == "z"
    dumped = s.model_dump()
    assert dumped["custom_flag"] is True
    assert dumped["other"] == "z"


def test_default_extra_is_ignore(depcheck):
    """A model with no extra config (pydantic default 'ignore') drops unknown
    keys rather than raising — the common response-model behaviour."""
    mod = depcheck.load(IMPORT_NAME)
    BaseModel = mod.BaseModel

    class Strict(BaseModel):
        a: int

    m = Strict(a=1, b=2, c=3)
    assert m.model_dump() == {"a": 1}


def test_extra_forbid_raises(depcheck):
    """extra='forbid' must reject unknown keys (used where strict bodies are
    required); confirms the extra knob is honoured in all three modes."""
    mod = depcheck.load(IMPORT_NAME)
    BaseModel = mod.BaseModel
    ConfigDict = mod.ConfigDict

    class NoExtra(BaseModel):
        model_config = ConfigDict(extra="forbid")
        a: int

    with pytest.raises(mod.ValidationError):
        NoExtra(a=1, b=2)


def test_subclass_overrides_config(depcheck):
    """models/users.py: UserModelResponse(UserModel) re-declares
    model_config = ConfigDict(extra='allow'). A subclass config override
    must take effect."""
    mod = depcheck.load(IMPORT_NAME)
    models = _models(mod)
    Base = models["UserModel"]
    ConfigDict = mod.ConfigDict

    class Resp(Base):
        model_config = ConfigDict(extra="allow", from_attributes=True)

    r = Resp(id="u1", email="a@b.c", name="A", created_at=1, surprise=9)
    assert r.surprise == 9


# ---------------------------------------------------------------------------
# Behavioural contracts: field_validator firing.
# ---------------------------------------------------------------------------


def test_field_validator_transforms_value(depcheck):
    """UpdateProfileForm.check_profile_image_url normalises its field — the
    validator must run on construction and the result must be stored."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["UpdateProfileForm"]
    f = M(profile_image_url="  /img.png  ", name="A")
    assert f.profile_image_url == "/img.png"


def test_field_validator_rejects_bad_value(depcheck):
    """A validator that raises ValueError surfaces as ValidationError."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["UpdateProfileForm"]
    with pytest.raises(mod.ValidationError):
        M(profile_image_url="", name="A")


def test_field_validator_mode_before(depcheck):
    """models/users.py uses field_validator(..., mode='before'); a before-
    validator sees the raw input prior to coercion."""
    mod = depcheck.load(IMPORT_NAME)
    BaseModel = mod.BaseModel
    field_validator = mod.field_validator

    class M(BaseModel):
        n: int

        @field_validator("n", mode="before")
        @classmethod
        def coerce(cls, v):
            if v is None:
                return 0
            return v

    assert M(n=None).n == 0
    assert M(n=5).n == 5


# ---------------------------------------------------------------------------
# Behavioural contracts: model_validator firing.
# ---------------------------------------------------------------------------


def test_model_validator_after_fills_default(depcheck):
    """UserModel._ensure_profile_image (model_validator mode='after') derives
    a profile image when none is given — must run and mutate self."""
    mod = depcheck.load(IMPORT_NAME)
    models = _models(mod)
    M = models["UserModel"]
    u = M(id="u42", email="a@b.c", name="A", created_at=1)
    assert u.profile_image_url == models["DEFAULT_IMG"].format(user_id="u42")


def test_model_validator_after_respects_existing(depcheck):
    """When a value IS provided, the after-validator must keep it."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["UserModel"]
    u = M(
        id="u42",
        email="a@b.c",
        name="A",
        created_at=1,
        profile_image_url="/custom.png",
    )
    assert u.profile_image_url == "/custom.png"


def test_model_validator_before(depcheck):
    """models/files.py uses model_validator(mode='before') to massage the
    incoming dict before field validation."""
    mod = depcheck.load(IMPORT_NAME)
    BaseModel = mod.BaseModel
    model_validator = mod.model_validator

    class M(BaseModel):
        a: int
        b: int = 0

        @model_validator(mode="before")
        @classmethod
        def inject(cls, data):
            if isinstance(data, dict) and "b" not in data:
                data = {**data, "b": data.get("a", 0) * 2}
            return data

    m = M(a=3)
    assert m.b == 6


# ---------------------------------------------------------------------------
# Behavioural contracts: Field() — defaults, default_factory, description,
# aliases.
# ---------------------------------------------------------------------------


def test_field_required_ellipsis(depcheck):
    """tools/builtin.py: `content: str = Field(..., description=...)`.
    Field(...) marks the field required."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["TaskItem"]
    with pytest.raises(mod.ValidationError):
        M(status="pending")  # content (Field(...)) missing


def test_field_default_value(depcheck):
    """Field('pending', ...) supplies a default."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["TaskItem"]
    assert M(content="x").status == "pending"


def test_field_default_factory(depcheck):
    """models/tools.py: Field(default_factory=list) — each instance gets its
    own fresh list (no shared-mutable-default bug)."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["ToolModel"]
    a = M(id="a")
    b = M(id="b")
    a.access_grants.append("x")
    assert a.access_grants == ["x"]
    assert b.access_grants == []  # not shared


def test_field_description_in_schema(depcheck):
    """Field descriptions feed the OpenAPI/tool JSON schema; they must land
    in model_json_schema() (used to build tool specs)."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["TaskItem"]
    schema = M.model_json_schema()
    assert schema["properties"]["content"]["description"] == "Task description."


def test_field_alias_population(depcheck):
    """routers/scim.py: Field(alias='$ref'). With the alias, input under the
    alias key populates the field."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["AliasModel"]
    m = M(**{"$ref": "https://x/y"})
    assert m.ref == "https://x/y"


def test_field_populate_by_name(depcheck):
    """scim.py models set ConfigDict(populate_by_name=True), so the Python
    field name also works as an input key (not only the alias)."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["AliasModel"]
    m = M(ref="by-name")
    assert m.ref == "by-name"


def test_dump_by_alias(depcheck):
    """utils/oauth.py: model_dump(..., by_alias=True). Output keys use the
    alias, so SCIM/OAuth payloads serialise with the wire field names."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["AliasModel"]
    d = M(ref="v").model_dump(by_alias=True)
    assert d == {"$ref": "v"}
    d2 = M(ref="v").model_dump()  # default: by field name
    assert d2 == {"ref": "v"}


# ---------------------------------------------------------------------------
# Behavioural contracts: ValidationError shape.
# ---------------------------------------------------------------------------


def test_validation_error_errors_structure(depcheck):
    """Error handlers read e.errors(); each entry must carry loc/msg/type."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["UserModel"]
    with pytest.raises(mod.ValidationError) as ei:
        M(id="u1", email="a@b.c", name="A", created_at="nope")
    errors = ei.value.errors()
    assert isinstance(errors, list) and errors
    first = errors[0]
    for key in ("loc", "msg", "type"):
        assert key in first, f"ValidationError entry missing {key!r}"
    assert first["loc"] == ("created_at",)


def test_validation_error_collects_multiple(depcheck):
    """Multiple bad/missing fields are reported together (not fail-fast),
    which the API relies on to return all field errors at once."""
    mod = depcheck.load(IMPORT_NAME)
    BaseModel = mod.BaseModel

    class M(BaseModel):
        a: int
        b: int

    with pytest.raises(mod.ValidationError) as ei:
        M()  # both missing
    assert ei.value.error_count() == 2


def test_validation_error_json_method(depcheck):
    """ValidationError.json() yields a JSON string (used when surfacing
    structured errors)."""
    import json

    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["UserModel"]
    with pytest.raises(mod.ValidationError) as ei:
        M(id="u1", email="a@b.c", name="A", created_at="nope")
    payload = ei.value.json()
    assert isinstance(payload, str)
    assert isinstance(json.loads(payload), list)


# ---------------------------------------------------------------------------
# Behavioural contracts: create_model + FieldInfo (utils/tools.py tool specs).
# ---------------------------------------------------------------------------


def test_create_model_basic(depcheck):
    """utils/tools.py builds a model dynamically from a function signature:
    create_model(name, field=(type, default)). Required uses `...`."""
    mod = depcheck.load(IMPORT_NAME)
    Dyn = mod.create_model(
        "Dyn",
        a=(int, ...),
        b=(str, "default"),
    )
    assert set(Dyn.model_fields.keys()) == {"a", "b"}
    inst = Dyn(a=5)
    assert inst.a == 5
    assert inst.b == "default"
    with pytest.raises(mod.ValidationError):
        Dyn()  # a is required


def test_create_model_with_field_descriptor(depcheck):
    """utils/tools.py: create_model(..., name=(type, Field(default, desc))).
    The Field() in the tuple must apply description + default."""
    mod = depcheck.load(IMPORT_NAME)
    Field = mod.Field
    Dyn = mod.create_model(
        "DynDesc",
        q=(str, Field("hi", description="a query")),
    )
    inst = Dyn()
    assert inst.q == "hi"
    schema = Dyn.model_json_schema()
    assert schema["properties"]["q"]["description"] == "a query"


def test_create_model_sets_docstring(depcheck):
    """utils/tools.py sets `model.__doc__ = function_description` after
    create_model — the dynamic class must accept a __doc__ assignment."""
    mod = depcheck.load(IMPORT_NAME)
    Dyn = mod.create_model("DynDoc", x=(int, 0))
    Dyn.__doc__ = "a tool"
    assert Dyn.__doc__ == "a tool"


def test_model_fields_carry_fieldinfo(depcheck):
    """model_fields maps names -> FieldInfo; utils/tools.py reads FieldInfo
    off models. Each entry must be a pydantic.fields.FieldInfo."""
    mod = depcheck.load(IMPORT_NAME)
    FieldInfo = depcheck.resolve(mod, "fields.FieldInfo")
    M = _models(mod)["TaskItem"]
    fields = M.model_fields
    assert isinstance(fields, dict)
    assert set(fields.keys()) == {"content", "status", "id"}
    for fi in fields.values():
        assert isinstance(fi, FieldInfo)
    # FieldInfo exposes the attributes utils/tools.py-style code reads.
    content_fi = fields["content"]
    assert content_fi.description == "Task description."
    assert content_fi.is_required() is True
    assert fields["status"].is_required() is False


# ---------------------------------------------------------------------------
# Behavioural contracts: HttpUrl (routers/tools.py, routers/functions.py).
# ---------------------------------------------------------------------------


def test_httpurl_accepts_valid_url(depcheck):
    """routers/tools.py & functions.py type a field as HttpUrl. A valid URL
    must validate; str() must give back the URL text."""
    mod = depcheck.load(IMPORT_NAME)
    BaseModel = mod.BaseModel

    class M(BaseModel):
        url: mod.HttpUrl

    m = M(url="https://example.com/path")
    assert str(m.url).startswith("https://example.com")


def test_httpurl_rejects_invalid(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    BaseModel = mod.BaseModel

    class M(BaseModel):
        url: mod.HttpUrl

    with pytest.raises(mod.ValidationError):
        M(url="not a url")


# ---------------------------------------------------------------------------
# Behavioural contracts: inheritance, model_copy, and equality.
# ---------------------------------------------------------------------------


def test_model_inheritance_adds_fields(depcheck):
    """models/users.py: UserStatusModel(UserModel) adds `is_active`. Subclass
    must inherit parent fields and add its own."""
    mod = depcheck.load(IMPORT_NAME)
    Base = _models(mod)["UserModel"]
    ConfigDict = mod.ConfigDict

    class StatusModel(Base):
        is_active: bool = False
        model_config = ConfigDict(from_attributes=True)

    s = StatusModel(id="u1", email="a@b.c", name="A", created_at=1)
    assert s.is_active is False
    assert s.id == "u1"
    assert "is_active" in s.model_dump()
    assert "email" in s.model_dump()


def test_model_copy_update(depcheck):
    """model_copy(update=...) clones with overrides (a common v2 idiom for
    deriving a tweaked model without re-validating everything)."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["TaskItem"]
    a = M(content="x")
    b = a.model_copy(update={"status": "done"})
    assert b.status == "done"
    assert a.status == "pending"  # original untouched


def test_model_equality(depcheck):
    """Two models with identical data compare equal (relied on implicitly in
    list/dedup logic)."""
    mod = depcheck.load(IMPORT_NAME)
    M = _models(mod)["TaskItem"]
    assert M(content="x") == M(content="x")
    assert M(content="x") != M(content="y")


# ---------------------------------------------------------------------------
# pydantic_settings: present as a sibling distribution (not used for
# BaseSettings in the backend, but installed and importable). Lightweight
# availability probe so a bump that drops it is noticed.
# ---------------------------------------------------------------------------


def test_pydantic_settings_importable_optional(depcheck):
    """pydantic_settings ships alongside pydantic v2. The backend config
    layer uses plain BaseModel, but the package should remain importable and
    expose BaseSettings; skip cleanly if it isn't installed."""
    mod = depcheck.try_load("pydantic_settings")
    if mod is None:
        pytest.skip("pydantic_settings not installed in this env")
    assert hasattr(mod, "BaseSettings")
    assert isinstance(mod.BaseSettings, type)
