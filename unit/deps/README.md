# Per-dependency contract tests (`unit/deps/`)

One `test_<dep>.py` per third-party dependency, pinning **exactly the API
surface and behaviour Open WebUI relies on** from that package. Run inside an
environment with a bumped version installed, these catch the "new release
removed / renamed / changed an API we use" class of breakage — the gap our
other suites can't see, because they run against whatever is already installed.

## The contract (so files compose and authors don't collide)

- **One file owns one dependency.** Never edit another dep's file, `conftest.py`,
  or `pyproject.toml`.
- **Shared machinery is fixtures, not imports.** Use the `depcheck` fixture
  (see `conftest.py`) and `open_webui_backend` (from `../conftest.py`). No
  cross-file imports — that keeps dozens of independently-authored modules from
  fighting over sys.path.
- **Marker:** `pytestmark = pytest.mark.depcheck`.
- **Skip, don't fail, when absent:** `depcheck.load(import_name)` skips the test
  if the package isn't importable, so the suite runs anywhere.
- **Deterministic + offline.** No real network, DB, redis, or model downloads —
  use in-memory fakes / mock transports, or assert the API surface instead.

## What a good file contains

1. **Symbol existence** — `depcheck.assert_symbols(mod, [...])` for every symbol
   the backend references (dotted paths like `"hazmat.primitives.hashes.SHA256"`).
2. **Signatures** — `depcheck.assert_params(fn, [...])` for functions called with
   specific kwargs (this is what catches the #24560-class "kwarg dropped" bug).
3. **Behavioural contracts** — exercise the *actual* usage offline (sign+verify a
   token, `chardet.detect()` a known byte string, a MockTransport HTTP roundtrip).

`test_requests.py` is the reference exemplar.

## `depcheck` API

`load` / `try_load` / `has` / `resolve` / `assert_symbols` / `assert_callable` /
`assert_params` / `dist_version`. See `conftest.py`.

## Gotchas

- Don't `hasattr()` an object whose attribute is a property that executes —
  use `dir(instance)` / class introspection (a blank object's getter can raise).
- Rust-backed classes (cryptography AEAD, etc.) have no introspectable
  signature — pin them behaviourally, not with `assert_params`.

## Running

```bash
pytest unit/deps/                    # all dependency contracts
pytest unit/deps/test_redis.py       # one dependency
pytest -m depcheck                   # the whole class, anywhere
```
