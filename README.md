# Open WebUI Test Suite

External test suite for [Open WebUI](https://github.com/open-webui/open-webui).

Four kinds of test, by how close they run to the product:

| Layer | Dir | Runs against | Needs a running instance? |
|-------|-----|--------------|---------------------------|
| **Unit** | `unit/` | the backend **source** (imported directly, or its source read and audited) | No |
| **Frontend** | `frontend/` | the frontend **source** (`src/lib` modules imported into vitest) | No |
| **Integration** | `integration/` | the HTTP **API** via `httpx` | Yes |
| **E2E** | `e2e/` | the **UI** via Playwright | Yes (+ browser) |

The bulk of the suite is `unit/` — fast, source-level regression tests pinned to specific upstream issues/PRs. They don't need a server: they import an `open_webui.*` module from a local checkout and exercise it with mocks, or read a source file and assert a contract over it. That's what makes them cheap enough to grow to thousands.

---

## Layout

```
tests/
├── conftest.py                 # browser + API fixtures (Playwright, httpx, auth, route lists)
├── pyproject.toml              # pytest config, marker registry, ruff/mypy
├── .env.example                # copy to .env for integration/e2e credentials
│
├── unit/                       # source-level tests — no running instance
│   ├── conftest.py             # source resolver + module-loader fixtures
│   ├── retrieval/              # RAG, web search, collection access control
│   ├── migrations/             # alembic schema: fresh install + full lifecycle
│   ├── tools/                  # builtin tool functions
│   ├── config/                 # boot / env / embedding-config safety
│   ├── chat/                   # chat message reconstruction
│   └── frontend/               # Svelte/TS source-contract audits (python, reads the files)
│
├── frontend/                   # vitest tests importing src/lib modules from the checkout
│   ├── package.json            # vitest only; app deps come from the checkout's node_modules
│   ├── vitest.config.ts        # $lib alias + resolver into the checkout
│   ├── shortcuts.test.ts
│   ├── i18n/
│   └── marked/
│
├── integration/                # httpx API tests, grouped by endpoint/router
│   ├── test_chat_completions.py
│   ├── test_notes.py
│   └── test_tasks.py
│
├── e2e/                        # Playwright UI tests
│   └── test_page_accessibility.py
│
└── utils/                      # shared helpers for the browser tests
```

**Where a new test goes**

- Exercises a backend function/module in isolation, or audits a source file → `unit/<subsystem>/`. Pick the subsystem dir that matches the code under test; add a new one if none fits (it's just a directory with an `__init__.py`).
- Calls a frontend `src/lib` module directly (TypeScript, vitest) → `frontend/<area>/<module>.test.ts`, importing through `$lib/...`.
- Hits an HTTP endpoint → `integration/test_<router>.py` (one file per router/endpoint group).
- Drives the browser → `e2e/`.

`unit/` is organised by **subsystem** (what part of the code), `integration/` by **endpoint** (what API surface). Both scale by adding files/dirs, not by growing existing files without bound.

---

## Setup

Python 3.11+. Install into any venv:

```bash
pip install -e ".[dev]"          # suite + ruff/mypy/pgserver
# or just the runtime deps:
pip install -e .
```

> `pip install -e .` works, but you can also install the dependency list directly if you prefer not to install the project package — see `pyproject.toml`.

For the **e2e** browser tests:

```bash
playwright install chromium
```

For the **postgres** migration tests (otherwise they skip):

```bash
pip install pgserver                     # embedded PostgreSQL, no system install
```

For **integration/e2e** credentials, copy and edit the env file:

```bash
cp .env.example .env
```

### Frontend suite

`frontend/` runs on vitest and imports `src/lib` modules straight out of the checkout, so the checkout needs its own `npm ci` first. Bare imports in a test (`marked`, `i18next`, `svelte/store`) resolve from the checkout's `node_modules`, so a test and the module it drives share one instance.

```bash
(cd ../open-webui && npm ci --force)
cd frontend && npm ci
OPEN_WEBUI_SOURCE_DIR=/path/to/open-webui/backend npx vitest run
```

Without `OPEN_WEBUI_SOURCE_DIR` it looks for a sibling `open-webui/` checkout, like the python suites.

### Pointing unit tests at the backend source

Unit tests need the Open WebUI **source tree** (not a server). Resolution order:

1. `OPEN_WEBUI_SOURCE_DIR` env var, if set, pointing at `.../open-webui/backend`.
2. Otherwise the `open_webui_backend` fixture walks up from the suite looking for a sibling `open-webui/backend/` checkout.

If neither resolves, the source-level tests **skip** (they never hard-fail for a missing checkout).

```bash
# explicit:
OPEN_WEBUI_SOURCE_DIR=/path/to/open-webui/backend pytest unit/

# implicit — works when this repo sits next to the open-webui checkout:
#   repos/
#   ├── open-webui/
#   └── tests/        <-- you are here
pytest unit/
```

`WEBUI_SECRET_KEY` is required by `open_webui.env` at import time; `unit/conftest.py` sets a throwaway default so you don't have to (a real value in the environment still wins).

---

## Running

```bash
pytest                                   # everything (integration/e2e skip without a server)
pytest unit/                             # all source-level tests — no server needed
pytest unit/retrieval/                   # one subsystem
pytest unit/retrieval/test_firecrawl.py  # one file
pytest -k collection_access              # name filter
pytest -m regression                     # only issue/PR-pinned regressions
pytest -m "not slow"                     # skip the long ones
pytest --lf                              # rerun last-failed
pytest -v                                # verbose (off by default; the suite is large)
(cd frontend && npx vitest run)          # the vitest suite; pytest never collects it
```

A run against the latest `dev` is expected to show **red for any regression whose fix isn't merged yet** — that's the point. Each failing test names the issue/PR that turns it green.

### Reports

```bash
pytest --html=reports/report.html --self-contained-html
pytest --alluredir=allure-results && allure serve allure-results
```

---

## Markers

Registered in `pyproject.toml` (`--strict-markers` is on, so an unregistered marker fails collection). Combine with `-m "<expr>"`.

| Marker | Axis | Meaning |
|--------|------|---------|
| `regression` | purpose | Pinned to a specific upstream issue/PR; fails only if that bug returns |
| `slow` | cost | Long-running (comprehensive scans, real postgres boot) |
| `public` | scope | Public pages, no auth |
| `auth_required` | scope | Needs an authenticated user |
| `admin_required` | scope | Needs an admin |
| `api` | type | API-level via `httpx`, no browser |
| `requires_source` | capability | Needs the backend source checkout |
| `requires_instance` | capability | Needs a running Open WebUI |
| `requires_browser` | capability | Needs Playwright browsers |
| `requires_postgres` | capability | Needs the `pgserver` package |

Capability markers are for **positive selection** in CI lanes. Tests also **auto-skip** when their dependency is absent (no source, no server, no `pgserver`, no browser), so you can run the whole suite anywhere and only the runnable subset executes.

---

## Writing tests

### The three unit patterns

1. **Behavioral** — import the real module from the checkout and drive it with mocks. Best when the function is callable in isolation.
   ```python
   async def test_search_web_coerces_string_count(builtin_tools_module):
       with patch.object(builtin_tools_module, "_search_web", AsyncMock(return_value=...)):
           out = await builtin_tools_module.search_web(query="x", count="3", ...)
       assert len(json.loads(out)) == 3
   ```

2. **Source audit** — read a source file and assert a contract over it. Best for code that's hard to call in isolation (Svelte components, shell scripts, cross-cutting invariants like "every numeric tool param is coerced").
   ```python
   def test_all_terminal_api_bearer_headers_are_normalized(open_webui_backend):
       src = (open_webui_backend.parent / "src" / "lib" / "apis" / "terminal" / "index.ts").read_text()
       ...  # assert no raw `Bearer ${token}` survives
   ```

3. **Subprocess** — run a real entrypoint (alembic, `start.sh`) in a child process and assert on exit code / output. Best for boot-time behavior that caches module state.

### Conventions

- **Lead the docstring with the issue:** `Regression for open-webui/open-webui#NNNNN.` then the before/after symptom. Future-you needs the link.
- **Assert the specific symptom**, not general behavior — a regression test should fail *only* if that bug comes back. Substring/contract assertions beat exact-match (wording drifts).
- **Verify discrimination:** a good regression test fails against the buggy ref and passes against the fix. Check both before committing (e.g. with `OPEN_WEBUI_SOURCE_DIR` pointed at a worktree of the fix branch).
- **Cover the class, not just the instance:** pair the specific repro with a broad guard (e.g. one behavioral test for the reported function + a source audit asserting *every* sibling does the right thing). This is what catches the *next* instance.
- **Clean up state:** integration tests that create notes/chats/files wrap in `try/finally` and delete in `finally`.

### Fixtures

**`unit/conftest.py`** (source-level)

| Fixture | Gives you |
|---------|-----------|
| `open_webui_backend` | `Path` to `.../open-webui/backend` (skips if not found) |
| `firecrawl_module` | imported `open_webui.retrieval.web.firecrawl` |
| `retrieval_utils_module` | imported `open_webui.retrieval.utils` |
| `retrieval_web_utils_module` | imported `open_webui.retrieval.web.utils` |
| `misc_module` | imported `open_webui.utils.misc` |
| `builtin_tools_module` | imported `open_webui.tools.builtin` |

Module-loader fixtures are session-scoped and `pytest.skip` if the import fails (missing dep). Need another module? Add a one-line loader fixture following the same pattern.

**`conftest.py`** (root — browser/API)

| Fixture | Gives you |
|---------|-----------|
| `api_client` | authenticated `httpx.Client`, `base_url` prefilled |
| `api_jwt` | a JWT (from `$API_JWT` or a signin) |
| `page` / `authenticated_page` / `admin_page` | Playwright pages |
| `config` | `AppConfig` from env |
| `public_routes` / `user_routes` / `admin_routes` | route lists for parametrization |

---

## Linting

```bash
ruff check .       # lint (E, F, I, W)
ruff format .      # format (line length 100)
```

Both are clean in CI. Run them before pushing.

---

## License

MIT — see LICENSE.
