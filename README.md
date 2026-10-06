# Open WebUI Test Suite

External regression suite for [Open WebUI](https://github.com/open-webui/open-webui).

| Layer | Dir | Drives | Needs |
|-------|-----|--------|-------|
| **Integration** | `integration/` | the HTTP API of a scratch instance the suite boots | the backend checkout |
| **E2E** | `e2e/` | the built frontend of that instance, in Chromium | the checkout, built (`npm run build`), and Playwright |
| **Unit** | `unit/` | backend modules imported from the checkout, or its source audited | the backend checkout |
| **Frontend** | `frontend/` | `src/lib` modules imported into vitest | the checkout's `node_modules` |

Upstream refactors its internals daily and changes its API and UI rarely, so a bug is pinned as far out as it can be seen: over HTTP when the symptom shows there and in the browser as well when it shows in the UI. No new unit tests are added; the ones in `unit/` stay until a twin further out replaces them. [`docs/regression-test-contract.md`](docs/regression-test-contract.md) has the rules; read it before adding or repairing a test.

---

## Layout

```
tests/
├── conftest.py            # loads the harness fixtures; drift label and skeptic note on failures
├── harness/               # scratch instances, the scripted model provider, accounts, local services
├── integration/<area>/    # httpx tests against the scratch instance, by subsystem
├── e2e/<area>/            # Playwright tests against the same instance, by subsystem
├── unit/<area>/           # source-level tests that nothing further out can see
├── frontend/              # vitest over src/lib
├── utils/                 # browser helpers (chat_ui.py: send, read replies)
├── scripts/               # e2e_instance.py (a manual instance), junit_summary.py (CI), seed_upgrade_data.py
└── docs/                  # the test contract
```

A twin of a unit test lives at the same `<area>/test_<name>.py` path under `integration/` or `e2e/`.

---

## Setup

Python 3.11+, with **uv** (`uv sync --extra dev`) or **pip** (`pip install -e ".[dev]"`). The suite imports and boots the backend from a checkout, so the checkout's own dependencies go into the same environment. They track the ref under test, which is why they are not in `uv.lock`:

```bash
uv pip install -r ../open-webui/backend/requirements.txt   # or: pip install -r ...
playwright install chromium                                 # for e2e/
(cd ../open-webui && npm ci --force && npm run build)       # for e2e/: the frontend it serves
```

The checkout is found through `OPEN_WEBUI_SOURCE_DIR` (pointing at `.../open-webui/backend`), else as a sibling `open-webui/` next to this repo. The frontend build is `build/` next to that backend, or `OPEN_WEBUI_BUILD_DIR`. The postgres migration tests also want `pgserver` (in the `dev` extra) and skip without it.

The Open Terminal tests start the real `open-terminal` from PyPI. Install it into an environment of its own, so its dependencies stay out of the backend's, and name its executable in `OPEN_TERMINAL_BIN` (or put it on `PATH`); the tests skip without it:

```bash
uv venv ../open-terminal && uv pip install --python ../open-terminal open-terminal
export OPEN_TERMINAL_BIN=../open-terminal/bin/open-terminal
```

### Frontend suite

`frontend/` imports `src/lib` modules straight out of the checkout, so the checkout needs `npm ci` and a `svelte-kit sync` first (its `tsconfig.json` extends the generated one).

```bash
(cd ../open-webui && npm ci --force && npx svelte-kit sync)
cd frontend && npm ci && npx vitest run
```

---

## Running

```bash
pytest integration e2e                   # boots the scratch instance once, about 15 s
pytest integration/security              # one area
pytest unit                              # set DATA_DIR and STATIC_DIR to scratch paths, see the contract
pytest -k collection_access              # name filter
pytest -m regression                     # only issue/PR-pinned regressions
pytest -m "not slow"                     # skip the modules that boot extra instances
```

A run against the latest `dev` is expected to show **red for any regression whose fix isn't merged yet**. Each failing test names the issue or PR that turns it green. Failures that stop before any assertion (an import, a renamed attribute, a changed signature) are listed apart as likely upstream drift.

A failing browser test leaves a Playwright trace per browser in `test-results/` (`playwright show-trace <file>.zip`). Set `OPEN_WEBUI_LOG_DIR` to keep each scratch instance's server log.

### Postgres and Redis

The scratch instances run on SQLite with in-process state by default. Two switches move every instance the suite boots onto the backends a multi-worker deployment uses:

```bash
OWUI_TEST_DATABASE=postgres pytest integration e2e   # a database of its own per instance, on one embedded Postgres (pgserver)
OWUI_TEST_REDIS=1 pytest integration e2e             # a redis-server of its own per instance, for sockets, config sync and tasks
```

They combine. The Redis switch needs `redis-server` on `PATH` and skips without it. An instance a test gives its own database or Redis (a prepared data directory, a Redis stand-in) keeps it. A test that only holds on one backend skips in the other mode and names why.

### Coverage

`OWUI_TEST_COVERAGE=1 pytest integration e2e` measures which backend lines the run executes (needs `pip install coverage`). Every scratch instance runs under coverage, as does the pytest process itself, so `pytest unit` can be measured the same way. At the end of the session the data is combined into `coverage-report/html/index.html` and `coverage-report/coverage.json` (`OWUI_TEST_COVERAGE_DIR` moves it), and the total is printed. Each session starts from empty data; to merge two sessions, give them separate directories and run `coverage combine --keep` over their `data/` folders.

`scripts/e2e_instance.py --clone ../open-webui` starts a standalone instance with the two seeded accounts, for poking at by hand.

`integration/migrations/test_upgrade_from_release.py` boots the checkout on data sets made by older releases (`integration/migrations/upgrade_data/`). To regenerate them, or add a release, run `python scripts/seed_upgrade_data.py --clone ../open-webui v0.9.6 v0.10.2 v0.11.4` (with `pgserver` installed for the Postgres sets) and commit the new files.

### CI

`.github/workflows/regression.yml` is called by Open WebUI's release pull requests with the ref under test. It runs the unit, integration, browser and vitest suites in parallel jobs; each writes a summary of failures to the job page and uploads its report, server logs and traces. A further job runs the integration suite with both switches on. It and the browser job report without gating a release for now.

The integration and browser suites run in parallel shards, split by `pytest-split` along the timings in `.test_durations`. A test missing from that file still runs and counts at the average, so the file only needs refreshing when the shards drift apart: run `pytest unit integration e2e --store-durations` against a current checkout and commit the result.

---

## Markers

Registered in `pyproject.toml` (`--strict-markers` is on). Combine with `-m "<expr>"`.

| Marker | Meaning |
|--------|---------|
| `regression` | pinned to a specific upstream issue/PR; fails only if that bug returns |
| `journey` | broad baseline coverage of a feature, not pinned to one issue |
| `slow` | long-running (extra instance boots, real postgres) |
| `api` | API-level via `httpx` |
| `requires_source` | needs the backend checkout |
| `requires_browser` | needs Playwright and a built frontend |
| `requires_postgres` | needs `pgserver` |
| `public` / `auth_required` / `admin_required` | page-access scope in the browser suite |
| `depcheck` | dependency contract test under `unit/deps/`, or its feature smoke test under `integration/deps/` |

Tests skip when what they need is absent, so the whole suite runs anywhere and only the runnable part executes.

---

## Fixtures

**`harness/fixtures.py`** (everywhere)

| Fixture | Gives you |
|---------|-----------|
| `instance` | the shared scratch instance (`.client()`, `.log_since()`, `.data_dir`) |
| `upstream` | its scripted model provider, reset per test: `queue(reply.text(...), reply.tool_call(...), reply.error(...))`, `chat_requests()` |
| `admin` / `user` / `make_user()` | accounts, each with its own token and client |
| `preserve(...)` | restores the global settings a test changes |
| `instance_with({...})` | a further instance for settings that only exist as environment variables |
| `package_instance_with({...})` | the same, kept for a whole test package whose modules share the env set |
| `refuses_unknown_names` | skips a c-ares test of a name that does not resolve where the machine's DNS server keeps c-ares from refusing one at once |
| `listener` | a local HTTP service for the instance to call, recording what it gets |
| `open_terminal` | a real Open Terminal for the session (`OPEN_TERMINAL_BIN`), skipped where it is not installed |

`harness.chat.ask(client, "text")` sends a message the way the web client does and returns the stored reply.

Local fakes, each described in its module docstring: `audio_engine` (speech and transcription), `realtime_provider` (a voice provider for realtime calls over OpenAI's Realtime API), `model_runners` (llama.cpp, LM Studio), `ldap_server`, `mfa` (an authenticator app, accounts enrolled in it and the admin's switches that require one), `external_knowledge` (Qdrant), `milvus_server` (Milvus over gRPC), `youtube` (YouTube behind the transcript loader's proxy), `duckduckgo` (DuckDuckGo's search page behind the same kind of proxy), `otlp_collector` (an OpenTelemetry trace collector over gRPC), `weaviate_server` (Weaviate over REST and gRPC), `local_voices` (tiny local text-to-speech voices, one per speaker), `outlook_message` (an Outlook .msg built in memory) `mcp_oauth` (an OAuth-protected MCP server), `http_proxy` (a forwarding proxy that knows names the instance cannot resolve) and `host_names` (a local service by `localhost`, an IPv6-only or another hosts-file name, and a name no resolver answers), plus model management and chat in `ollama_provider`, a Responses-API connection in `responses_provider`, a browser tab answering for a direct connection in `direct_connection`, a provider the browser itself calls for a user's own connection in `browser_provider`, token refresh in `oidc_provider`, live note edits in `socket_client` and an install without an optional package in `missing_packages`. `harness.access` sends one route as the owner, a stranger, a reader, a writer and the admin of a shared resource, and tells when a refused write still changed the owner's copy.

**`e2e/conftest.py`**: `page_for(actor)` opens a signed-in page in a browser of its own; `page`, `authenticated_page` and `admin_page` as before.

**`unit/conftest.py`**: `open_webui_backend` (the checkout path) and module loaders such as `builtin_tools_module`.

---

## Linting

```bash
ruff check .
ruff format .
```

---

## License

BSD-3-Clause, see LICENSE.
