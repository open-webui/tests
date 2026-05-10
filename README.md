# Open WebUI Test Suite

External test suite for [Open WebUI](https://github.com/open-webui/open-webui) that provides:

- **Page Accessibility Tests** (`e2e/`): Playwright UI smoke tests that load each route and check for console errors
- **API Integration Tests** (`integration/`): `httpx`-based tests that hit the backend HTTP API directly — fast, no browser, ideal for regression coverage of specific issues/PRs
- **SSO / DB Integration Tests**: planned
- **Reporting Dashboard**: Allure reports for test result visualization (planned)

## TL;DR — run everything against a local Open WebUI

```bash
# 1. Open WebUI must be running locally, e.g. http://localhost:8080
# 2. Install the test deps into any Python 3.11+ env
pip install "pytest~=8.4.1" "pytest-playwright>=0.5.0" "playwright==1.49.1" \
            "httpx==0.28.1" "python-dotenv>=1.0.0" "allure-pytest>=2.13.0" \
            "pytest-html>=4.0.0"
playwright install chromium

# 3. Configure credentials (or set API_JWT directly)
cp .env.example .env
$EDITOR .env

# 4. Run everything
pytest                                   # all tests
pytest -m api                            # only API integration tests (no browser)
pytest -m regression                     # only regression tests pinned to issues/PRs
pytest -m "api and regression"           # both: only API regression tests
pytest -m "not slow"                     # skip slow comprehensive scans
```

## Prerequisites

- Python 3.11+
- A running Open WebUI instance to test against
- Test user accounts (regular user and admin) created in Open WebUI

## Installation

1. **Create a virtual environment** (or reuse one — the suite has no `setup.py`/`__init__.py` of its own that pip needs to install):

   ```bash
   cd tests
   python -m venv venv
   source venv/bin/activate                # macOS/Linux
   .\venv\Scripts\Activate.ps1             # Windows PowerShell
   ```

2. **Install dependencies.** `pip install -e .` is currently broken because setuptools auto-discovery can't pick a single top-level package from `e2e/`, `utils/`, and `integration/` — install from the dep list directly:

   ```bash
   pip install "pytest~=8.4.1" "pytest-asyncio>=0.23.0" \
               "pytest-playwright>=0.5.0" "playwright==1.49.1" \
               "httpx==0.28.1" "python-dotenv>=1.0.0" \
               "allure-pytest>=2.13.0" "pytest-html>=4.0.0"
   ```

3. **Install Playwright browsers** (only needed for the `e2e/` UI tests):

   ```bash
   playwright install chromium
   ```

4. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env with your Open WebUI URL and test credentials
   ```

## Configuration

Edit `.env` to configure your test environment:

```bash
# Base URL of the Open WebUI instance to test
OPEN_WEBUI_URL=http://localhost:8080

# Test user credentials (must exist in Open WebUI)
TEST_USER_EMAIL=test@example.com
TEST_USER_PASSWORD=testpassword123

# Admin user credentials (must have admin role in Open WebUI)
ADMIN_USER_EMAIL=admin@example.com
ADMIN_USER_PASSWORD=adminpassword123

# Browser settings
HEADLESS=true          # Set to false to see browser during tests
SLOW_MO=0              # Milliseconds to slow down operations (useful for debugging)

# Timeout settings (milliseconds)
DEFAULT_TIMEOUT=30000
NAVIGATION_TIMEOUT=60000

# Optional: pre-issued JWT for API tests. If set, the api_client fixture
# uses it directly and skips the signin call (useful for CI or when the
# test user is OAuth-only and has no usable password).
# API_JWT=
```

### Getting an `API_JWT`

If you'd rather not put credentials in `.env`, mint a JWT once and reuse it:

```bash
curl -s -X POST "$OPEN_WEBUI_URL/api/v1/auths/signin" \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"..."}' | jq -r .token
```

Then export it before running tests:

```bash
export API_JWT="eyJhbGciOi..."
pytest -m api
```

## Running Tests

### Toggling test categories with markers

Every test carries one or more markers (see the [marker table below](#test-markers)). Markers are how you switch test sets on and off without editing files. The grammar is the standard pytest expression:

```bash
pytest -m api                            # only API tests (no browser)
pytest -m "auth_required and not slow"   # logged-in tests, but skip the comprehensive scans
pytest -m "regression"                   # only regressions tied to a specific issue/PR
pytest -m "public or api"                # union of two categories
pytest -m "not admin_required"           # everything that doesn't need an admin account
```

The markers are orthogonal axes (type / purpose / scope / cost) — combine them freely.

### Common invocations

```bash
pytest                                              # all tests
pytest e2e/test_page_accessibility.py -v            # single file
pytest e2e -m public                                # public UI smoke only
pytest integration -m regression                    # all regression API tests
pytest -k chat_completion                           # name-substring filter
pytest --lf                                         # rerun last-failed only
HEADLESS=false SLOW_MO=300 pytest e2e -m public     # watch the browser
```

### Reports

```bash
pytest --html=reports/report.html --self-contained-html    # standalone HTML

pytest --alluredir=allure-results                          # Allure
allure serve allure-results
```

### Running just the API integration tests

The `integration/` suite uses `httpx` — no browser, no playwright. Fast enough for a pre-push hook:

```bash
# With credentials in .env:
pytest integration -v

# Or pass a JWT directly (good for CI):
API_JWT="$(curl -sX POST $OPEN_WEBUI_URL/api/v1/auths/signin \
  -H 'Content-Type: application/json' \
  -d '{"email":"...","password":"..."}' | jq -r .token)" \
  pytest integration -v -m api
```

Each test starts session-scoped via the `api_jwt` fixture: it prefers `$API_JWT`; otherwise it signs in once at session start and reuses the token across tests.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures (auth, browser, api_client)
├── pyproject.toml           # Pytest config + marker definitions
├── .env.example             # Environment variable template
├── .env                     # Your local configuration (not in git)
│
├── e2e/                     # Playwright UI tests
│   ├── __init__.py
│   └── test_page_accessibility.py
│
├── integration/             # httpx-based API tests
│   ├── __init__.py
│   ├── test_chat_completions.py    # regression: open-webui#24553
│   └── test_notes.py               # regression: open-webui#24484
│
├── utils/                   # Shared helpers
│   ├── __init__.py
│   └── page_utils.py
│
├── database/                # Database integration tests (planned)
└── sso/                     # SSO integration tests (planned)
```

## Page Categories

### Public Pages (No Authentication)
- `/auth` - Login/signup page
- `/error` - Error page

### User Pages (Requires Authentication)
- `/` - Home
- `/playground` - Playground
- `/workspace/*` - Workspace pages (models, prompts, tools, knowledge)
- `/notes/*` - Notes pages

### Admin Pages (Requires Admin Role)
- `/admin` - Admin dashboard
- `/admin/settings/*` - Admin settings pages
- `/admin/users` - User management
- `/admin/functions` - Admin functions

## Test Markers

Markers are the toggle scheme — combine them with `pytest -m "<expr>"`. They're orthogonal axes; a single test can carry several.

| Marker | Axis | Description |
|--------|------|-------------|
| `@pytest.mark.public` | scope | Tests for public pages (no auth) |
| `@pytest.mark.auth_required` | scope | Tests requiring user authentication |
| `@pytest.mark.admin_required` | scope | Tests requiring admin authentication |
| `@pytest.mark.api` | type | API-level test using `httpx` (no browser) |
| `@pytest.mark.regression` | purpose | Pinned to a specific issue / PR fix |
| `@pytest.mark.slow` | cost | Long-running comprehensive scans |

When adding a new marker, register it in `[tool.pytest.ini_options].markers` in `pyproject.toml` — `--strict-markers` is on, so unregistered markers fail collection.

## Writing New Tests

### Basic Page Test (UI / Playwright)

```python
import pytest
from playwright.sync_api import Page

@pytest.mark.auth_required
def test_my_page(authenticated_page: Page):
    """Test a specific page functionality."""
    authenticated_page.goto("/my-page")
    authenticated_page.wait_for_load_state("networkidle")
    
    # Check for expected element
    assert authenticated_page.locator("h1").is_visible()
```

### Basic API Test (`httpx`)

```python
import httpx
import pytest

@pytest.mark.api
@pytest.mark.auth_required
def test_models_endpoint(api_client: httpx.Client):
    """List models for the authenticated user."""
    resp = api_client.get("/api/models")
    assert resp.status_code == 200
    assert "data" in resp.json()
```

The `api_client` fixture (function-scoped, in `conftest.py`) yields an authenticated `httpx.Client` with `base_url` prefilled — just call `.get()` / `.post()` with paths.

### Writing a regression test

Each known-bug regression test follows the same pattern: small, focused, narrowly-asserting. The goal is to fail *only* if the specific bug reappears — not to drift into general feature testing.

```python
@pytest.mark.api
@pytest.mark.auth_required
@pytest.mark.regression
def test_something_does_not_regress(api_client):
    """Regression for open-webui/open-webui#NNNNN.

    In version X.Y.Z, doing <thing> returned HTTP <code> with
    <error message>. Fixed in PR #MMMM by <one-line summary>.
    """
    resp = api_client.post("/api/...", json={...})

    # The specific symptom — assert it can't come back.
    if resp.status_code == 400:
        detail = resp.json().get("detail", "")
        assert "<the exact substring from the original crash>" not in detail, (
            f"Regression of #NNNNN: {detail!r}"
        )
```

Conventions:
- File name: `integration/test_<endpoint-group>.py` (collect related issues in one file).
- Docstring: lead with `Regression for open-webui/open-webui#NNNNN.` followed by the before/after symptom. Future-you will thank you.
- Substring assertions over exact-match — provider error wording drifts.
- If the test creates state (notes, chats, files), wrap in `try / finally` and delete in `finally` so re-runs don't leak.

### Using Test Utilities

```python
from utils import PageCheckResult, create_page_visitor

def test_multiple_pages(page: Page):
    visitor = create_page_visitor(page)
    
    results = []
    for path, desc in [("/page1", "Page 1"), ("/page2", "Page 2")]:
        result = visitor(path, desc)
        results.append(result)
    
    failed = [r for r in results if not r.status == PageStatus.PASSED]
    assert not failed, f"Pages failed: {[r.path for r in failed]}"
```

## Troubleshooting

### API tests skip with "Could not reach Open WebUI for signin"

The `api_jwt` fixture couldn't sign in to mint a token. Either:
- Start Open WebUI at `$OPEN_WEBUI_URL`, or
- Export `$API_JWT` directly (see [Getting an `API_JWT`](#getting-an-api_jwt))

### API tests skip with `Signin failed: HTTP 4xx`

Credentials in `.env` don't match a real user, or password auth is disabled on that instance. Mint a JWT manually via UI/browser devtools and export it as `API_JWT`.

### Browser tests skip with "Could not authenticate"

1. Ensure Open WebUI is running at the URL in `.env`
2. Verify test user credentials are correct
3. Check that the users exist in Open WebUI

### Timeout errors

Increase timeouts in `.env`:

```bash
DEFAULT_TIMEOUT=60000
NAVIGATION_TIMEOUT=120000
```

### Browser not found

Run Playwright browser installation:

```bash
playwright install chromium
```

### See what's happening in the browser

```bash
HEADLESS=false SLOW_MO=500 pytest e2e/test_page_accessibility.py -v -k "test_auth"
```

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Page Accessibility Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      open-webui:
        image: ghcr.io/open-webui/open-webui:main
        ports:
          - 8080:8080
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        working-directory: tests
        run: |
          pip install "pytest~=8.4.1" "pytest-asyncio>=0.23.0" \
                      "pytest-playwright>=0.5.0" "playwright==1.49.1" \
                      "httpx==0.28.1" "python-dotenv>=1.0.0" \
                      "allure-pytest>=2.13.0" "pytest-html>=4.0.0"
          playwright install chromium
      
      - name: Run tests
        working-directory: tests
        env:
          OPEN_WEBUI_URL: http://localhost:8080
          TEST_USER_EMAIL: test@example.com
          TEST_USER_PASSWORD: testpassword
        run: pytest --alluredir=allure-results
      
      - name: Run API regression tests only (fast)
        working-directory: tests
        env:
          OPEN_WEBUI_URL: http://localhost:8080
          API_JWT: ${{ secrets.OPENWEBUI_TEST_JWT }}
        run: pytest -m "api and regression"
      
      - name: Upload Allure results
        uses: actions/upload-artifact@v4
        with:
          name: allure-results
          path: tests/allure-results
```

## Future Development

- [x] API endpoint tests (initial regression coverage)
- [ ] Database integration tests (PostgreSQL, MySQL)
- [ ] SSO integration tests (OAuth, OIDC, LDAP)
- [ ] Performance benchmarking
- [ ] Accessibility compliance (WCAG)
- [ ] Allure dashboard hosting
- [ ] Fix `pip install -e .` (configure `[tool.setuptools]` packages or switch to a `src/` layout)

## License

MIT License - See LICENSE file for details.
