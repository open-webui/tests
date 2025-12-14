# Open WebUI Test Suite

External test suite for [Open WebUI](https://github.com/open-webui/open-webui) that provides:

- **Page Accessibility Tests**: Automated testing to verify all pages load correctly
- **Integration Tests**: SSO, external databases, and other integrations (planned)
- **Reporting Dashboard**: Allure reports for test result visualization (planned)

## Quick Start

The easiest way to run tests is with the included test scripts:

### Test Against Latest Stable (pip)

```bash
cd tests
./test.sh
```

This will:
1. Create a virtual environment and install Open WebUI via pip
2. Start a local Open WebUI server (port 8081)
3. Create test users (admin + regular user)
4. Run the full test suite
5. Clean up when done

### Test Against Dev Branch (Docker)

```bash
cd tests
./test-dev.sh
```

This will:
1. Pull the latest `ghcr.io/open-webui/open-webui:dev` image (always fresh)
2. Start a container (port 8082)
3. Create test users (admin + regular user)
4. Run the full test suite
5. Clean up when done

### Passing Arguments to pytest

Both scripts pass arguments through to pytest:
```bash
./test.sh -k "admin"           # Run only admin tests
./test-dev.sh --html=report.html   # Generate HTML report
./test.sh -x                   # Stop on first failure
```

## Testing Against External Instances

To test against an existing Open WebUI deployment (Docker, Kubernetes, etc.):

```bash
# Set up test environment
python -m venv venv && source venv/bin/activate
pip install -e .
playwright install chromium

# Configure and run
cp .env.example .env
# Edit .env with your URL and credentials
pytest
```

Or inline:
```bash
OPEN_WEBUI_URL=http://your-server:8080 \
ADMIN_USER_EMAIL=admin@example.com \
ADMIN_USER_PASSWORD=yourpassword \
pytest -v
```

## Supported Environments

| Deployment | How to Test |
|------------|-------------|
| **Latest stable (pip)** | `./test.sh` |
| **Dev branch (Docker)** | `./test-dev.sh` |
| **Existing Docker** | `OPEN_WEBUI_URL=http://localhost:3000 pytest` |
| **Kubernetes** | `OPEN_WEBUI_URL=https://your-cluster pytest` |
| **Remote server** | `OPEN_WEBUI_URL=https://your-server pytest` |

## Prerequisites

For `./test.sh`:
- Python 3.11+
- Internet connection (to pip install open-webui)

For `./test-dev.sh`:
- Docker installed and running
- Python 3.11+ (for test dependencies)

For testing external instances:
- Python 3.11+
- A running Open WebUI instance
- Test user accounts created in Open WebUI

## Manual Installation

If you prefer manual setup over `./test.sh`:

1. **Create a virtual environment:**

   ```bash
   cd tests
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -e .
   ```

3. **Install Playwright browsers:**

   ```bash
   playwright install chromium
   ```

4. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env with your Open WebUI URL and test credentials
   ```

## Configuration

Edit `.env` to configure your test environment, following the comment prompts in `.env.example`. 

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Run only public page tests (no auth needed)
pytest -m public

# Run only authenticated user page tests
pytest -m auth_required

# Run only admin page tests
pytest -m admin_required

# Run slow/comprehensive tests
pytest -m slow
```

### Run With Visible Browser

```bash
HEADLESS=false pytest e2e/test_page_accessibility.py -v
```

### Run Specific Test File

```bash
pytest e2e/test_page_accessibility.py -v
```

### Run With HTML Report

```bash
pytest --html=reports/report.html --self-contained-html
```

### Run With Allure Report

```bash
# Run tests and generate Allure results
pytest --alluredir=allure-results

# Generate and open Allure report
allure serve allure-results
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── pyproject.toml           # Project dependencies and pytest config
├── .env.example             # Environment variable template
├── .env                     # Your local configuration (not in git)
│
├── e2e/                     # End-to-end browser tests
│   ├── __init__.py
│   └── test_page_accessibility.py
│
├── utils/                   # Test utilities
│   ├── __init__.py
│   └── page_utils.py
│
├── integration/             # API integration tests (planned)
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

| Marker | Description |
|--------|-------------|
| `@pytest.mark.public` | Tests for public pages |
| `@pytest.mark.auth_required` | Tests requiring user authentication |
| `@pytest.mark.admin_required` | Tests requiring admin authentication |
| `@pytest.mark.slow` | Long-running comprehensive tests |

## Writing New Tests

### Basic Page Test

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

### Tests skip with "Could not authenticate"

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
          pip install -e .
          playwright install chromium
      
      - name: Run tests
        working-directory: tests
        env:
          OPEN_WEBUI_URL: http://localhost:8080
          TEST_USER_EMAIL: test@example.com
          TEST_USER_PASSWORD: testpassword
        run: pytest --alluredir=allure-results
      
      - name: Upload Allure results
        uses: actions/upload-artifact@v4
        with:
          name: allure-results
          path: tests/allure-results
```

## Future Development

- [ ] Database integration tests (PostgreSQL, MySQL)
- [ ] SSO integration tests (OAuth, OIDC, LDAP)
- [ ] API endpoint tests
- [ ] Performance benchmarking
- [ ] Accessibility compliance (WCAG)
- [ ] Allure dashboard hosting

## License

MIT License - See LICENSE file for details.
