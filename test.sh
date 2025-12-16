#!/usr/bin/env bash
#
# Open WebUI Test Runner
#
# This script tests against a cloned Open WebUI repository:
# 1. Clones the Open WebUI repository (or uses existing clone)
# 2. Checks out the specified branch (default: dev)
# 3. Builds the frontend (npm run build)
# 4. Installs backend dependencies
# 5. Starts the server with uvicorn
# 6. Creates test users (admin + regular user)
# 7. Runs the test suite
# 8. Cleans up on exit
#
# Usage:
#   ./test.sh                     # Test against dev branch (default)
#   ./test.sh --branch main       # Test against main branch
#   ./test.sh --branch feature/x  # Test against a feature branch
#   ./test.sh -- -k "admin"       # Pass arguments to pytest
#
# Requirements:
#   - Git installed
#   - Python 3.11+ with pip
#   - Node.js 18+ with npm (for frontend build)
#
set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.test-repo"
VENV_DIR="${SCRIPT_DIR}/.test-venv"
DATA_DIR="${SCRIPT_DIR}/.test-data"
LOG_FILE="${SCRIPT_DIR}/.test-server.log"

# Git repository
REPO_URL="https://github.com/open-webui/open-webui.git"
DEFAULT_BRANCH="dev"
BRANCH="${DEFAULT_BRANCH}"

# Server configuration
export WEBUI_SECRET_KEY="test-secret-key-for-dev-testing"
export DATA_DIR="${DATA_DIR}"
export WEBUI_AUTH="true"
export ENABLE_SIGNUP="true"
export PORT="${TEST_PORT:-8083}"
export OPEN_WEBUI_URL="http://localhost:${PORT}"

# Test user credentials (must match .env or conftest.py defaults)
ADMIN_EMAIL="${ADMIN_USER_EMAIL:-admin@example.com}"
ADMIN_PASSWORD="${ADMIN_USER_PASSWORD:-adminpassword123}"
ADMIN_NAME="Admin User"

TEST_EMAIL="${TEST_USER_EMAIL:-test@example.com}"
TEST_PASSWORD="${TEST_USER_PASSWORD:-testpassword123}"
TEST_NAME="Test User"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Argument Parsing
# =============================================================================

show_help() {
    echo "Usage: $0 [OPTIONS] [-- PYTEST_ARGS]"
    echo ""
    echo "Options:"
    echo "  --branch, -b BRANCH  Git branch to test (default: dev)"
    echo "  --fresh              Force fresh clone (removes existing repo)"
    echo "  --help, -h           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                         # Test against dev branch (default)"
    echo "  $0 --branch main           # Test against main branch"
    echo "  $0 --branch feature/xyz    # Test against a feature branch"
    echo "  $0 --fresh                 # Force fresh clone"
    echo "  $0 --fresh --branch main   # Fresh clone of main branch"
    echo "  $0 -- -k 'admin'           # Pass args to pytest"
    echo "  $0 --branch main -- -v     # Custom branch + pytest args"
}

PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --branch|-b)
            BRANCH="$2"
            shift 2
            ;;
        --fresh)
            # Force fresh clone
            FRESH_CLONE="true"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        --)
            shift
            PYTEST_ARGS=("$@")
            break
            ;;
        *)
            # Pass remaining args to pytest
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_git() {
    echo -e "${BLUE}[GIT]${NC} $1"
}

cleanup() {
    log_info "Cleaning up..."
    
    # Kill the server if running
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log_info "Stopping Open WebUI server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    
    # Deactivate virtual environment if active
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        deactivate 2>/dev/null || true
    fi
    
    # Remove the virtual environment (fresh on every run)
    if [[ -d "$VENV_DIR" ]]; then
        log_info "Removing virtual environment..."
        rm -rf "$VENV_DIR"
    fi
    
    # Remove the data directory (fresh on every run)
    if [[ -d "$DATA_DIR" ]]; then
        log_info "Removing data directory..."
        rm -rf "$DATA_DIR"
    fi
    
    log_info "Cleanup complete"
}

trap cleanup EXIT

check_git() {
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed. Please install Git first."
        exit 1
    fi
}

wait_for_server() {
    local max_attempts=120
    local attempt=1
    
    log_info "Waiting for Open WebUI to be ready..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s --connect-timeout 2 "${OPEN_WEBUI_URL}/api/version" > /dev/null 2>&1; then
            local version
            version=$(curl -s "${OPEN_WEBUI_URL}/api/version" | grep -o '"version":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
            log_info "Open WebUI is ready! (version: $version)"
            return 0
        fi
        
        # Check if server process is still running
        if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log_error "Server process died unexpectedly"
            log_error "Check the log file: $LOG_FILE"
            return 1
        fi
        
        if [[ $((attempt % 10)) -eq 0 ]]; then
            log_info "Still waiting... (attempt $attempt/$max_attempts)"
        fi
        
        sleep 1
        ((attempt++))
    done
    
    log_error "Open WebUI failed to start within $max_attempts seconds"
    log_error "Check the log file: $LOG_FILE"
    return 1
}

create_user() {
    local email="$1"
    local password="$2"
    local name="$3"
    local is_admin_signup="${4:-false}"
    local admin_token="${5:-}"
    
    local response
    
    if [[ "$is_admin_signup" == "true" ]]; then
        # First user signup (becomes admin automatically)
        response=$(curl -s -X POST "${OPEN_WEBUI_URL}/api/v1/auths/signup" \
            -H "Content-Type: application/json" \
            -d "{\"email\": \"${email}\", \"password\": \"${password}\", \"name\": \"${name}\"}" \
            2>&1)
    else
        # Use admin API to add subsequent users
        response=$(curl -s -X POST "${OPEN_WEBUI_URL}/api/v1/auths/add" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer ${admin_token}" \
            -d "{\"email\": \"${email}\", \"password\": \"${password}\", \"name\": \"${name}\", \"role\": \"user\"}" \
            2>&1)
    fi
    
    # Check if signup was successful
    if echo "$response" | grep -q '"id"'; then
        if [[ "$is_admin_signup" == "true" ]]; then
            log_info "Created admin user: $email" >&2
            # Extract and return the token to stdout (logs go to stderr)
            echo "$response" | grep -o '"token":"[^"]*"' | cut -d'"' -f4
        else
            log_info "Created test user: $email" >&2
        fi
        return 0
    elif echo "$response" | grep -q "EMAIL_TAKEN\|already exists\|Email already"; then
        log_warn "User already exists: $email" >&2
        return 0
    else
        log_error "Failed to create user: $email" >&2
        log_error "Response: $response" >&2
        return 1
    fi
}

get_admin_token() {
    local email="$1"
    local password="$2"
    
    local response
    response=$(curl -s -X POST "${OPEN_WEBUI_URL}/api/v1/auths/signin" \
        -H "Content-Type: application/json" \
        -d "{\"email\": \"${email}\", \"password\": \"${password}\"}" \
        2>&1)
    
    if echo "$response" | grep -q '"token"'; then
        echo "$response" | grep -o '"token":"[^"]*"' | cut -d'"' -f4
        return 0
    else
        log_error "Failed to get admin token" >&2
        log_error "Response: $response" >&2
        return 1
    fi
}

# =============================================================================
# Main Script
# =============================================================================

main() {
    log_info "Open WebUI Test Runner"
    log_info "======================"
    log_info "Branch: $BRANCH"
    
    # Step 0: Check prerequisites
    check_git
    
    # Step 1: Clone or update the repository
    if [[ "${FRESH_CLONE:-}" == "true" ]] && [[ -d "$REPO_DIR" ]]; then
        log_git "Removing existing repository for fresh clone..."
        rm -rf "$REPO_DIR"
    fi
    
    if [[ ! -d "$REPO_DIR" ]]; then
        log_git "Cloning Open WebUI repository..."
        git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
    else
        log_git "Updating existing repository..."
        cd "$REPO_DIR"
        
        # Reset any local changes to allow clean branch switching
        git reset --hard HEAD 2>/dev/null || true
        git clean -fd 2>/dev/null || true
        
        # Get current branch
        CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
        
        if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
            log_git "Switching from $CURRENT_BRANCH to $BRANCH..."
            # Fetch all branches (unshallow if needed) to enable branch switching
            git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
            git fetch --unshallow origin 2>/dev/null || git fetch origin
            git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"
        fi
        
        git pull origin "$BRANCH"
        cd "$SCRIPT_DIR"
    fi
    
    log_git "Repository ready at: $REPO_DIR"
    
    # Step 2: Create fresh virtual environment (always clean)
    if [[ -d "$VENV_DIR" ]]; then
        log_info "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    fi
    
    log_info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    
    # Activate the venv
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    
    # Step 3: Build the frontend (required for serving pages)
    log_info "Building frontend (this may take a minute)..."
    cd "$REPO_DIR"
    
    # Check if node/npm is available and version is sufficient
    if ! command -v npm &> /dev/null; then
        log_error "npm is not installed. Please install Node.js 18+ first."
        exit 1
    fi
    
    NODE_VERSION=$(node --version | sed 's/v//' | cut -d. -f1)
    if [[ "$NODE_VERSION" -lt 18 ]]; then
        log_error "Node.js 18+ is required, but found v${NODE_VERSION}."
        log_error "Please upgrade Node.js: https://nodejs.org/"
        log_error "Or use nvm: nvm install 22 && nvm use 22"
        exit 1
    fi
    
    # Install npm dependencies and build (use --force like the Dockerfile does)
    npm ci --force 2>&1 | tail -5 || npm install --force 2>&1 | tail -5
    npm run build
    
    # Step 4: Install Open WebUI backend dependencies
    log_info "Installing Open WebUI backend dependencies..."
    cd "$REPO_DIR/backend"
    
    # Install the backend dependencies
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt 2>/dev/null || true
    pip install --quiet -e . 2>/dev/null || pip install --quiet . 2>/dev/null || true
    
    # Step 5: Create fresh data directory (clean slate for each run)
    if [[ -d "$DATA_DIR" ]]; then
        log_info "Removing existing data directory..."
        rm -rf "$DATA_DIR"
    fi
    mkdir -p "$DATA_DIR"
    
    # Step 6: Start the server using dev.sh pattern (uvicorn directly)
    log_info "Starting Open WebUI server on port $PORT..."
    
    # Set up environment for the server
    export CORS_ALLOW_ORIGIN="http://localhost:5173;http://localhost:8080;http://localhost:${PORT}"
    
    # Start uvicorn in the background
    uvicorn open_webui.main:app --port "$PORT" --host 0.0.0.0 --forwarded-allow-ips '*' > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    
    cd "$SCRIPT_DIR"
    
    # Step 7: Wait for server to be ready
    if ! wait_for_server; then
        log_error "Server startup failed"
        exit 1
    fi
    
    # Step 8: Create test users
    log_info "Creating test users..."
    
    # First user signup becomes admin - capture the token
    ADMIN_TOKEN=$(create_user "$ADMIN_EMAIL" "$ADMIN_PASSWORD" "$ADMIN_NAME" "true")
    if [[ $? -ne 0 ]] || [[ -z "$ADMIN_TOKEN" ]]; then
        # User might already exist - try to get token via signin
        log_info "Admin user may already exist, attempting signin..."
        ADMIN_TOKEN=$(get_admin_token "$ADMIN_EMAIL" "$ADMIN_PASSWORD")
        if [[ $? -ne 0 ]] || [[ -z "$ADMIN_TOKEN" ]]; then
            log_error "Failed to create or authenticate admin user"
            exit 1
        fi
    fi
    
    # Use admin API to create the regular test user
    if ! create_user "$TEST_EMAIL" "$TEST_PASSWORD" "$TEST_NAME" "false" "$ADMIN_TOKEN"; then
        log_error "Failed to create test user"
        exit 1
    fi
    
    # Step 9: Install test dependencies
    log_info "Installing test dependencies..."
    cd "$SCRIPT_DIR"
    
    if [[ -f "${SCRIPT_DIR}/pyproject.toml" ]]; then
        pip install --quiet -e "${SCRIPT_DIR}[dev]" || {
            log_info "pyproject.toml install failed, using fallback..."
            pip install --quiet pytest playwright python-dotenv pytest-html pytest-metadata
        }
    else
        pip install --quiet pytest playwright python-dotenv pytest-html pytest-metadata
    fi
    
    # Ensure playwright browsers are installed
    log_info "Ensuring Playwright browsers are installed..."
    playwright install chromium --with-deps 2>/dev/null || \
    python -m playwright install chromium 2>/dev/null || true
    
    # Step 10: Run tests
    log_info "Running tests..."
    log_info "Server URL: ${OPEN_WEBUI_URL}"
    log_git "Branch: ${BRANCH}"
    echo ""
    
    # Export credentials for the test suite
    export ADMIN_USER_EMAIL="$ADMIN_EMAIL"
    export ADMIN_USER_PASSWORD="$ADMIN_PASSWORD"
    export TEST_USER_EMAIL="$TEST_EMAIL"
    export TEST_USER_PASSWORD="$TEST_PASSWORD"
    
    # Run pytest with any additional arguments
    if [[ ${#PYTEST_ARGS[@]} -gt 0 ]]; then
        pytest "${PYTEST_ARGS[@]}"
    else
        pytest
    fi
    
    TEST_EXIT_CODE=$?
    
    log_info "Tests complete!"
    exit $TEST_EXIT_CODE
}

# Run main function
main
