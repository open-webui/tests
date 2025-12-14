#!/usr/bin/env bash
#
# Open WebUI Test Runner
#
# This script sets up a complete test environment:
# 1. Creates a Python virtual environment
# 2. Installs Open WebUI and test dependencies
# 3. Starts the Open WebUI server
# 4. Creates test users (admin + regular user)
# 5. Runs the test suite
# 6. Cleans up on exit
#
# Usage:
#   ./test.sh              # Run all tests with local Open WebUI instance
#   ./test.sh -k "admin"   # Pass arguments to pytest
#
# To test against an external instance instead:
#   OPEN_WEBUI_URL=http://localhost:8080 pytest
#
set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.test-venv"
DATA_DIR="${SCRIPT_DIR}/.test-data"
LOG_FILE="${SCRIPT_DIR}/.test-server.log"

# Server configuration
export WEBUI_SECRET_KEY="test-secret-key-for-testing-only"
export DATA_DIR="${DATA_DIR}"
export WEBUI_AUTH="true"
export ENABLE_SIGNUP="true"
export PORT="${TEST_PORT:-8081}"
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
NC='\033[0m' # No Color

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

cleanup() {
    log_info "Cleaning up..."
    
    # Kill the server if running
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log_info "Stopping Open WebUI server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    
    # Optionally clean up test data (uncomment to enable)
    # if [[ -d "$DATA_DIR" ]]; then
    #     log_info "Removing test data directory..."
    #     rm -rf "$DATA_DIR"
    # fi
    
    log_info "Cleanup complete"
}

trap cleanup EXIT

wait_for_server() {
    local max_attempts=60
    local attempt=1
    
    log_info "Waiting for Open WebUI to be ready..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s --connect-timeout 2 "${OPEN_WEBUI_URL}/api/version" > /dev/null 2>&1; then
            log_info "Open WebUI is ready!"
            return 0
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
    
    # Step 1: Create/activate virtual environment for Open WebUI
    if [[ ! -d "$VENV_DIR" ]]; then
        log_info "Creating virtual environment for Open WebUI..."
        python3 -m venv "$VENV_DIR"
    else
        log_info "Using existing virtual environment"
    fi
    
    # Activate the venv
    source "${VENV_DIR}/bin/activate"
    
    # Step 2: Install Open WebUI
    log_info "Installing/updating Open WebUI..."
    pip install --quiet --upgrade pip
    pip install --quiet open-webui
    
    # Step 3: Create fresh data directory
    if [[ -d "$DATA_DIR" ]]; then
        log_info "Removing old test data..."
        rm -rf "$DATA_DIR"
    fi
    mkdir -p "$DATA_DIR"
    
    # Step 4: Start the server
    log_info "Starting Open WebUI on port ${PORT}..."
    open-webui serve --port "$PORT" > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    
    # Give it a moment to potentially fail fast
    sleep 2
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log_error "Server failed to start. Check log: $LOG_FILE"
        tail -20 "$LOG_FILE"
        exit 1
    fi
    
    # Wait for server to be ready
    if ! wait_for_server; then
        log_error "Server startup failed"
        tail -50 "$LOG_FILE"
        exit 1
    fi
    
    # Step 5: Create test users
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
    
    # Step 6: Ensure test dependencies are installed
    log_info "Checking test dependencies..."
    if [[ -f "${SCRIPT_DIR}/pyproject.toml" ]]; then
        pip install --quiet -e "${SCRIPT_DIR}[dev]" 2>/dev/null || \
        pip install --quiet pytest playwright python-dotenv pytest-html pytest-metadata
    fi
    
    # Ensure playwright browsers are installed
    playwright install chromium --with-deps 2>/dev/null || \
    python -m playwright install chromium 2>/dev/null || true
    
    # Step 7: Run tests
    log_info "Running tests..."
    log_info "Server URL: ${OPEN_WEBUI_URL}"
    echo ""
    
    # Export credentials for the test suite
    export ADMIN_USER_EMAIL="$ADMIN_EMAIL"
    export ADMIN_USER_PASSWORD="$ADMIN_PASSWORD"
    export TEST_USER_EMAIL="$TEST_EMAIL"
    export TEST_USER_PASSWORD="$TEST_PASSWORD"
    
    # Run pytest with any additional arguments passed to this script
    cd "$SCRIPT_DIR"
    pytest "${@:---verbose}"
    
    log_info "Tests complete!"
}

# Run main function with all script arguments
main "$@"
