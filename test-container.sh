#!/usr/bin/env bash
#
# Open WebUI Container Test Runner
#
# This script tests against a Docker container image:
# 1. Pulls the specified image (default: ghcr.io/open-webui/open-webui:dev)
# 2. Starts a container on port 8082
# 3. Creates test users (admin + regular user)
# 4. Runs the test suite
# 5. Cleans up on exit
#
# Usage:
#   ./test-container.sh                         # Test against :dev image (default)
#   ./test-container.sh --image :main           # Test against :main image
#   ./test-container.sh --image :latest         # Test against :latest image
#   ./test-container.sh --image myregistry/img  # Test against custom image
#   ./test-container.sh -k "admin"              # Pass arguments to pytest
#
# Requirements:
#   - Docker installed and running
#   - Python 3.11+ (for test dependencies)
#
set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.test-venv-container"
CONTAINER_NAME="openwebui-test-container"
DEFAULT_IMAGE="ghcr.io/open-webui/open-webui:dev"
DOCKER_IMAGE="${DEFAULT_IMAGE}"

# Server configuration
PORT="${TEST_PORT:-8082}"
OPEN_WEBUI_URL="http://localhost:${PORT}"

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

log_docker() {
    echo -e "${BLUE}[DOCKER]${NC} $1"
}

cleanup() {
    log_info "Cleaning up..."
    
    # Stop and remove the container if it exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_docker "Stopping container: ${CONTAINER_NAME}..."
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
    fi
    
    # Remove the virtual environment if --clean was specified
    if [[ "${CLEAN_ALL:-}" == "true" ]] && [[ -d "$VENV_DIR" ]]; then
        log_info "Removing virtual environment..."
        rm -rf "$VENV_DIR"
    fi
    
    log_info "Cleanup complete"
}

trap cleanup EXIT

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
}

wait_for_server() {
    local max_attempts=120  # Container startup can be slower
    local attempt=1
    
    log_info "Waiting for Open WebUI to be ready..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s --connect-timeout 2 "${OPEN_WEBUI_URL}/api/version" > /dev/null 2>&1; then
            local version
            version=$(curl -s "${OPEN_WEBUI_URL}/api/version" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
            log_info "Open WebUI is ready! (version: ${version})"
            return 0
        fi
        
        if [[ $((attempt % 10)) -eq 0 ]]; then
            log_info "Still waiting... (attempt $attempt/$max_attempts)"
        fi
        
        sleep 1
        ((attempt++))
    done
    
    log_error "Open WebUI failed to start within $max_attempts seconds"
    log_error "Container logs:"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -50
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

show_help() {
    echo "Usage: $0 [OPTIONS] [-- PYTEST_ARGS]"
    echo ""
    echo "Options:"
    echo "  --image IMAGE    Docker image to test (default: ghcr.io/open-webui/open-webui:dev)"
    echo "                   Can use :tag shorthand for ghcr.io/open-webui/open-webui:tag"
    echo "  --clean          Remove virtual environment after testing"
    echo "  --help, -h       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                              # Test against :dev image (default)"
    echo "  $0 --image :main                # Test against :main image"
    echo "  $0 --image :latest              # Test against :latest image"
    echo "  $0 --image myregistry/myimage   # Test against custom image"
    echo "  $0 --clean                      # Clean up venv after tests"
    echo "  $0 -- -k 'admin'                # Pass args to pytest"
    echo "  $0 --image :main -- -v          # Custom image + pytest args"
}

main() {
    # Parse arguments
    local pytest_args=()
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --image)
                if [[ -n "${2:-}" ]]; then
                    if [[ "$2" == :* ]]; then
                        # Shorthand: :tag -> ghcr.io/open-webui/open-webui:tag
                        DOCKER_IMAGE="ghcr.io/open-webui/open-webui${2}"
                    else
                        DOCKER_IMAGE="$2"
                    fi
                    shift 2
                else
                    log_error "--image requires an argument"
                    exit 1
                fi
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            --clean)
                CLEAN_ALL="true"
                shift
                ;;
            --)
                shift
                pytest_args=("$@")
                break
                ;;
            *)
                # Pass through to pytest
                pytest_args+=("$1")
                shift
                ;;
        esac
    done

    log_info "Open WebUI Container Test Runner"
    log_info "================================="
    
    # Step 0: Check Docker is available
    check_docker
    
    # Step 1: Stop any existing test container
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_docker "Removing existing container: ${CONTAINER_NAME}..."
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
    fi
    
    # Step 2: Pull the latest :dev image (always pull, don't use cache)
    log_docker "Pulling latest image: ${DOCKER_IMAGE}..."
    docker pull "$DOCKER_IMAGE"
    
    # Step 3: Start the container
    log_docker "Starting container on port ${PORT}..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        -p "${PORT}:8080" \
        -e WEBUI_AUTH=true \
        -e ENABLE_SIGNUP=true \
        -e WEBUI_SECRET_KEY="test-secret-key-for-dev-testing" \
        "$DOCKER_IMAGE"
    
    # Verify container is running
    sleep 2
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_error "Container failed to start"
        docker logs "$CONTAINER_NAME" 2>&1 | tail -20
        exit 1
    fi
    
    log_docker "Container started: ${CONTAINER_NAME}"
    
    # Step 4: Wait for server to be ready
    if ! wait_for_server; then
        log_error "Server startup failed"
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
    
    # Create venv if it doesn't exist
    if [[ ! -d "$VENV_DIR" ]]; then
        log_info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
    
    # Activate venv
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    
    # Install test dependencies
    log_info "Installing test dependencies..."
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
    
    # Step 7: Run tests
    log_info "Running tests..."
    log_info "Server URL: ${OPEN_WEBUI_URL}"
    log_docker "Image: ${DOCKER_IMAGE}"
    echo ""
    
    # Export credentials for the test suite
    export OPEN_WEBUI_URL
    export ADMIN_USER_EMAIL="$ADMIN_EMAIL"
    export ADMIN_USER_PASSWORD="$ADMIN_PASSWORD"
    export TEST_USER_EMAIL="$TEST_EMAIL"
    export TEST_USER_PASSWORD="$TEST_PASSWORD"
    
    # Run pytest with any additional arguments passed to this script
    cd "$SCRIPT_DIR"
    if [[ ${#pytest_args[@]} -gt 0 ]]; then
        pytest "${pytest_args[@]}"
    else
        pytest --verbose
    fi
    
    log_info "Tests complete!"
}

# Run main function with all script arguments
main "$@"
