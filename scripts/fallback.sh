#!/usr/bin/env bash
# Setup steps that fetch from outside services: every attempt is bounded and retried, and the retry
# moves to another source where there is one. Steps source it (`. tests/scripts/fallback.sh`) and
# call its functions; `bash fallback.sh <function> [args]` runs one.

apt_sources=/etc/apt/sources.list.d/ubuntu.sources

# retry <minutes per attempt> <command...>: three attempts, 15 s and then 60 s apart.
retry() {
  local minutes=$1 attempt
  shift
  for attempt in 1 2 3; do
    timeout --kill-after=30s "${minutes}m" "$@" && return 0
    echo "::warning::Attempt $attempt of '$*' failed"
    [ "$attempt" = 3 ] || sleep $((attempt * attempt * 15))
  done
  return 1
}

# apt_mirror [failed mirror]: points apt at a single mirror, the first from the runner's list (then
# archive.ubuntu.com) after the failed one whose update completes in two minutes. With the list
# itself, apt falls back to the next mirror when one fails, and on 2026-11-23 that fallback hung
# apt-get for over an hour in a tenth of the jobs that hit it.
apt_mirror() {
  local failed=${1:-} mirror
  if [ ! -f "$RUNNER_TEMP/ubuntu.sources" ]; then
    sudo cp "$apt_sources" "$RUNNER_TEMP/ubuntu.sources"
    printf 'Acquire::Retries "3";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\nDPkg::Lock::Timeout "60";\n' | sudo tee /etc/apt/apt.conf.d/99-timeouts > /dev/null
  fi
  for mirror in $(awk '{print $1}' /etc/apt/apt-mirrors.txt) http://archive.ubuntu.com/ubuntu/; do
    if [ -n "$failed" ]; then
      [ "$mirror" = "$failed" ] && failed=
      continue
    fi
    sed "s|mirror+file:/etc/apt/apt-mirrors.txt|$mirror|" "$RUNNER_TEMP/ubuntu.sources" | sudo tee "$apt_sources" > /dev/null
    if sudo timeout 120 apt-get update --error-on=any; then
      echo "$mirror" > "$RUNNER_TEMP/apt-mirror"
      echo "apt uses $mirror"
      return 0
    fi
  done
  return 1
}

# apt_install <package...>: on a failed attempt apt moves to the next mirror, or retries the last one.
apt_install() {
  local attempt
  for attempt in 1 2 3; do
    sudo timeout --kill-after=30s 5m apt-get install -y "$@" && return 0
    echo "::warning::Attempt $attempt of installing $* failed"
    [ "$attempt" = 3 ] || apt_mirror "$(cat "$RUNNER_TEMP/apt-mirror")" || sleep 30
  done
  return 1
}

# install_browser: Playwright's Chromium with its system libraries, which it installs from the mirror
# apt_mirror picked. Its own apt-get cannot change mirrors, so the retry installs the libraries
# through apt_install and then the browser alone.
install_browser() {
  timeout --kill-after=30s 5m python -m playwright install --with-deps chromium && return 0
  echo "::warning::Installing the browser with its libraries failed"
  # Playwright's apt-get runs under sudo, which outlives the timeout and holds the dpkg lock.
  sudo pkill -x apt-get || true
  apt_install --no-install-recommends $(python -m playwright install-deps --dry-run chromium | awk '/^  /{print $1}') &&
    retry 5 python -m playwright install chromium
}

# docker_build <docker build args...>: the last attempt pulls Docker Hub images through mirror.gcr.io.
docker_build() {
  local attempt
  for attempt in 1 2; do
    timeout --kill-after=30s 10m docker build "$@" && return 0
    echo "::warning::Attempt $attempt of 'docker build $*' failed"
    sleep 15
  done
  { cat /etc/docker/daemon.json 2> /dev/null || echo '{}'; } |
    jq '."registry-mirrors" = ["https://mirror.gcr.io"]' > "$RUNNER_TEMP/daemon.json"
  sudo install -D -m 644 "$RUNNER_TEMP/daemon.json" /etc/docker/daemon.json
  sudo systemctl restart docker
  timeout --kill-after=30s 10m docker build "$@"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -euo pipefail
  "$@"
fi
