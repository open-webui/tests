#!/usr/bin/env bash
# Points apt at a single mirror from the runner's mirror list: the first whose update completes in two
# minutes. With the list itself, apt falls back to the next mirror when one fails, and on 2026-11-23
# that fallback hung apt-get for over an hour in a tenth of the jobs that hit it.
set -euo pipefail

sources=/etc/apt/sources.list.d/ubuntu.sources
sudo cp "$sources" "$RUNNER_TEMP/ubuntu.sources"
printf 'Acquire::Retries "3";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' | sudo tee /etc/apt/apt.conf.d/99-timeouts > /dev/null
for mirror in $(awk '{print $1}' /etc/apt/apt-mirrors.txt) http://archive.ubuntu.com/ubuntu/; do
  sed "s|mirror+file:/etc/apt/apt-mirrors.txt|$mirror|" "$RUNNER_TEMP/ubuntu.sources" | sudo tee "$sources" > /dev/null
  if sudo timeout 120 apt-get update --error-on=any; then
    echo "apt uses $mirror"
    exit 0
  fi
done
exit 1
