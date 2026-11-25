#!/bin/bash
# The pytest arguments of one regression job: the selected tests of its suite when the run selects
# some (SELECT, the workflow's select input), else the test modules of this shard of its suite.
# usage: select_tests.sh <suite directory> <shards> <shard> <durations file>
suite=$1
if [ -z "$SELECT" ]; then
    exec python3 scripts/shard.py "$@"
fi
selected=()
for path in $SELECT; do
    case $path in
        "$suite"/*) selected+=("$path") ;;
    esac
done
echo "${selected[*]}"
