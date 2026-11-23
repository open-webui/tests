#!/bin/bash
# The pytest arguments of one regression job: the selected tests of its suite when the run selects
# some (SELECT, the workflow's select input), else its whole suite and this shard of it.
# usage: select_tests.sh <suite directory> "<shard arguments>"
suite=$1
shard=$2
if [ -z "$SELECT" ]; then
    echo "$suite $shard --splitting-algorithm duration_based_chunks"
    exit 0
fi
selected=()
for path in $SELECT; do
    case $path in
        "$suite"/*) selected+=("$path") ;;
    esac
done
echo "${selected[*]}"
