#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Run one step with its output appended to testing/live.log (what `ui/savante.py --mode view` shows):
#   testing/live.sh "title" cargo test --release -- --ignored bench_x --nocapture --test-threads=1
set -uo pipefail
cd "$(dirname "$0")/.."
title=$1; shift
{ echo "## $title — $(date -u +%H:%M:%SZ)"; "$@" 2>&1 | grep -vE '^(running|$)|Compiling|Finished|Running|^test result: ok. 0'; echo "## done: $title (exit ${PIPESTATUS[0]})"; } | tee -a testing/live.log
