#!/usr/bin/env bash
# The release gate: every check that decides whether a version ships, with the output kept as its record.
# Offline checks always; the oracles, A/Bs and whole-model budgets when the models and b11192 are present.
#   BANKML_GGML_LIB=/path/to/llama-b11192 testing/release_gate.sh      → testing/results/<version>.txt
# Everything it prints is also appended to testing/live.log, which `ui/savante.py --mode view` shows live.
# A kernel change counts only if every oracle still passes: speed without the same bits is not a result.
set -euo pipefail
cd "$(dirname "$0")/.."
v=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
mkdir -p testing/results
out=testing/results/$v.txt
{
  echo "# bankml $v release gate — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# $(rustc --version) · $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //') · $(nproc) threads"
  cargo build --release --locked -q
  cargo test --release --locked 2>&1 | awk '/Running/{r=$2} /^test result/ && !/ 0 passed; 0 failed; 0 ignored/{print r": "$0}'
  cargo clippy --release --all-targets --locked -q -- -D warnings && echo "clippy: clean"
  python3 testing/test_gguf_guard.py | tail -1
  python3 testing/guard_agree.py target/release/bankml $(ls .models/*.gguf 2>/dev/null) | tail -1
  if [ -n "${BANKML_GGML_LIB:-}" ]; then
    for t in oracle_ggml_b11192_real_bonsai_1_7b oracle_ggml_b11192_real_bonsai_8b_q1_0 oracle_ggml_b11192_real_ternary_bonsai_8b \
             ab_vs_ggml ab_vs_ggml_q2_0 bench_q1_0_prefill_act bench_memory_floor decode_budget_q1_0 decode_budget_q2_0; do
      echo "## $t"
      name=$(cargo test --release -q -- --list --ignored 2>/dev/null | sed -n "s/^\(.*::$t\): test$/\1/p")
      log=$(cargo test --release --locked -q -- --ignored --exact "$name" --nocapture --test-threads=1 2>&1) || { echo "$log"; echo "FAILED: $t"; exit 1; }
      echo "$log" | grep -vE '^(running|$)|^test result: ok. 0' || true
    done
  else
    echo "BANKML_GGML_LIB unset: oracles, A/B and budgets skipped"
  fi
  echo "# gate done"
} 2>&1 | tee "$out" | tee -a testing/live.log
