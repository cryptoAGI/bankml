#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# The release gate: every check that decides whether a version ships, with the output kept as its record.
# Offline checks always; the oracles, A/Bs and whole-model budgets when the models and b11192 are present.
#   BANKML_GGML_LIB=/path/to/llama-b11192 testing/release_gate.sh      → testing/results/<version>.txt
# Everything it prints is also appended to testing/live.log, which `sAGI/savante.py --mode view` shows live.
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
  # 0.3.2: the C API is a second workspace member (capi/, libbankml): its unit tests, and clippy over both packages
  cargo test --release --locked -p bankml-capi 2>&1 | awk '/^test result/{print "capi (bankml-capi): "$0}'
  cargo clippy --release --workspace --all-targets --locked -q -- -D warnings && echo "clippy: clean"
  python3 testing/spdx_check.py | tail -1
  python3 testing/test_gguf_guard.py | tail -1
  python3 -B testing/test_ui.py | tail -1 | sed 's/^/ui data layer: /'
  python3 -B testing/test_connectors.py | tail -1 | sed 's/^/postgres connector (throwaway cluster): /'
  python3 -B testing/test_chain.py | tail -1 | sed 's/^/iNFT mint and load (throwaway anvil devnet): /'
  BANKML_TEST_CARRIER=1 python3 -B testing/test_models.py | tail -1 | sed 's/^/model importer (loopback source; carrier on spare ports): /'
  python3 testing/guard_agree.py target/release/bankml $(ls .models/*.gguf 2>/dev/null) | tail -1
  # 0.3.2: the C library, and bankml_log (a C-variadic function defined in Rust) against libc snprintf, from C
  cargo build --release --locked -q -p bankml-capi
  python3 -B testing/capi/capi_oracle.py --printf || { echo "FAILED: printf oracle"; exit 1; }
  if [ -n "${BANKML_GGML_LIB:-}" ]; then
    for t in oracle_tokenizer oracle_chat_template oracle_forward_embed_norm oracle_forward_qkv_rope oracle_forward_attention oracle_forward_attention_tiled oracle_forward_attention_split oracle_forward_swiglu_sweep oracle_forward_model oracle_forward_model_ternary oracle_greedy_llama_server oracle_greedy_llama_server_ternary oracle_greedy_llama_server_long oracle_greedy_llama_server_deep oracle_sample_llama_server oracle_native_serve gpu_q1_0_mat_vec_bit_exact oracle_train_script oracle_train_imprint oracle_ggml_b11192_real_bonsai_1_7b oracle_ggml_b11192_real_bonsai_8b_q1_0 oracle_ggml_b11192_real_ternary_bonsai_8b \
             ab_vs_ggml ab_vs_ggml_q2_0 bench_q1_0_prefill_act bench_memory_floor decode_budget_q1_0 decode_budget_q2_0; do
      echo "## $t"
      name=$(cargo test --release -q -- --list --ignored 2>/dev/null | sed -n "s/^\(.*::$t\): test$/\1/p")
      log=$(cargo test --release --locked -q -- --ignored --exact "$name" --nocapture --test-threads=1 2>&1) || { echo "$log"; echo "FAILED: $t"; exit 1; }
      echo "$log" | grep -vE '^(running|$)|^test result: ok. 0' || true
    done
    # 0.3.1: the Ollama shape — /api/chat == /v1/chat/completions == llama-server's record, and unload/reload unchanged
    echo "## serve_oracle_ollama_shape"
    python3 -B testing/serve_oracle.py --bankml || { echo "FAILED: serve_oracle_ollama_shape"; exit 1; }
    # 0.3.2: the C API — bankml_chat == serve --native (ternary, live) == llama-server's record (1-bit); refusals
    echo "## capi_chat_oracle"
    python3 -B testing/capi/capi_oracle.py --chat || { echo "FAILED: capi_chat_oracle"; exit 1; }
  else
    echo "BANKML_GGML_LIB unset: oracles, A/B and budgets skipped"
  fi
  echo "# gate done"
} 2>&1 | tee "$out" | tee -a testing/live.log
