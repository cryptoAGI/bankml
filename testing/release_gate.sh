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
  python3 -B testing/test_console.py | tail -1 | sed 's/^/console: /'
  python3 -B testing/test_connectors.py | tail -1 | sed 's/^/postgres connector (throwaway cluster): /'
  python3 -B testing/test_chain.py | tail -1 | sed 's/^/iNFT mint and load (throwaway anvil devnet): /'
  BANKML_TEST_CARRIER=1 python3 -B testing/test_models.py | tail -1 | sed 's/^/model importer (loopback source; carrier on spare ports): /'
  python3 testing/guard_agree.py target/release/bankml $(ls .models/*.gguf 2>/dev/null) | tail -1
  # 0.3.2: the C library, and bankml_log (a C-variadic function defined in Rust) against libc snprintf, from C
  cargo build --release --locked -q -p bankml-capi
  python3 -B testing/capi/capi_oracle.py --printf || { echo "FAILED: printf oracle"; exit 1; }
  if [ -n "${BANKML_GGML_LIB:-}" ]; then
    # O6b: the JSON-schema grammar text, from b11192's own libllama-common (no model): re-recorded when the source is here
    if [ -n "${LLAMA_SRC:-}" ]; then python3 -B testing/schema_oracle.py 2>/dev/null | tail -3; fi
    # 0.3.5: the content rule from llama.cpp's own chat parser, over every prefix of every recorded constrained answer
    if [ -n "${LLAMA_SRC:-}" ]; then python3 -B testing/content_oracle.py 2>/dev/null | tail -3; fi
    # 0.3.7: libstdc++'s own std::sort orders (typical-p's sort), re-recorded whenever a C++ compiler is here
    if command -v g++ >/dev/null; then
      mkdir -p .models/oracle-sort && g++ -O2 -o target/sort_oracle testing/sort_oracle.cpp && target/sort_oracle > .models/oracle-sort/cases.txt
    fi
    for t in oracle_tokenizer oracle_chat_template oracle_forward_embed_norm oracle_forward_qkv_rope oracle_forward_attention oracle_forward_attention_tiled oracle_forward_attention_split oracle_forward_swiglu_sweep oracle_forward_model oracle_forward_model_ternary oracle_greedy_llama_server oracle_greedy_llama_server_ternary oracle_greedy_llama_server_long oracle_greedy_llama_server_deep oracle_sample_llama_server oracle_native_serve oracle_grammar_masks oracle_json_mode oracle_json_mode_ternary oracle_schema_grammars oracle_json_schema oracle_json_schema_ternary oracle_json_schema_o4 oracle_json_content oracle_persona_layer oracle_penalties oracle_penalties_8b oracle_samplers oracle_samplers_8b oracle_std_sort oracle_ggml_b11192_q8_0_kv_kernels gpu_q1_0_mat_vec_bit_exact oracle_train_script oracle_train_imprint oracle_ggml_b11192_real_bonsai_1_7b oracle_ggml_b11192_real_bonsai_8b_q1_0 oracle_ggml_b11192_real_ternary_bonsai_8b \
             oracle_ggml_b11192_f16 oracle_tokenizer_smollm oracle_chat_template_chatml oracle_forward_model_bonsai_1_7b oracle_forward_model_llama_f16 \
             oracle_llama_server_bonsai_1_7b oracle_llama_server_llama_f16 oracle_native_serve_o4 \
             ab_vs_ggml ab_vs_ggml_q2_0 bench_q1_0_prefill_act bench_memory_floor decode_budget_q1_0 decode_budget_q2_0; do
      echo "## $t"
      name=$(cargo test --release -q -- --list --ignored 2>/dev/null | sed -n "s/^\(.*::$t\): test$/\1/p")
      log=$(cargo test --release --locked -q -- --ignored --exact "$name" --nocapture --test-threads=1 2>&1) || { echo "$log"; echo "FAILED: $t"; exit 1; }
      echo "$log" | grep -vE '^(running|$)|^test result: ok. 0' || true
    done
    # 0.3.1: the Ollama shape — /api/chat == /v1/chat/completions == llama-server's record, and unload/reload unchanged
    echo "## serve_oracle_ollama_shape"
    python3 -B testing/serve_oracle.py --bankml || { echo "FAILED: serve_oracle_ollama_shape"; exit 1; }
    # 0.3.4 (O4): the same, and JSON mode live, on the models O4 opened; mindx-gen39 asked for by Ollama's tag
    echo "## o4_live (serve_oracle + json_oracle --bankml on Bonsai-1.7B, SmolLM2-135M-Instruct, mindx-gen39)"
    for m in "Bonsai-1.7B-Q1_0" "SmolLM2-135M-Instruct-F16" "mindx-gen39-F16 mindx-gen39"; do
      python3 -B testing/serve_oracle.py --bankml $m || { echo "FAILED: serve_oracle --bankml $m"; exit 1; }
      python3 -B testing/json_oracle.py --bankml $m || { echo "FAILED: json_oracle --bankml $m"; exit 1; }
      python3 -B testing/json_schema_oracle.py --bankml $m || { echo "FAILED: json_schema_oracle --bankml $m"; exit 1; }
    done
    # 0.3.5 (O5): the created mindx-gen39 with promote.py's persona layer, live: /api/chat, /api/generate, /v1, /api/ps
    echo "## persona_oracle_live"
    python3 -B testing/persona_oracle.py --bankml || { echo "FAILED: persona_oracle_live"; exit 1; }
    # O2: the penalties live — /v1 top-level fields and /api/chat options, refusals as 400s with llama-server's message
    echo "## penalty_oracle_live"
    python3 -B testing/penalty_oracle.py --bankml mindx-gen39-F16 mindx-gen39 || { echo "FAILED: penalty_oracle_live"; exit 1; }
    # 0.3.7: the rest of the chain live (DRY, XTC, top-n-σ and dynamic temperature through /v1; typical-p also /api)
    echo "## sampler_oracle_live"
    python3 -B testing/penalty_oracle.py --kind sampler --bankml mindx-gen39-F16 mindx-gen39 || { echo "FAILED: sampler_oracle_live"; exit 1; }
    # 0.3.8: the context limit as llama-server's (context shift off): stop at the full context; past it, its 400 body
    echo "## context_oracle_live"
    python3 -B testing/context_oracle.py --bankml Bonsai-1.7B-Q1_0 || { echo "FAILED: context_oracle_live"; exit 1; }
    # 0.3.8: slot save/restore on the native engine — answers after a restore identical to an empty slot's, across a restart
    echo "## slot_oracle_live"
    python3 -B testing/slot_oracle.py Bonsai-1.7B-Q1_0 || { echo "FAILED: slot_oracle_live"; exit 1; }
    # 0.3.8: one slot, interleaved conversations through llama-server's host prompt cache; simultaneous requests queued
    echo "## session_oracle_live"
    python3 -B testing/session_oracle.py --bankml Bonsai-1.7B-Q1_0 || { echo "FAILED: session_oracle_live"; exit 1; }
    # 0.3.9: the q8_0 KV cache (BANKML_CACHE_TYPE=q8_0) against llama-server --cache-type-k/v q8_0, Hadamard rotation included
    echo "## kv_oracle_live"
    python3 -B testing/kv_oracle.py --bankml Bonsai-1.7B-Q1_0 || { echo "FAILED: kv_oracle_live"; exit 1; }
    # 0.3.8: /v1 logprobs — ids, texts, bytes and every logprob the same float as llama-server's; stop words, UTF-8 splits
    echo "## logprobs_oracle_live"
    python3 -B testing/logprobs_oracle.py --bankml Bonsai-1.7B-Q1_0 || { echo "FAILED: logprobs_oracle_live"; exit 1; }
    # 0.3.3: JSON mode and grammars live — /v1 response_format (streamed once) and grammar, /api/chat format "json"
    echo "## json_oracle_live"
    python3 -B testing/json_oracle.py --bankml || { echo "FAILED: json_oracle_live"; exit 1; }
    # O6b: JSON schemas live — /v1 response_format json_schema, top-level json_schema, json_object with a schema (one
    # streamed), and /api/chat with format: <schema>
    echo "## json_schema_oracle_live"
    python3 -B testing/json_schema_oracle.py --bankml || { echo "FAILED: json_schema_oracle_live"; exit 1; }
    # 0.3.2: the C API — bankml_chat == serve --native (ternary, live) == llama-server's record (1-bit); refusals
    echo "## capi_chat_oracle"
    python3 -B testing/capi/capi_oracle.py --chat || { echo "FAILED: capi_chat_oracle"; exit 1; }
  else
    echo "BANKML_GGML_LIB unset: oracles, A/B and budgets skipped"
  fi
  # O5: bankml convert, byte for byte against llama.cpp b11192's convert_hf_to_gguf.py --outtype f16 — each safetensors
  # directory under .models/convert/ beside llama.cpp's GGUF of it (.models/convert/<dir>.oracle.gguf; the directory's
  # name is part of the input: llama.cpp names the model from it); and the name heuristics against b11192's gguf-py
  for d in .models/convert/*/; do
    d=${d%/}
    [ -f "$d.oracle.gguf" ] || continue
    echo "## oracle_convert_b11192 $(basename "$d")"
    log=$(BANKML_CONVERT_DIR="$d" BANKML_CONVERT_ORACLE="$d.oracle.gguf" cargo test --release --locked -q --lib -- --ignored --exact convert::tests::oracle_convert_b11192 --nocapture 2>&1) || { echo "$log"; echo "FAILED: oracle_convert_b11192 $d"; exit 1; }
    echo "$log" | grep -vE '^(running|$)|^test result: ok. 0' || true
  done
  if [ -n "${BANKML_LLAMA_SRC:-}" ]; then
    echo "## oracle_name_heuristics"
    python3 -B testing/convert_oracle.py --names target/names.jsonl >/dev/null
    log=$(BANKML_NAMES_ORACLE=target/names.jsonl cargo test --release --locked -q --lib -- --ignored --exact convert::tests::oracle_name_heuristics --nocapture 2>&1) || { echo "$log"; echo "FAILED: oracle_name_heuristics"; exit 1; }
    echo "$log" | grep -vE '^(running|$)|^test result: ok. 0' || true
  fi
  echo "# gate done"
} 2>&1 | tee "$out" | tee -a testing/live.log
