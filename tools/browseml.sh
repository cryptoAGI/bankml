#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# browseML: build bankML's engine for the browser into hf/space/, where the Space serves it beside browseml.js:
# browseml.wasm (wasm32-wasip1, one thread) and browseml-mt.wasm (wasm32-wasip1-threads, its pool on Web Workers,
# for cross-origin-isolated pages). The same engine as the native build; the targets are installed on first use.
#   tools/browseml.sh            build
#   tools/browseml.sh oracle     build, then check it against llama-server b11192's recorded answers (Node 20+)
#   tools/browseml.sh test       the kernels' unit tests (wasm_simd, f16, q1_0) on wasm32-wasip1 under Node's WASI, with
#                                SIMD and with relaxed SIMD: every lane equal to the scalar reference, bit for bit
set -eu
cd "$(dirname "$0")/.."
if [ "${1:-}" = test ]; then
  rustup target list --installed | grep -qx wasm32-wasip1 || rustup target add wasm32-wasip1
  export CARGO_TARGET_WASM32_WASIP1_RUNNER="node --no-warnings $PWD/browseML/testing/wasi-run.mjs"
  for f in +simd128 +simd128,+relaxed-simd; do
    # --allow-undefined: the test binary links the GPU loader's dlopen/dlsym (LTO drops them from the real builds);
    # the runner traps if one is called. Tests that start a thread pool or need the models are skipped.
    CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target}/wasm-test" RUSTFLAGS="-C target-feature=$f -C link-arg=--allow-undefined" \
      cargo test --release --target wasm32-wasip1 --lib -p bankml -- wasm_simd f16::tests q1_0::tests --skip avx2_equals --skip par_ --skip oracle --test-threads=1
  done
  exit
fi
for t in wasm32-wasip1 wasm32-wasip1-threads; do rustup target list --installed | grep -qx $t || rustup target add $t; done
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
case "$TARGET_DIR" in /*) OUT="$TARGET_DIR" ;; *) OUT="$PWD/$TARGET_DIR" ;; esac
(cd browseML && CARGO_TARGET_DIR="$OUT" cargo build --release --target wasm32-wasip1 && CARGO_TARGET_DIR="$OUT" cargo build --release --target wasm32-wasip1-threads)
# the relaxed-SIMD threaded build (browseml-mt-relaxed.wasm): RUSTFLAGS replaces .cargo/config.toml's for this one
(cd browseML && CARGO_TARGET_DIR="$OUT/relaxed" RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd -C link-arg=--max-memory=4294967296" \
  cargo build --release --target wasm32-wasip1-threads)
cp "$OUT/wasm32-wasip1/release/browseml.wasm" hf/space/browseml.wasm
cp "$OUT/wasm32-wasip1-threads/release/browseml.wasm" hf/space/browseml-mt.wasm
cp "$OUT/relaxed/wasm32-wasip1-threads/release/browseml.wasm" hf/space/browseml-mt-relaxed.wasm
for f in hf/space/browseml.wasm hf/space/browseml-mt.wasm hf/space/browseml-mt-relaxed.wasm; do echo "$f: $(wc -c < $f) bytes, sha256 $(sha256sum $f | cut -c1-16)…"; done
if [ "${1:-}" = oracle ]; then
  MODELS="${BANKML_MODELS:-.models}"
  FORK="${BANKML_FORKS:-$HOME/.local/share/bankml/forks}/Bonsai-1.7B-Q1_0.gguf.FORK.json"
  node browseML/testing/oracle.mjs hf/space/browseml.wasm "$MODELS" Bonsai-1.7B-Q1_0 "$FORK" "${BROWSEML_TURNS:-}"
fi
