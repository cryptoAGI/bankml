#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# browseML: build bankML's engine for the browser into hf/space/, where the Space serves it beside browseml.js:
# browseml.wasm (wasm32-wasip1, one thread) and browseml-mt.wasm (wasm32-wasip1-threads, its pool on Web Workers,
# for cross-origin-isolated pages). The same engine as the native build; the targets are installed on first use.
#   tools/browseml.sh            build
#   tools/browseml.sh oracle     build, then check it against llama-server b11192's recorded answers (Node 20+)
set -eu
cd "$(dirname "$0")/.."
for t in wasm32-wasip1 wasm32-wasip1-threads; do rustup target list --installed | grep -qx $t || rustup target add $t; done
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
case "$TARGET_DIR" in /*) OUT="$TARGET_DIR" ;; *) OUT="$PWD/$TARGET_DIR" ;; esac
(cd browseML && CARGO_TARGET_DIR="$OUT" cargo build --release --target wasm32-wasip1 && CARGO_TARGET_DIR="$OUT" cargo build --release --target wasm32-wasip1-threads)
cp "$OUT/wasm32-wasip1/release/browseml.wasm" hf/space/browseml.wasm
cp "$OUT/wasm32-wasip1-threads/release/browseml.wasm" hf/space/browseml-mt.wasm
for f in hf/space/browseml.wasm hf/space/browseml-mt.wasm; do echo "$f: $(wc -c < $f) bytes, sha256 $(sha256sum $f | cut -c1-16)…"; done
if [ "${1:-}" = oracle ]; then
  MODELS="${BANKML_MODELS:-.models}"
  FORK="${BANKML_FORKS:-$HOME/.local/share/bankml/forks}/Bonsai-1.7B-Q1_0.gguf.FORK.json"
  node browseML/testing/oracle.mjs hf/space/browseml.wasm "$MODELS" Bonsai-1.7B-Q1_0 "$FORK" "${BROWSEML_TURNS:-}"
fi
