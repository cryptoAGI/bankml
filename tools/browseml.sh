#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# browseML: build bankML's engine for the browser (WebAssembly, wasm32-wasip1) into hf/space/browseml.wasm, where the
# Space serves it beside browseml.js. The same engine as the native build; the target is installed on first use.
#   tools/browseml.sh            build
#   tools/browseml.sh oracle     build, then check it against llama-server b11192's recorded answers (Node 20+)
set -eu
cd "$(dirname "$0")/.."
rustup target list --installed | grep -qx wasm32-wasip1 || rustup target add wasm32-wasip1
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
(cd browseML && CARGO_TARGET_DIR="../$TARGET_DIR" cargo build --release --target wasm32-wasip1)
case "$TARGET_DIR" in /*) OUT="$TARGET_DIR" ;; *) OUT="$PWD/$TARGET_DIR" ;; esac
cp "$OUT/wasm32-wasip1/release/browseml.wasm" hf/space/browseml.wasm
echo "hf/space/browseml.wasm: $(wc -c < hf/space/browseml.wasm) bytes, sha256 $(sha256sum hf/space/browseml.wasm | cut -c1-16)…"
if [ "${1:-}" = oracle ]; then
  MODELS="${BANKML_MODELS:-.models}"
  FORK="${BANKML_FORKS:-$HOME/.local/share/bankml/forks}/Bonsai-1.7B-Q1_0.gguf.FORK.json"
  node browseML/testing/oracle.mjs hf/space/browseml.wasm "$MODELS" Bonsai-1.7B-Q1_0 "$FORK" "${BROWSEML_TURNS:-}"
fi
