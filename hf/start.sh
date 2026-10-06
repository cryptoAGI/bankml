#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# The Hugging Face Space: bankML's own engine (`bankml serve --native`, the ternary Bonsai-8B verified against its pin)
# on loopback, and the bankML console in front of it, public and read-only, speaking as sAGI/personas/bankml.persona.
set -e
cd "$HOME/bankml"
export BANKML_THREADS="${BANKML_THREADS:-$(nproc)}" BANKML_GPU=off BANKML_SERVE_LISTEN=127.0.0.1:18093
./target/release/bankml serve models/Ternary-Bonsai-8B-Q2_0_g64.gguf --fork models/FORK.json --native \
  --listen 127.0.0.1:18093 --upstream 127.0.0.1:18094 --ctx 4096 &
# the console answers at once; it reports "no verified engine" until serve has hashed and loaded the model
exec python3 sAGI/console.py --public "${SPACE_HOST:-localhost}" --host 0.0.0.0 --port 7860
