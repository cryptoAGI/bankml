#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# install.sh — bankml and Savante, from a fresh machine to a verified answer, in the order of docs/usage.md.
#
#   ./install.sh                 check, build, engine, python, canon, model, then start Savante
#   ./install.sh <step> …        run chosen steps: check build engine python canon model voice start stop status
#   curl -fsSL https://raw.githubusercontent.com/cryptoAGI/bankml/main/install.sh | bash
#                                outside a checkout: clone bankml to $BANKML_DIR (default ~/bankml) and run from there
#
#   --skip-tests   build without `cargo test --release`
#   --no-start     install everything, start nothing
#   --view         also start view mode for the LAN (0.0.0.0:7874)
#   --voice        also install Savante's Piper voice (step `voice`)
#   -h, --help     this text
#
# Nothing here needs sudo. Every download is checked before it is used: llama.cpp b11192 against its published
# sha256, and the model by the importer (the publisher's sha256, bankml guard, a FORK.json pin), as bankml serve
# refuses anything else. Environment: BANKML_DIR, BANKML_DATA (~/.local/share/bankml), BANKML_LLAMA_SERVER,
# SAVANTE_CANON (~/savante), BANKML_PYTHON, and every BANKML_* variable docs/usage.md §13 lists.
set -euo pipefail

REPO_URL="https://github.com/cryptoAGI/bankml"
LLAMA_TAG="b11192"
LLAMA_TGZ="llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz"
LLAMA_URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/${LLAMA_TGZ}"
LLAMA_SHA256="34cf6fa5de9da0db3932c78fe15fed2fbca17451e665dac0a4f6a3c8fc881ec7"
CANON_URL="https://github.com/cryptoAGI/savante"
PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
PIPER_VOICE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/cori/high"
RUST_MIN="1.95"
PY_MIN="3.10"
GRADIO_SPEC="gradio>=3.37,<4"   # the UI is written for Gradio 3 (its layout and scripts are Gradio 3 workarounds)

# ── where we are: a checkout, or a pipe that has to fetch one ────────────────────────────────────────────────
HERE=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [ -z "$HERE" ] || [ ! -f "$HERE/Cargo.toml" ] || ! grep -q '^name = "bankml"' "$HERE/Cargo.toml"; then
  dir="${BANKML_DIR:-$HOME/bankml}"
  command -v git >/dev/null || { echo "install.sh: git is required to fetch bankml" >&2; exit 1; }
  if [ -d "$dir/.git" ]; then echo ">> using the checkout at $dir"; else echo ">> cloning $REPO_URL into $dir"; git clone -q "$REPO_URL" "$dir"; fi
  exec bash "$dir/install.sh" "$@"
fi
cd "$HERE"

# ── bashmoji: glyphs and colour that degrade by themselves; a plain fallback if the file is missing ──────────
if [ -f "$HERE/tools/bashmoji.sh" ]; then
  # shellcheck source=tools/bashmoji.sh
  . "$HERE/tools/bashmoji.sh"
else
  bm_say()  { printf '>> %s\n' "$*"; }
  bm_sub()  { printf '   %s\n' "$*"; }
  bm_ok()   { printf '[ok] %s\n' "$*"; }
  bm_warn() { printf '[!] %s\n' "$*"; }
  bm_info() { printf '[i] %s\n' "$*"; }
  bm_err()  { printf '[x] %s\n' "$*" >&2; }
  bm_die()  { bm_err "$1"; exit "${2:-1}"; }
  bm_step() { printf '>> [%s/%s] %s\n' "$1" "$2" "$3"; }
  bm_link() { printf '%s' "${2:-$1}"; }
  bm_banner() { printf '%s\n' "$1"; [ -n "${2:-}" ] && printf '  %s\n' "$2"; return 0; }
fi

DATA="${BANKML_DATA:-$HOME/.local/share/bankml}"
ENV_FILE="$DATA/install.env"       # what this installer chose (engine, python), read back by later runs
BIN="$HERE/target/release/bankml"
CANON="${SAVANTE_CANON:-$HOME/savante}"
LOGS="$DATA/logs"

# ── options ──────────────────────────────────────────────────────────────────────────────────────────────────
SKIP_TESTS=0 NO_START=0 WITH_VIEW=0 WITH_VOICE=0
STEPS=()
for a in "$@"; do
  case "$a" in
    --skip-tests) SKIP_TESTS=1;;
    --no-start)   NO_START=1;;
    --view)       WITH_VIEW=1;;
    --voice)      WITH_VOICE=1;;
    -h|--help)    sed -n '3,19p' "$HERE/install.sh" | sed 's/^# \{0,1\}//'; exit 0;;
    check|build|engine|python|canon|model|voice|start|stop|status) STEPS+=("$a");;
    *) bm_die "unknown argument: $a (see ./install.sh --help)" 2;;
  esac
done
if [ ${#STEPS[@]} -eq 0 ]; then
  STEPS=(check build engine python canon model)
  [ "$WITH_VOICE" = 1 ] && STEPS+=(voice)
  [ "$NO_START" = 1 ] || STEPS+=(start)
fi

# ── small helpers ────────────────────────────────────────────────────────────────────────────────────────────
have() { command -v "$1" >/dev/null 2>&1; }
# version_ge A B: A >= B for dotted versions
version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }
save_env() {  # save_env KEY VALUE: remember a choice in $ENV_FILE
  mkdir -p "$DATA"; touch "$ENV_FILE"
  grep -v "^$1=" "$ENV_FILE" > "$ENV_FILE.tmp" || true
  printf '%s=%q\n' "$1" "$2" >> "$ENV_FILE.tmp"; mv "$ENV_FILE.tmp" "$ENV_FILE"
}
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
sha256_of() { if have sha256sum; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fetch() {  # fetch URL FILE: resumable download
  local bar=--silent; [ -t 2 ] && bar=--progress-bar
  curl -fL "$bar" --retry 3 --retry-delay 2 -C - -o "$2" "$1" || curl -fL "$bar" --retry 3 -o "$2" "$1"
}
listening() { ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$1\$"; }
# pid_on PORT: the pid of this user's process listening on PORT (never by pattern-matching command lines)
pid_on() { ss -ltnpH "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2; }

# the engine: an explicit BANKML_LLAMA_SERVER, else what a previous run installed, else the importer's default
LLAMA_SERVER="${BANKML_LLAMA_SERVER:-${INSTALL_LLAMA_SERVER:-}}"
if [ -z "$LLAMA_SERVER" ]; then
  for c in "$DATA/llama-$LLAMA_TAG/llama-server" "$HOME/sAGI/bonsai/llama-$LLAMA_TAG/llama-server"; do
    [ -x "$c" ] && { LLAMA_SERVER="$c"; break; }
  done
fi
PY="${BANKML_PYTHON:-${INSTALL_PYTHON:-python3}}"

# ── the steps ────────────────────────────────────────────────────────────────────────────────────────────────
step_check() {
  local bad=0 v
  if have cargo; then
    v="$(cargo --version | awk '{print $2}')"
    if version_ge "$v" "$RUST_MIN"; then bm_ok "Rust $v"; else bm_err "Rust $v is older than $RUST_MIN: rustup update"; bad=1; fi
  else
    bm_err "Rust is missing: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh, then open a new shell"; bad=1
  fi
  if have python3; then
    v="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    if version_ge "$v" "$PY_MIN"; then bm_ok "Python $v"; else bm_err "Python $v is older than $PY_MIN"; bad=1; fi
  else
    bm_err "python3 is missing"; bad=1
  fi
  for t in git curl tar ss; do
    if have "$t"; then bm_ok "$t"; else bm_err "$t is missing"; bad=1; fi
  done
  if have sha256sum || have shasum; then bm_ok "sha256sum"; else bm_err "sha256sum (coreutils) is missing"; bad=1; fi
  case "$(uname -s)/$(uname -m)" in
    Linux/x86_64) bm_ok "Linux x86_64";;
    *) bm_warn "$(uname -s)/$(uname -m): the engine download is Linux x86_64 only; set BANKML_LLAMA_SERVER to your own llama.cpp $LLAMA_TAG llama-server";;
  esac
  if grep -qw avx2 /proc/cpuinfo 2>/dev/null; then bm_ok "AVX2: bankml's fast kernels"; else bm_warn "no AVX2: bankml falls back to its scalar kernels (correct, slower)"; fi
  if [ -r /proc/meminfo ]; then
    local total avail
    total=$(awk '/MemTotal/ {printf "%.1f", $2/1048576}' /proc/meminfo)
    avail=$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)
    bm_info "RAM ${total} GB, ${avail} GB available (the 1-bit 8B model maps 1.16 GB; the stack runs in about 2 GB)"
  fi
  local free_gb
  free_gb=$(df -Pk "$HERE" | awk 'NR==2 {printf "%.1f", $4/1048576}')
  bm_info "disk ${free_gb} GB free here (the default model is 1.16 GB; the importer keeps 1.5 GB spare)"
  [ "$bad" = 0 ] || bm_die "fix the items marked above, then run ./install.sh again"
}

step_build() {
  bm_sub "cargo build --release (no crates: bankml has no dependencies)"
  cargo build --release
  if [ "$SKIP_TESTS" = 1 ]; then
    bm_info "tests skipped (--skip-tests)"
  else
    bm_sub "cargo test --release (unit + end-to-end, offline)"
    cargo test --release --quiet
  fi
  bm_ok "$("$BIN" version)"
}

step_engine() {
  if [ -n "$LLAMA_SERVER" ] && [ -x "$LLAMA_SERVER" ]; then
    bm_ok "llama-server: $LLAMA_SERVER"
    save_env INSTALL_LLAMA_SERVER "$LLAMA_SERVER"
    return
  fi
  [ "$(uname -s)/$(uname -m)" = "Linux/x86_64" ] ||
    bm_die "no llama.cpp $LLAMA_TAG build for $(uname -s)/$(uname -m) here: build it, then set BANKML_LLAMA_SERVER"
  local tgz="$DATA/$LLAMA_TGZ" got
  mkdir -p "$DATA"
  if [ -f "$tgz" ] && [ "$(sha256_of "$tgz")" = "$LLAMA_SHA256" ]; then
    bm_ok "$LLAMA_TGZ already here, sha256 matches"
  else
    rm -f "$tgz"   # absent, or not the published file: start clean
    bm_sub "downloading llama.cpp $LLAMA_TAG (17 MB)"
    fetch "$LLAMA_URL" "$tgz"
    got="$(sha256_of "$tgz")"
    if [ "$got" != "$LLAMA_SHA256" ]; then
      rm -f "$tgz"
      bm_die "$LLAMA_TGZ sha256 $got is not the published $LLAMA_SHA256: refused"
    fi
    bm_ok "sha256 $LLAMA_SHA256"
  fi
  tar -xzf "$tgz" -C "$DATA"
  LLAMA_SERVER="$DATA/llama-$LLAMA_TAG/llama-server"
  [ -x "$LLAMA_SERVER" ] || bm_die "the archive did not contain llama-$LLAMA_TAG/llama-server"
  save_env INSTALL_LLAMA_SERVER "$LLAMA_SERVER"
  bm_ok "llama-server: $LLAMA_SERVER"
}

step_python() {
  if "$PY" -c 'import gradio, numpy' 2>/dev/null; then
    bm_ok "$("$PY" -c 'import gradio, numpy; print("gradio", gradio.__version__, "· numpy", numpy.__version__)') ($PY)"
    save_env INSTALL_PYTHON "$PY"
    return
  fi
  local venv="$DATA/venv"
  bm_sub "gradio or numpy is missing for $PY: installing $GRADIO_SPEC and numpy into $venv"
  python3 -m venv --system-site-packages "$venv" ||
    bm_die "python3 -m venv failed (on Debian/Ubuntu: sudo apt install python3-venv), or set BANKML_PYTHON"
  "$venv/bin/python" -m pip install --quiet --upgrade pip
  "$venv/bin/python" -m pip install --quiet "$GRADIO_SPEC" numpy
  PY="$venv/bin/python"
  save_env INSTALL_PYTHON "$PY"
  bm_ok "$("$PY" -c 'import gradio, numpy; print("gradio", gradio.__version__, "· numpy", numpy.__version__)') ($PY)"
}

step_canon() {
  if [ -d "$CANON" ]; then
    bm_ok "Savante's canon: $CANON (left as it is: the canon is read-only)"
  else
    bm_sub "cloning $CANON_URL into $CANON"
    git clone -q "$CANON_URL" "$CANON"
    bm_ok "Savante's canon: $CANON"
  fi
}

# the importer runs with the engine and binary this installer chose
importer() { BANKML_LLAMA_SERVER="$LLAMA_SERVER" BANKML_BIN="$BIN" "$PY" -B ui/models.py "$@"; }

step_model() {
  [ -x "$BIN" ] || bm_die "no $BIN yet: run ./install.sh build"
  [ -n "$LLAMA_SERVER" ] && [ -x "$LLAMA_SERVER" ] || bm_die "no llama-server yet: run ./install.sh engine"
  mkdir -p "$LOGS"
  bm_sub "Bonsai-8B 1-bit: imported only if absent (1.16 GB), checked against the publisher's sha256, guarded and pinned,"
  bm_sub "then bankml serve verifies it and starts llama-server on it. The first start hashes the whole file."
  importer first-run | tee "$LOGS/first-run.json"
  if curl -fsS --max-time 5 http://127.0.0.1:18093/bankml >/dev/null 2>&1; then
    bm_ok "bankml serve answers on 127.0.0.1:18093 with a verified model"
  else
    bm_die "bankml serve did not come up: see $DATA/savante/carrier.log and docs/usage.md §12"
  fi
}

step_voice() {
  local dir="${BANKML_PIPER:-$DATA/piper}" f
  mkdir -p "$dir"
  if [ -x "$dir/piper/piper" ]; then
    bm_ok "Piper: $dir/piper/piper"
  else
    bm_sub "downloading Piper 2023.11.14-2"
    fetch "$PIPER_URL" "$dir/piper_linux_x86_64.tar.gz"
    tar -xzf "$dir/piper_linux_x86_64.tar.gz" -C "$dir"
    bm_ok "Piper: $dir/piper/piper"
  fi
  for f in en_GB-cori-high.onnx en_GB-cori-high.onnx.json MODEL_CARD; do
    [ -s "$dir/$f" ] || fetch "$PIPER_VOICE/$f" "$dir/$f"
  done
  bm_ok "voice en_GB-cori-high (public-domain LibriVox recordings)"
  bm_info "render her clips once, one render at a time on a small machine: $PY ui/speak.py"
}

start_ui() {  # start_ui NAME PORT ARGS…: a detached UI with its own log
  local name="$1" port="$2"; shift 2
  if listening "$port"; then bm_ok "$name already on :$port"; return; fi
  mkdir -p "$LOGS"
  BANKML_LLAMA_SERVER="$LLAMA_SERVER" BANKML_BIN="$BIN" SAVANTE_CANON="$CANON" \
    setsid nohup "$PY" -B "$@" > "$LOGS/$name.log" 2>&1 < /dev/null &
  local i
  for i in $(seq 1 60); do listening "$port" && break; sleep 1; done
  if listening "$port"; then bm_ok "$name on :$port (log: $LOGS/$name.log)"; else bm_die "$name did not start: see $LOGS/$name.log"; fi
}

step_start() {
  start_ui interact 7873 ui/savante.py --mode interact --port 7873
  [ "$WITH_VIEW" = 1 ] && start_ui view 7874 ui/view.py --host 0.0.0.0 --port 7874
  echo
  bm_say "Savante is ready"
  bm_sub "talk:   $(bm_link http://127.0.0.1:7873)   (this computer only)"
  [ "$WITH_VIEW" = 1 ] && bm_sub "watch:  http://<this computer's LAN address>:7874"
  bm_sub "check:  curl -s 127.0.0.1:18093/bankml"
  bm_sub "docs:   $(bm_link https://cryptoagi.github.io/bankml/)"
}

step_stop() {
  local port pid
  for port in 7874 7873 18093 18092; do
    pid="$(pid_on "$port")"
    if [ -n "$pid" ]; then
      kill "$pid" 2>/dev/null && bm_ok "stopped :$port (pid $pid)"
    else
      bm_info ":$port was not running"
    fi
  done
}

step_status() {
  local port name
  for port in 18092:llama-server 18093:"bankml serve" 7873:interact 7874:view; do
    name="${port#*:}"; port="${port%%:*}"
    if listening "$port"; then bm_ok "$name on :$port"; else bm_info "$name not running (:$port)"; fi
  done
  if curl -fsS --max-time 3 http://127.0.0.1:18093/bankml > "$DATA/.status.json" 2>/dev/null; then
    "$PY" - "$DATA/.status.json" <<'PY' || true
import json, sys
s = json.load(open(sys.argv[1]))
print("   verified:", s.get("verified"), "·", s.get("model", "?").rsplit("/", 1)[-1], "· sha256", str(s.get("model_sha256", "?"))[:16] + "…")
PY
  fi
  [ -x "$BIN" ] && bm_info "$("$BIN" version)"
  bm_info "engine: ${LLAMA_SERVER:-not installed} · python: $PY"
}

# ── run ──────────────────────────────────────────────────────────────────────────────────────────────────────
title() {
  case "$1" in
    check)  echo "checking this machine";;
    build)  echo "building bankml";;
    engine) echo "llama.cpp $LLAMA_TAG, verified";;
    python) echo "Python for the Savante UI";;
    canon)  echo "Savante's canon";;
    model)  echo "the model, verified, and bankml serve";;
    voice)  echo "Savante's voice (Piper)";;
    start)  echo "starting Savante";;
    stop)   echo "stopping bankml and Savante";;
    status) echo "status";;
  esac
}

bm_banner "bankml installer" "verified low-bit inference for the CPU you already have · $(bm_link https://cryptoagi.github.io/bankml/ docs)"
echo
n=${#STEPS[@]} i=0
for s in "${STEPS[@]}"; do
  i=$((i + 1))
  bm_step "$i" "$n" "$(title "$s")"
  "step_$s"
  echo
done
