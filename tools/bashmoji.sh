#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# =============================================================================
# bashmoji.sh — emoji + colour for shell scripts, safe to source.
#   vendored:  tools/bashmoji.sh in bankml, from cryptoAGI/bashmoji e81d425 (MIT OR Apache-2.0); update by copying the file
#
#   home:      https://github.com/cryptoAGI/bashmoji
#   used by:   the DeltaVerse installer — https://deltaverse.pythai.net
#   author:    codephreak
#
#   source ./bashmoji.sh            # in a script: defines BM_* glyphs/colours + bm_* helpers
#   ./bashmoji.sh                   # run it: shows every glyph, colour and helper
#   ./bashmoji.sh --plain           # the same, as the ASCII fallback renders it
#
# Rules this file keeps, so sourcing it can never break the script that sources it:
#   · every name is prefixed BM_ (variables) or bm_ (functions) — it never assigns
#     PATH, PORT, HOME, NODE or any other name a script or the shell already uses;
#   · it runs nothing and prints nothing when sourced;
#   · it works in bash 3.2 (macOS) and later: no associative arrays, no ${x,,};
#   · it degrades by itself: no colour when stdout is not a terminal, when
#     NO_COLOR is set (https://no-color.org) or TERM=dumb; ASCII instead of emoji
#     when the locale is not UTF-8 or BASHMOJI=plain.
#
# Overrides:  BASHMOJI=plain|emoji   BASHMOJI_COLOR=never|always   NO_COLOR=1
# =============================================================================

# guard against double sourcing
if [ -n "${BM_LOADED:-}" ]; then return 0 2>/dev/null || exit 0; fi
BM_LOADED=1
BM_VERSION="2.0.0"
BM_HOME="https://github.com/cryptoAGI/bashmoji"
BM_DELTAVERSE="https://deltaverse.pythai.net"

# ── capability detection ─────────────────────────────────────────────────────
bm__utf8() {
  case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in *[Uu][Tt][Ff]-8*|*[Uu][Tt][Ff]8*) return 0;; esac
  return 1
}
bm__color() {
  case "${BASHMOJI_COLOR:-auto}" in always) return 0;; never) return 1;; esac
  [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != "dumb" ] && [ -t 1 ]
}
case "${BASHMOJI:-auto}" in
  plain) BM_EMOJI=0;;
  emoji) BM_EMOJI=1;;
  *)     if bm__utf8 && [ "${TERM:-dumb}" != "dumb" ]; then BM_EMOJI=1; else BM_EMOJI=0; fi;;
esac
if bm__color; then BM_COLOR=1; else BM_COLOR=0; fi

# bm__g NAME "emoji" "ascii" — define one glyph with its fallback
bm__g() { if [ "$BM_EMOJI" = 1 ]; then eval "$1=\"\$2\""; else eval "$1=\"\$3\""; fi; }
# bm__c NAME "escape" — define one colour (empty when colour is off)
bm__c() { if [ "$BM_COLOR" = 1 ]; then eval "$1=\"\$2\""; else eval "$1=''"; fi; }

# ── colours (ANSI; the full 256/true-colour palette lives in colorbash.sh) ──
bm__c BM_NC      $'\033[0m'
bm__c BM_BOLD    $'\033[1m'
bm__c BM_DIM     $'\033[2m'
bm__c BM_ULINE   $'\033[4m'
bm__c BM_RED     $'\033[0;31m'
bm__c BM_GREEN   $'\033[0;32m'
bm__c BM_YELLOW  $'\033[0;33m'
bm__c BM_BLUE    $'\033[0;34m'
bm__c BM_MAGENTA $'\033[0;35m'
bm__c BM_CYAN    $'\033[0;36m'
bm__c BM_GOLD    $'\033[1;33m'
bm__c BM_ORANGE  $'\033[38;5;208m'
bm__c BM_VIOLET  $'\033[38;5;177m'

# ── glyphs: status ───────────────────────────────────────────────────────────
bm__g BM_OK       "✅" "[ok]"
bm__g BM_FAIL     "❌" "[x]"
bm__g BM_WARN     "⚠️ " "[!]"
bm__g BM_INFO     "ℹ️ " "[i]"
bm__g BM_WORKING  "🔄" "[~]"
bm__g BM_PENDING  "⏳" "[..]"
bm__g BM_SKIP     "⏭️ " "[-]"
bm__g BM_BLOCKED  "🚫" "[no]"
bm__g BM_DEBUG    "🔍" "[?]"
bm__g BM_DONE     "☑️ " "[v]"
# ── glyphs: build & deploy ───────────────────────────────────────────────────
bm__g BM_ROCKET   "🚀" ">>"
bm__g BM_SPARK    "✨" "*"
bm__g BM_WRENCH   "🔧" "[cfg]"
bm__g BM_HAMMER   "🔨" "[build]"
bm__g BM_PACKAGE  "📦" "[pkg]"
bm__g BM_INSTALL  "📥" "[in]"
bm__g BM_TEST     "🧪" "[test]"
bm__g BM_TARGET   "🎯" "[go]"
bm__g BM_BUILDER  "👷" "[build]"
# ── glyphs: system & tools ───────────────────────────────────────────────────
bm__g BM_ROBOT    "🤖" "[ai]"
bm__g BM_BRAIN    "🧠" "[ml]"
bm__g BM_GEAR     "⚙️ " "[sys]"
bm__g BM_TERMINAL "📟" "[$]"
bm__g BM_FOLDER   "📁" "[dir]"
bm__g BM_FILE     "📄" "[file]"
bm__g BM_LINK     "🔗" "[link]"
bm__g BM_GLOBE    "🌐" "[www]"
bm__g BM_SERVER   "🖥️ " "[srv]"
bm__g BM_DOOR     "🚪" "[port]"
bm__g BM_ROUTE    "🛣️ " "[path]"
bm__g BM_BOOK     "📚" "[doc]"
bm__g BM_CLOCK    "⏰" "[t]"
bm__g BM_PYTHON   "🐍" "[py]"
bm__g BM_NODEJS   "🟢" "[node]"
bm__g BM_PENGUIN  "🐧" "[linux]"
bm__g BM_APPLE    "🍎" "[mac]"
bm__g BM_WHALE    "🐳" "[docker]"
# ── glyphs: security ─────────────────────────────────────────────────────────
bm__g BM_KEY      "🔑" "[key]"
bm__g BM_LOCK     "🔒" "[lock]"
bm__g BM_UNLOCK   "🔓" "[open]"
bm__g BM_SHIELD   "🛡️ " "[guard]"
bm__g BM_SIGN     "✍️ " "[sign]"
# ── glyphs: chain ────────────────────────────────────────────────────────────
bm__g BM_DELTA    "Δ"  "D"
bm__g BM_CHAIN    "⛓️ " "[chain]"
bm__g BM_CONTRACT "📜" "[sol]"
bm__g BM_WALLET   "👛" "[wallet]"
bm__g BM_TOKEN    "🪙" "[token]"
bm__g BM_GAS      "⛽" "[gas]"
bm__g BM_NFT      "🎨" "[nft]"
bm__g BM_BRIDGE   "🌉" "[bridge]"
bm__g BM_ORACLE   "🔮" "[oracle]"
bm__g BM_BITCOIN  "₿"  "BTC"
bm__g BM_ETHEREUM "⟠"  "ETH"
bm__g BM_UP       "📈" "[up]"
bm__g BM_DOWN     "📉" "[down]"

# ── helpers ──────────────────────────────────────────────────────────────────
# bm_say  "message"            headline in gold, with BM_PREFIX (default: the rocket)
# bm_sub  "message"            dim detail line under a headline
# bm_ok / bm_warn / bm_info    one line with the matching glyph + colour (stdout)
# bm_err  "message"            one line to stderr
# bm_die  "message" [code]     bm_err, then exit (default 1)
# bm_step N TOTAL "message"    "[N/TOTAL]" progress headline
# bm_link URL [text]           OSC-8 clickable link on terminals that support it, plain text elsewhere
# bm_banner "title" "subtitle" a boxed title line
bm_say()  { printf '%s %s%s%s\n' "${BM_PREFIX:-$BM_ROCKET}" "$BM_GOLD" "$*" "$BM_NC"; }
bm_sub()  { printf '   %s%s%s\n' "$BM_DIM" "$*" "$BM_NC"; }
bm_ok()   { printf '%s %s%s%s\n' "$BM_OK"   "$BM_GREEN"  "$*" "$BM_NC"; }
bm_warn() { printf '%s %s%s%s\n' "$BM_WARN" "$BM_YELLOW" "$*" "$BM_NC"; }
bm_info() { printf '%s %s%s%s\n' "$BM_INFO" "$BM_CYAN"   "$*" "$BM_NC"; }
bm_err()  { printf '%s %s%s%s\n' "$BM_FAIL" "$BM_RED"    "$*" "$BM_NC" >&2; }
bm_die()  { bm_err "$1"; exit "${2:-1}"; }
bm_step() { printf '%s %s[%s/%s]%s %s%s%s\n' "${BM_PREFIX:-$BM_ROCKET}" "$BM_DIM" "$1" "$2" "$BM_NC" "$BM_GOLD" "$3" "$BM_NC"; }
bm_link() {
  if [ "$BM_COLOR" = 1 ] && [ -z "${BM_NO_OSC8:-}" ]; then
    printf '\033]8;;%s\033\\%s\033]8;;\033\\' "$1" "${2:-$1}"
  else
    printf '%s' "${2:-$1}"
  fi
}
bm_banner() {
  printf '%s%s %s %s%s\n' "$BM_BOLD" "$BM_CYAN" "$BM_DELTA" "$1" "$BM_NC"
  [ -n "${2:-}" ] && printf '  %s%s%s\n' "$BM_DIM" "$2" "$BM_NC"
  return 0
}

# ── run directly: show the whole set ─────────────────────────────────────────
bm__main() {
  case "${1:-}" in
    --plain)   BASHMOJI=plain BM_LOADED= exec bash "$0";;
    --version) echo "bashmoji $BM_VERSION"; return 0;;
    -h|--help) sed -n '3,26p' "$0"; return 0;;
  esac
  bm_banner "bashmoji $BM_VERSION" "emoji $([ "$BM_EMOJI" = 1 ] && echo on || echo 'off (ASCII)') · colour $([ "$BM_COLOR" = 1 ] && echo on || echo off)"
  echo
  set | grep -E '^BM_[A-Z_]+=' | grep -vE '^BM_(LOADED|VERSION|HOME|DELTAVERSE|EMOJI|COLOR|NC|BOLD|DIM|ULINE|RED|GREEN|YELLOW|BLUE|MAGENTA|CYAN|GOLD|ORANGE|VIOLET)=' |
    while IFS='=' read -r name _; do eval "printf '  %-4s %s\n' \"\$$name\" \"$name\""; done
  echo
  bm_ok   "bm_ok — it worked"
  bm_warn "bm_warn — look at this"
  bm_info "bm_info — for your information"
  bm_err  "bm_err — to stderr" 2>&1
  bm_step 3 7 "bm_step — a numbered headline"
  echo
  printf '  %s  ·  made for the DeltaVerse: %s\n' "$(bm_link "$BM_HOME" bashmoji)" "$(bm_link "$BM_DELTAVERSE")"
}
if [ "${BASH_SOURCE[0]:-$0}" = "$0" ]; then bm__main "$@"; fi
