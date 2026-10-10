#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# browseml-space: get hf/browseml-space/ (PYTHAI/browseML, "bankML in your browser") ready to publish. It does not
# upload. The page's engine files are copies of hf/space's (browseml*.js, browseML/*.FORK.json); COPIED.json holds
# their sha256 at the time of copying, so a copy that has drifted from its source is visible.
#   tools/browseml-space.sh          copy the three .wasm built by tools/browseml.sh from hf/space/, print their sha256,
#                                    and check every copied file against hf/space/ and COPIED.json
#   tools/browseml-space.sh sync     copy the scripts and the FORK.json again from hf/space/ and rewrite COPIED.json
set -eu
cd "$(dirname "$0")/.."
SRC=hf/space DST=hf/browseml-space
COPIED="browseml.js browseml-worker.js browseml-thread.js browseml-wasi.js browseML/Bonsai-1.7B-Q1_0.gguf.FORK.json"
sha() { sha256sum "$1" | cut -d' ' -f1; }
if [ "${1:-}" = sync ]; then
  for f in $COPIED; do mkdir -p "$DST/$(dirname "$f")"; cp "$SRC/$f" "$DST/$f"; done
  {
    printf '{\n "source": "%s/ of github.com/cryptoAGI/bankml, at %s%s",\n "files": {\n' "$SRC" \
      "$(git rev-parse --short HEAD 2>/dev/null || echo '?')" "$(git diff --quiet -- $SRC 2>/dev/null || echo ' with uncommitted changes')"
    n=0; for f in $COPIED; do n=$((n + 1)); printf '  "%s": "%s"%s\n' "$f" "$(sha "$DST/$f")" "$([ $n -lt 5 ] && echo ,)"; done
    printf ' }\n}\n'
  } > "$DST/COPIED.json"
  echo "copied $n files from $SRC/ into $DST/; COPIED.json rewritten"
fi
for f in browseml.wasm browseml-mt.wasm browseml-mt-relaxed.wasm; do
  if [ -f "$SRC/$f" ]; then cp "$SRC/$f" "$DST/$f"; echo "$DST/$f: $(wc -c < "$DST/$f") bytes, sha256 $(sha "$DST/$f")"
  else echo "$SRC/$f is missing: build it with tools/browseml.sh" >&2; missing=1; fi
done
drift=0
for f in $COPIED; do
  have=$(sha "$DST/$f"); want=$(sed -n "s|^  \"$f\": \"\([0-9a-f]*\)\".*|\1|p" "$DST/COPIED.json")
  [ "$have" = "$want" ] || { echo "drift: $DST/$f is not the copy COPIED.json records" >&2; drift=1; }
  [ "$have" = "$(sha "$SRC/$f")" ] || { echo "drift: $DST/$f differs from $SRC/$f (tools/browseml-space.sh sync copies it again)" >&2; drift=1; }
done
[ $drift = 0 ] && echo "the copied scripts and FORK.json match $SRC/ and COPIED.json"
[ "${missing:-0}" = 0 ] && [ $drift = 0 ]
