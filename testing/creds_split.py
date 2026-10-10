#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The credential split on bankML's Hugging Face page (LICENSING.md): hf/space/bankml-creds.js — the GPL-3.0-only module
that signs a visitor in and holds their token — is the only file there with token code; the page's core (index.html,
bankml-chat.js, the dashboard, the input field's bundle uif-space.js) contains none of it, and loads the module with its
own <script type="module"> instead of importing it. The same check as ultimate-bankml-ui's scripts/check-creds-split.mjs.
run: python3 testing/creds_split.py   (exit 0 = split holds)"""
import re, sys
from pathlib import Path

space = Path(__file__).resolve().parents[1] / "hf" / "space"
MARKERS = ["Authorization", "Bearer ", "oauthLoginUrl", "oauthHandleRedirectIfPresent", "bankml-creds:session",
           "bankml-creds:device", "whoami-v2", "accessToken", "usePastedToken", "OAUTH_CLIENT_ID", "@huggingface/hub"]
bad = []
creds = space / "bankml-creds.js"
if not creds.is_file():
    bad.append("hf/space/bankml-creds.js is missing")
elif not creds.read_text(encoding="utf-8").startswith("/*! bankml-creds.js — SPDX-License-Identifier: GPL-3.0-only"):
    bad.append("hf/space/bankml-creds.js lacks its GPL-3.0-only banner")
if not (space / "bankml-creds-LICENSE.txt").is_file() or "GNU GENERAL PUBLIC LICENSE" not in (space / "bankml-creds-LICENSE.txt").read_text(encoding="utf-8"):
    bad.append("hf/space/bankml-creds-LICENSE.txt (the GPL-3.0 text) is missing")
core = [p for p in sorted(space.glob("*")) if p.suffix in (".js", ".html") and p.name != "bankml-creds.js"]
for p in core:
    t = p.read_text(encoding="utf-8", errors="replace")
    for m in MARKERS:
        if m in t:
            bad.append(f"hf/space/{p.name}: contains {m!r} (token code belongs in bankml-creds.js)")
    if re.search(r"""(import\s[^;]*from\s*|import\(\s*)['"][^'"]*bankml-creds""", t):
        bad.append(f"hf/space/{p.name}: imports bankml-creds.js (the core talks to it only through window.bankmlCreds)")
page = (space / "index.html").read_text(encoding="utf-8")
if not re.search(r'<script type="module" src="bankml-creds\.js"', page):
    bad.append("hf/space/index.html does not load bankml-creds.js with its own <script type=\"module\">")
for b in bad:
    print("FAIL", b)
print(f"credential split: {'ok' if not bad else 'FAILED'} — {len(core)} core files checked; bankml-creds.js GPL-3.0-only, its licence beside it")
sys.exit(1 if bad else 0)
