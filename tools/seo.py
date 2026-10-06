#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Audit a page's search and sharing metadata, the way a crawler and a link preview read it. Standard library only.

Checks, each PASS, WARN or FAIL:
- the basics: charset, viewport, lang, a title and a description of a length search results show whole
- canonical: present, absolute, and (for a URL) answered 200 without a redirect, since a canonical that redirects
  is ignored
- Open Graph: title, description, type, url (equal to the canonical), site_name, and every og:image with its width,
  height, type and alt
- X/Twitter: card, title, description, image, image:alt (a large card needs a wide image)
- icons: rel=icon and apple-touch-icon present and reachable
- JSON-LD: every block parses, carries @context, and names the page's url somewhere
- each card image fetched: its real size read from the file (PNG, JPEG, GIF, WebP header) must equal the declared
  og:image:width/height; at least 200×200 (Facebook's floor), wide cards near 1.91:1, under 5 MB (X's limit)

usage: python3 tools/seo.py https://deltaverse.pythai.net/bankml
       python3 tools/seo.py page.html --base https://deltaverse.pythai.net/bankml   (a local file; fetches its images)
       python3 tools/seo.py page.html --offline                                     (no network: tags only)
exit 0 when nothing FAILs. Pairs with tools/makecards.py, which draws the cards."""
import argparse, html.parser, json, struct, sys, urllib.error, urllib.parse, urllib.request

UA = "Mozilla/5.0 (compatible; bankml-seo/1.0; +https://github.com/cryptoAGI/bankml)"


class Head(html.parser.HTMLParser):
    """Collects <html lang>, <title>, every <meta> and <link>, and the JSON-LD blocks."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lang, self.title, self.metas, self.links, self.ld = None, None, [], [], []
        self._in_title = self._in_ld = False
        self._buf = []

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "html":
            self.lang = a.get("lang")
        elif tag == "title":
            self._in_title, self._buf = True, []
        elif tag == "meta":
            self.metas.append(a)
        elif tag == "link":
            self.links.append(a)
        elif tag == "script" and a.get("type", "").lower() == "application/ld+json":
            self._in_ld, self._buf = True, []

    def handle_endtag(self, tag):
        if tag == "title" and self._in_title:
            self.title, self._in_title = "".join(self._buf).strip(), False
        elif tag == "script" and self._in_ld:
            self.ld.append("".join(self._buf))
            self._in_ld = False

    def handle_data(self, data):
        if self._in_title or self._in_ld:
            self._buf.append(data)

    def meta(self, key):
        """Every content for a name= or property= key, in page order."""
        return [m.get("content", "") for m in self.metas if (m.get("property") or m.get("name") or "").lower() == key]

    def first(self, key):
        v = self.meta(key)
        return v[0] if v else None

    def link(self, rel):
        return [l.get("href", "") for l in self.links if rel in l.get("rel", "").lower().split()]


def fetch(url, follow=True, limit=6 << 20):
    """(status, final url, headers, body) — `follow=False` reports a redirect instead of following it."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    opener = urllib.request.build_opener(*([] if follow else [NoRedirect()]))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with opener.open(req, timeout=30) as r:
            return r.status, r.geturl(), r.headers, r.read(limit)
    except urllib.error.HTTPError as e:
        return e.code, url, e.headers, b""


def image_size(b):
    """(kind, width, height) from the first bytes of a PNG, JPEG, GIF or WebP; None when unknown."""
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return ("png",) + struct.unpack(">II", b[16:24])
    if b[:6] in (b"GIF87a", b"GIF89a"):
        return ("gif",) + struct.unpack("<HH", b[6:10])
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        if b[12:16] == b"VP8X":
            return ("webp", 1 + int.from_bytes(b[24:27], "little"), 1 + int.from_bytes(b[27:30], "little"))
        if b[12:16] == b"VP8 ":
            w, h = struct.unpack("<HH", b[26:30])
            return ("webp", w & 0x3FFF, h & 0x3FFF)
        if b[12:16] == b"VP8L":
            v = int.from_bytes(b[21:25], "little")
            return ("webp", (v & 0x3FFF) + 1, ((v >> 14) & 0x3FFF) + 1)
    if b[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(b):
            if b[i] != 0xFF:
                i += 1
                continue
            marker = b[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", b[i + 5:i + 9])
                return ("jpeg", w, h)
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            i += 2 + struct.unpack(">H", b[i + 2:i + 4])[0]
    return None


class Report:
    def __init__(self):
        self.rows = []

    def add(self, level, what, detail=""):
        self.rows.append((level, what, detail))

    def ok(self, cond, what, detail="", level="FAIL"):
        self.add("PASS" if cond else level, what, detail)
        return cond

    def show(self):
        for level, what, detail in self.rows:
            print(f"{level:4}  {what}" + (f"  — {detail}" if detail else ""))
        n = {k: sum(1 for r in self.rows if r[0] == k) for k in ("PASS", "WARN", "FAIL")}
        print(f"\n{n['PASS']} pass, {n['WARN']} warn, {n['FAIL']} fail")
        return n["FAIL"] == 0


def audit(text, page_url, online):
    h = Head()
    h.feed(text)
    r = Report()
    absu = lambda u: urllib.parse.urljoin(page_url or "", u) if u else u
    # the basics
    r.ok(any(m.get("charset") for m in h.metas), "charset declared")
    r.ok(any((m.get("name") or "").lower() == "viewport" for m in h.metas), "viewport meta (mobile layout)")
    r.ok(bool(h.lang), "<html lang>", h.lang or "missing", level="WARN")
    t = h.title or ""
    r.ok(bool(t), "title", t[:80])
    if t:
        r.ok(15 <= len(t) <= 70, "title length 15–70 (shown whole in results)", f"{len(t)} characters", level="WARN")
    desc = h.first("description") or ""
    r.ok(bool(desc), "meta description")
    if desc:
        r.ok(70 <= len(desc) <= 160, "description length 70–160", f"{len(desc)} characters", level="WARN")
    r.ok(bool(h.first("robots")), "robots meta", h.first("robots") or "absent (defaults to index, follow)", level="WARN")
    r.ok(bool(h.first("theme-color")), "theme-color", level="WARN")
    # canonical
    canon = (h.link("canonical") or [None])[0]
    r.ok(bool(canon), "canonical link", canon or "")
    if canon:
        r.ok(canon.startswith("https://"), "canonical is absolute https")
        if online:
            st, _, hd, _ = fetch(canon, follow=False, limit=1)
            r.ok(st == 200, "canonical answers 200 without a redirect", f"{st} {hd.get('Location', '') if hd else ''}".strip())
    # Open Graph
    for k in ("og:title", "og:description", "og:type", "og:url", "og:site_name"):
        r.ok(bool(h.first(k)), k, (h.first(k) or "")[:90])
    if canon and h.first("og:url"):
        r.ok(h.first("og:url") == canon, "og:url equals the canonical", h.first("og:url"))
    imgs = h.meta("og:image")
    r.ok(bool(imgs), "og:image", f"{len(imgs)} image(s)")
    widths, heights, alts, types = h.meta("og:image:width"), h.meta("og:image:height"), h.meta("og:image:alt"), h.meta("og:image:type")
    r.ok(len(widths) == len(imgs) and len(heights) == len(imgs), "og:image:width/height for every image", level="WARN")
    r.ok(len(alts) >= len(imgs), "og:image:alt for every image", level="WARN")
    # X / Twitter
    card = h.first("twitter:card")
    r.ok(card in ("summary", "summary_large_image", "player", "app"), "twitter:card", card or "missing")
    for k in ("twitter:title", "twitter:description", "twitter:image"):
        r.ok(bool(h.first(k)) or bool(h.first(k.replace("twitter:", "og:"))), k, (h.first(k) or "(falls back to og)")[:90], level="WARN")
    r.ok(bool(h.first("twitter:image:alt")), "twitter:image:alt", level="WARN")
    # icons
    icons, touch = h.link("icon"), h.link("apple-touch-icon")
    r.ok(bool(icons), "rel=icon", ", ".join(icons))
    r.ok(bool(touch), "apple-touch-icon", ", ".join(touch), level="WARN")
    # JSON-LD
    r.ok(bool(h.ld), "JSON-LD present", f"{len(h.ld)} block(s)", level="WARN")
    for i, block in enumerate(h.ld):
        try:
            d = json.loads(block)
            r.ok("@context" in d, f"JSON-LD {i + 1} parses and has @context")
            if canon:
                r.ok(canon in block, f"JSON-LD {i + 1} names the canonical url", level="WARN")
        except ValueError as e:
            r.add("FAIL", f"JSON-LD {i + 1} parses", str(e))
    if not online:
        return r
    # fetch what a preview fetches
    for u in icons[:1] + touch[:1]:
        st, _, _, _ = fetch(absu(u), limit=1)
        r.ok(st == 200, f"icon reachable: {u}", str(st))
    declared = list(zip(imgs, widths + [None] * len(imgs), heights + [None] * len(imgs), types + [None] * len(imgs)))
    tw = h.first("twitter:image")
    if tw and tw not in imgs:
        declared.append((tw, None, None, None))
    for u, w, hh, ty in declared:
        st, final, hd, body = fetch(absu(u))
        if not r.ok(st == 200, f"image reachable: {u}", str(st)):
            continue
        size = image_size(body)
        if not r.ok(size is not None, f"image format readable: {u}"):
            continue
        kind, W, H = size
        if w and hh:
            r.ok((str(W), str(H)) == (w, hh), f"declared {w}×{hh} = real {W}×{H}", u)
        if ty:
            r.ok(ty.endswith(kind), f"og:image:type {ty} matches the file ({kind})", level="WARN")
        r.ok(W >= 200 and H >= 200, f"{W}×{H} at least 200×200", u)
        r.ok(len(body) < 5 << 20, "under 5 MB (X's limit)", f"{len(body) / 1024:.0f} KB")
        if W > H:
            r.ok(abs(W / H - 1.91) < 0.15, "wide card near 1.91:1 (1200×630)", f"{W / H:.2f}:1", level="WARN")
    if card == "summary_large_image" and declared:
        st, _, _, body = fetch(absu(tw or declared[0][0]))
        s = image_size(body) if st == 200 else None
        r.ok(bool(s) and s[1] > s[2], "summary_large_image uses a wide image", level="WARN")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("target", help="a page URL, or a local .html file")
    ap.add_argument("--base", help="for a local file: the URL it will be served at (resolves relative image and icon paths)")
    ap.add_argument("--offline", action="store_true", help="read the tags only; fetch nothing")
    a = ap.parse_args()
    if a.target.startswith(("http://", "https://")):
        st, final, _, body = fetch(a.target)
        if st != 200:
            sys.exit(f"{a.target}: HTTP {st}")
        if final != a.target:
            print(f"note: {a.target} redirects to {final}\n")
        text, url = body.decode("utf-8", "replace"), final
    else:
        text, url = open(a.target, encoding="utf-8").read(), a.base
    sys.exit(0 if audit(text, url, online=not a.offline and bool(url)).show() else 1)


if __name__ == "__main__":
    main()
