#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Draw a page's share cards: the images a link preview shows (Open Graph, X/Twitter, LinkedIn, Discord, Telegram,
Slack). Three files, deterministic:
  og-<name>.jpg          1200×630, the wide card (og:image, twitter:image with summary_large_image)
  og-<name>-square.jpg   1200×1200, for platforms that crop square
  <name>-touch-icon.png  180×180, the apple-touch-icon

The look is the DeltaVerse's: a triangulated mesh glowing cyan to violet on the landing's radial dark, the frame's
corner marks, and the DeltaVerse $ (the green of deltaverse.pythai.net/favicon.ico) drawn large as an outlined glyph.
The words are arguments, so a card says only what the page measured.

usage: python3 tools/makecards.py [--out DIR] [--name bankml] [--title bankML]
         [--tagline "line one|line two"] [--stat "bit-exact|vs llama.cpp"]... [--foot "text"] [--fonts DIR]
needs: Pillow (pip install pillow) and the Noto Sans fonts (fonts-noto-core on Debian/Ubuntu). Check the result with
tools/seo.py, which also confirms each card's real size matches what the page declares."""
import argparse, math, random
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
except ImportError:
    raise SystemExit("tools/makecards.py needs Pillow: pip install pillow")

GREEN, CYAN, VIOLET = (51, 190, 103), (34, 211, 238), (157, 78, 221)
DEFAULT_STATS = ["bit-exact|vs llama.cpp", "9.4–10×|ternary kernel", "0|dependencies"]


def mesh(w, h, seed, n):
    """A jittered grid, every cell cut on a diagonal: a triangulation with no long slivers."""
    rnd = random.Random(seed)
    cols, rows = n, max(3, round(n * h / w))
    jit = lambda i, last, span: rnd.uniform(-.38, .38) * span if 0 < i < last else 0
    pts = [[(c * w / (cols - 1) + jit(c, cols - 1, w / cols), r * h / (rows - 1) + jit(r, rows - 1, h / rows))
            for c in range(cols)] for r in range(rows)]
    tris = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            a, b, cc, d = pts[r][c], pts[r][c + 1], pts[r + 1][c], pts[r + 1][c + 1]
            tris += [(a, b, d), (a, d, cc)] if (r + c) % 2 else [(a, b, cc), (b, d, cc)]
    return tris


class Cards:
    def __init__(self, fonts: Path):
        self.fonts = fonts
        for f in ("NotoSans-Bold.ttf", "NotoSans-Light.ttf", "NotoSans-Regular.ttf", "NotoSans-SemiBold.ttf", "NotoSansMono-Regular.ttf"):
            if not (fonts / f).exists():
                raise SystemExit(f"{fonts / f} not found: install fonts-noto-core or pass --fonts DIR")

    def font(self, name, size):
        return ImageFont.truetype(str(self.fonts / name), size)

    def ground(self, w, h):
        g = Image.new("L", (w, h))
        gd = ImageDraw.Draw(g)
        cx, cy, R = w * .5, h * .5, math.hypot(w, h) * .55
        for i in range(60, 0, -1):
            gd.ellipse([cx - R * i / 60, cy - R * i / 60, cx + R * i / 60, cy + R * i / 60], fill=int(22 * (1 - i / 60) ** .7))
        img = Image.merge("RGB", (g.point(lambda v: v * .35), g.point(lambda v: v * .45), g.point(lambda v: v)))
        m = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        md = ImageDraw.Draw(m)
        for t in mesh(w, h, 1129, 15 if w > h else 11):
            x = sum(p[0] for p in t) / 3 / w
            col = tuple(int(CYAN[k] * (1 - x) + VIOLET[k] * x) for k in range(3))
            md.polygon(t, outline=col + (70,), width=1)
            md.polygon(t, fill=col + (int(10 + 18 * random.Random(int(x * 1e6)).random()),))
        glow = m.filter(ImageFilter.GaussianBlur(3))
        img.paste(glow, (0, 0), glow)
        img.paste(m, (0, 0), m)
        veil = Image.new("RGBA", (w, h), (4, 6, 14, 150))  # so the type reads over the mesh
        img.paste(veil, (0, 0), veil)
        return img

    def dollar(self, img, x, y, size):
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((x, y), "$", font=self.font("NotoSans-Bold.ttf", size), fill=GREEN + (255,), anchor="mm")
        glow = layer.filter(ImageFilter.GaussianBlur(16))
        img.paste(glow, (0, 0), glow)
        ImageDraw.Draw(img).text((x, y), "$", font=self.font("NotoSans-Bold.ttf", size), fill=(6, 22, 12), anchor="mm",
                                 stroke_width=max(4, size // 40), stroke_fill=GREEN)

    def card(self, w, h, title, tagline, stats, foot, path):
        img = self.ground(w, h)
        d = ImageDraw.Draw(img)
        square = h >= w
        self.dollar(img, *((w * .5, h * .27) if square else (w * .2, h * .5)), int(h * (.42 if square else .78)))
        tx, anchor = (w * .5, "ma") if square else (w * .40, "la")
        y = h * .52 if square else h * .17
        d.text((tx, y), title, font=self.font("NotoSans-Light.ttf", int(h * (.115 if square else .17))), fill=(255, 255, 255), anchor=anchor)
        y += h * (.14 if square else .22)
        for line in tagline:
            d.text((tx, y), line, font=self.font("NotoSans-Regular.ttf", int(h * (.036 if square else .052))), fill=(214, 226, 236), anchor=anchor)
            y += h * (.05 if square else .072)
        y += h * .03
        colw = (w * .82 if square else w * .55) / max(1, len(stats))
        x0 = (w - colw * len(stats)) / 2 if square else tx
        for i, (big, small) in enumerate(stats):
            x = x0 + colw * i
            d.text((x, y), big, font=self.font("NotoSans-SemiBold.ttf", int(h * (.045 if square else .062))), fill=CYAN, anchor="la")
            d.text((x, y + h * (.058 if square else .082)), small, font=self.font("NotoSans-Regular.ttf", int(h * (.026 if square else .036))),
                   fill=(160, 172, 190), anchor="la")
        d.text((w / 2, h * .94), foot, font=self.font("NotoSansMono-Regular.ttf", int(h * (.022 if square else .032))), fill=(150, 160, 178), anchor="mm")
        for ax, ay, sx, sy in [(14, 14, 1, 1), (w - 14, 14, -1, 1), (14, h - 14, 1, -1), (w - 14, h - 14, -1, -1)]:
            d.line([(ax, ay), (ax + 34 * sx, ay)], fill=CYAN, width=2)  # the landing's corner marks
            d.line([(ax, ay), (ax, ay + 34 * sy)], fill=CYAN, width=2)
        img.save(path, quality=90, optimize=True, progressive=True)
        print(f"{path}  {w}×{h}")

    def touch(self, path, size=180):
        img = Image.new("RGB", (size, size), (0, 0, 0))
        ImageDraw.Draw(img).text((size / 2, size * .53), "$", font=self.font("NotoSans-Bold.ttf", int(size * .92)), fill=(6, 22, 12),
                                 anchor="mm", stroke_width=max(3, size // 36), stroke_fill=GREEN)
        img.save(path)
        print(f"{path}  {size}×{size}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=".", help="directory to write into (default: here)")
    ap.add_argument("--name", default="bankml", help="file stem: og-<name>.jpg, og-<name>-square.jpg, <name>-touch-icon.png")
    ap.add_argument("--title", default="bankML")
    ap.add_argument("--tagline", default="Verified 1-bit and ternary language|models on the CPU you already have", help="lines split by |")
    ap.add_argument("--stat", action="append", help='"big|small", repeatable (default: the three measured claims)')
    ap.add_argument("--foot", default="deltaverse.pythai.net/bankml  ·  github.com/cryptoAGI/bankml  ·  Rust")
    ap.add_argument("--fonts", default="/usr/share/fonts/truetype/noto", help="directory holding the Noto Sans .ttf files")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = [tuple((s.split("|", 1) + [""])[:2]) for s in (a.stat or DEFAULT_STATS)]
    c = Cards(Path(a.fonts))
    tagline = a.tagline.split("|")
    c.card(1200, 630, a.title, tagline, stats, a.foot, out / f"og-{a.name}.jpg")
    c.card(1200, 1200, a.title, tagline, stats, a.foot, out / f"og-{a.name}-square.jpg")
    c.touch(out / f"{a.name}-touch-icon.png")


if __name__ == "__main__":
    main()
