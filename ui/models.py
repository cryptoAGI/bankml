"""bankml's model importer: the models Savante can speak through, brought in without friction and never unverified.

Three sources, one discipline:
  - a curated catalogue of open-source models (Bonsai 1-bit/ternary, Qwen3, SmolLM, Granite), each pinned here to
    the sha256 its publisher's repository lists at a fixed revision;
  - any Hugging Face GGUF by URL (https://huggingface.co/OWNER/REPO[/blob|resolve/REV/FILE]): the licence is read from
    the repository and must be open source, and the pin is the repository's own LFS sha256 at the resolved revision;
  - Ollama: search ollama.com, import from the registry (the layer digest is the GGUF's sha256), or adopt a model
    the local Ollama already holds, with no download at all.

Every import is streamed to disk, hashed on the way, and kept only if it equals the published sha256. Then bankml's
header guard (`bankml guard --json`) must say play, and a FORK.json record is written so `bankml serve` will pin it.
A model whose licence is not open source is refused, not hidden behind a warning ("open source or go away"); Gemma and
Llama are refused this way. Nothing is ever uploaded. The only network calls are GETs to huggingface.co, ollama.com
and registry.ollama.ai.

The carrier is `bankml serve MODEL --fork FORK.json --spawn llama-server`. `switch()` replaces it, and rolls back to
the previous model if the new one does not come up."""
import hashlib, json, os, re, shutil, signal, subprocess, threading, time, urllib.parse, urllib.request
from pathlib import Path

HOME = Path.home()
REPO = Path(os.environ.get("BANKML_REPO", Path(__file__).resolve().parents[1])).expanduser()
MODELS = Path(os.environ.get("BANKML_MODELS", REPO / ".models")).expanduser()
FORKS = Path(os.environ.get("BANKML_FORKS", HOME / ".local" / "share" / "bankml" / "forks")).expanduser()
LLAMA = Path(os.environ.get("BANKML_LLAMA_SERVER", HOME / "sAGI" / "bonsai" / "llama-b11192" / "llama-server")).expanduser()
BANKML = Path(os.environ.get("BANKML_BIN", REPO / "target" / "release" / "bankml")).expanduser()
OLLAMA_STORE = Path(os.environ.get("BANKML_OLLAMA_MODELS", "/usr/share/ollama/.ollama/models"))
LISTEN = os.environ.get("BANKML_SERVE_LISTEN", "127.0.0.1:18093")
UPSTREAM = os.environ.get("BANKML_UPSTREAM", "127.0.0.1:18092")
LOG = Path(os.environ.get("BANKML_UI_STATE", HOME / ".local" / "share" / "bankml" / "savante")).expanduser() / "carrier.log"
DISK_MARGIN = 1_500_000_000  # never fill the disk: keep 1.5 GB free after an import
UA = {"User-Agent": "bankml-importer (+https://github.com/cryptoAGI/bankml)"}

# OSI-approved licences (SPDX ids as Hugging Face tags them), plus public-domain dedications
OPEN = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "bsd", "isc", "mpl-2.0", "gpl-2.0", "gpl-3.0", "lgpl-2.1",
        "lgpl-3.0", "agpl-3.0", "unlicense", "cc0-1.0", "zlib", "artistic-2.0", "epl-2.0", "ecl-2.0", "upl-1.0", "0bsd"}

# the curated catalogue: sha256 and bytes as each repository lists them at `revision` (read 2026-09-28); an import
# re-reads the repository and refuses unless all three still agree
CATALOG = [
    {"id": "bonsai-8b", "title": "Bonsai-8B · 1-bit (Q1_0)", "repo": "PYTHAI/Bonsai-8B-gguf-fork", "file": "Bonsai-8B-Q1_0.gguf",
     "revision": "87c4ff9b04e32b7bc71d66d62a803f3e3aa002b4", "bytes": 1158654496, "sha256": "284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54", "licence": "apache-2.0", "default": True,
     "note": "Savante's default carrier: an 8B Qwen3 in 1.16 GB; bankml's Q1_0 kernel is bit-exact against it"},
    {"id": "ternary-bonsai-8b", "title": "Ternary-Bonsai-8B · ternary (Q2_0_g64)", "repo": "PYTHAI/Ternary-Bonsai-8B-gguf-fork",
     "file": "Ternary-Bonsai-8B-Q2_0_g64.gguf", "revision": "2445960052d465ace262a43b5f5fef52a0c1e1ef", "bytes": 2310125920, "sha256": "e17b298d84ee78797916ae5c2ecc8211469cc65cccfe3080cd9a9bb503fbc55e", "licence": "apache-2.0",
     "note": "the better answers of the two Bonsai-8B, slower in the reference runtime (TECHNICAL.md §I.2)"},
    {"id": "bonsai-4b", "title": "Bonsai-4B · 1-bit (Q1_0)", "repo": "prism-ml/Bonsai-4B-gguf", "file": "Bonsai-4B-Q1_0.gguf",
     "revision": "78f2c2bacd0904ffaba24b4873ed975e5818354a", "bytes": 572270624, "sha256": "4524b3f997f0f06444e568d1f26e2efd69effa3218c7ad3047432fb171e42168", "licence": "apache-2.0", "note": "small and quick"},
    {"id": "bonsai-1.7b", "title": "Bonsai-1.7B · 1-bit (Q1_0)", "repo": "prism-ml/Bonsai-1.7B-gguf", "file": "Bonsai-1.7B-Q1_0.gguf",
     "revision": "210a9e99f79cb184909d49595906526eb2b3dd9a", "bytes": 248302272, "sha256": "3d7c6c90dd98717a203adb22d5eacd2581850e40aa5327e144b97766cae5f7e3", "licence": "apache-2.0", "note": "the oracle's test model; a quarter gigabyte"},
    {"id": "qwen3-0.6b", "title": "Qwen3-0.6B · Q8_0", "repo": "Qwen/Qwen3-0.6B-GGUF", "file": "Qwen3-0.6B-Q8_0.gguf",
     "revision": "23749fefcc72300e3a2ad315e1317431b06b590a", "bytes": 639446688, "sha256": "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031", "licence": "apache-2.0", "note": "Qwen's own GGUF; the smallest Qwen3"},
    {"id": "qwen3-1.7b", "title": "Qwen3-1.7B · Q4_K_M", "repo": "ggml-org/Qwen3-1.7B-GGUF", "file": "Qwen3-1.7B-Q4_K_M.gguf",
     "revision": "daeb8e2d528a760970442092f6bf1e55c3b659eb", "bytes": 1282439264, "sha256": "d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5", "licence": "apache-2.0", "note": "a standard 4-bit K-quant"},
    {"id": "qwen3-4b", "title": "Qwen3-4B · Q4_K_M", "repo": "Qwen/Qwen3-4B-GGUF", "file": "Qwen3-4B-Q4_K_M.gguf",
     "revision": "bc640142c66e1fdd12af0bd68f40445458f3869b", "bytes": 2497280256, "sha256": "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5", "licence": "apache-2.0", "note": "Qwen's own GGUF"},
    {"id": "qwen3-8b", "title": "Qwen3-8B · Q4_K_M", "repo": "Qwen/Qwen3-8B-GGUF", "file": "Qwen3-8B-Q4_K_M.gguf",
     "revision": "7c41481f57cb95916b40956ab2f0b139b296d974", "bytes": 5027783488, "sha256": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785", "licence": "apache-2.0",
     "note": "the full-precision-quality 8B; needs about 5 GB of disk and memory"},
    {"id": "smollm2-1.7b", "title": "SmolLM2-1.7B-Instruct · Q4_K_M", "repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
     "file": "smollm2-1.7b-instruct-q4_k_m.gguf", "revision": "2d4a76a30b4af41ecd395c35725ac11688d4cfe4", "bytes": 1055609536, "sha256": "decd2598bc2c8ed08c19adc3c8fdd461ee19ed5708679d1c54ef54a5a30d4f33",
     "licence": "apache-2.0", "note": "Hugging Face's own small model, open data"},
    {"id": "smollm3-3b", "title": "SmolLM3-3B · Q4_K_M", "repo": "ggml-org/SmolLM3-3B-GGUF", "file": "SmolLM3-Q4_K_M.gguf",
     "revision": "4965cb60b150737b68a0408c36aeefb65078f894", "bytes": 1915305312, "sha256": "8334b850b7bd46238c16b0c550df2138f0889bf433809008cc17a8b05761863e", "licence": "apache-2.0", "note": "reasoning on or off (/no_think)"},
    {"id": "granite-3.3-2b", "title": "Granite-3.3-2B-Instruct · Q4_K_M", "repo": "ibm-granite/granite-3.3-2b-instruct-GGUF",
     "file": "granite-3.3-2b-instruct-Q4_K_M.gguf", "revision": "7cdf86ccd1f1bb3491c9b7017b033f2e51367397", "bytes": 1545303328, "sha256": "ac71e9e32c0bea919b409c5918f69ca74339854b0319c5065e4e9fb6d95c4852",
     "licence": "apache-2.0", "note": "IBM's open model, tuned for instructions and tools"},
]


# ── small helpers ──────────────────────────────────────────────────────────────────────────────────────────
def _get(url: str, timeout=30, raw=False):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        b = r.read()
    return b if raw else json.loads(b)


def licence_open(tag) -> bool:
    tags = tag if isinstance(tag, list) else [tag]
    return any(isinstance(t, str) and t.lower().removeprefix("license:") in OPEN for t in tags)


def free_bytes(p: Path = MODELS) -> int:
    p = p if p.exists() else p.parent
    return shutil.disk_usage(p.resolve()).free


def mem_total() -> int:
    try:
        return int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1)) * 1024
    except (OSError, AttributeError):
        return 0


def fits(nbytes: int, have: bool = False) -> tuple:
    """(ok, why): disk for the file plus the margin, and a model the memory can hold (weights ≲ 70 % of RAM)."""
    if not have and nbytes + DISK_MARGIN > free_bytes():
        return False, f"needs {nbytes / 1e9:.1f} GB + {DISK_MARGIN / 1e9:.1f} GB margin; {free_bytes() / 1e9:.1f} GB free on disk"
    mt = mem_total()
    if mt and nbytes > 0.7 * mt:
        return False, f"{nbytes / 1e9:.1f} GB of weights on a {mt / 1e9:.1f} GB machine would page to swap"
    return True, "fits"


def _safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-.")
    if not s.endswith(".gguf"):
        s += ".gguf"
    return s


# ── installed models and their pins ───────────────────────────────────────────────────────────────────────
def forks() -> dict:
    """basename -> (FORK.json path, record) for every pinned file in FORKS."""
    out = {}
    for f in sorted(FORKS.glob("*.json")):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for rec in j.get("files") or []:
            if isinstance(rec, dict) and str(rec.get("path", "")).endswith(".gguf") and re.fullmatch(r"[0-9a-f]{64}", str(rec.get("sha256", ""))):
                out[rec["path"]] = (f, {**rec, "source": j.get("source_repo") or j.get("source"), "licence": j.get("licence_tag")})
    return out


def installed() -> list:
    """[{file, path, bytes, pinned, fork, source, licence, catalog}] for every GGUF in MODELS."""
    pins, cat = forks(), {c["file"]: c for c in CATALOG}
    out = []
    for p in sorted(MODELS.glob("*.gguf")):
        try:
            size = p.stat().st_size
        except OSError:
            continue  # a dangling link
        fk = pins.get(p.name)
        out.append({"file": p.name, "path": str(p), "bytes": size, "pinned": bool(fk), "fork": str(fk[0]) if fk else None,
                    "source": (fk[1]["source"] if fk else None) or (cat.get(p.name) or {}).get("repo"),
                    "licence": (fk[1]["licence"] if fk else None) or (cat.get(p.name) or {}).get("licence"), "catalog": (cat.get(p.name) or {}).get("id")})
    return out


def write_fork(file: str, nbytes: int, sha: str, source: str, url: str, revision: str, licence: str, pinned_from: str) -> Path:
    FORKS.mkdir(parents=True, exist_ok=True)
    f = FORKS / f"{file}.FORK.json"
    f.write_text(json.dumps({"kind": "bankml import (weights downloaded, verified against the published sha256)", "source_repo": source,
                             "source_url": url, "source_revision": revision, "licence_tag": licence,
                             "imported_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pinned_from": pinned_from,
                             "files": [{"path": file, "bytes": nbytes, "sha256": sha}]}, indent=1) + "\n", encoding="utf-8")
    return f


def guard(path: Path) -> dict:
    """bankml's header guard on a file: {verdict, arch, types, reasons}."""
    p = subprocess.run([str(BANKML), "guard", str(path), "--json"], capture_output=True, text=True, timeout=120)
    try:
        return json.loads(p.stdout)
    except ValueError:
        return {"verdict": "refuse", "reasons": [(p.stderr or p.stdout).strip()[-300:] or "bankml guard gave no report"]}


# ── Hugging Face ──────────────────────────────────────────────────────────────────────────────────────────
def hf_parse(url: str) -> tuple:
    """(repo, revision|None, file|None) from a Hugging Face URL or an OWNER/REPO id."""
    u = url.strip()
    u = re.sub(r"^hf://", "", u)
    u = re.sub(r"^https?://(www\.)?(huggingface\.co|hf\.co)/", "", u)
    u = u.split("?")[0].split("#")[0].strip("/")
    parts = u.split("/")
    if len(parts) < 2 or not all(re.fullmatch(r"[A-Za-z0-9._-]+", x) for x in parts[:2]):
        raise ValueError("not a Hugging Face model: give https://huggingface.co/OWNER/REPO (optionally …/blob/REV/FILE.gguf)")
    repo = "/".join(parts[:2])
    if len(parts) >= 4 and parts[2] in ("blob", "resolve", "tree"):
        rev, f = parts[3], "/".join(parts[4:]) or None
        return repo, urllib.parse.unquote(rev), urllib.parse.unquote(f) if f else None
    return repo, None, None


def hf_resolve(url: str) -> dict:
    """{repo, revision, licence, open, files:[{file, bytes, sha256}], chosen} for a Hugging Face URL."""
    repo, rev, file = hf_parse(url)
    m = _get(f"https://huggingface.co/api/models/{repo}" + (f"/revision/{urllib.parse.quote(rev)}" if rev else ""))
    sha = m.get("sha") or rev
    lic = (m.get("cardData") or {}).get("license") or [t for t in m.get("tags", []) if t.startswith("license:")]
    tree = _get(f"https://huggingface.co/api/models/{repo}/tree/{sha}?recursive=true")
    files = [{"file": e["path"], "bytes": e.get("size", 0), "sha256": (e.get("lfs") or {}).get("oid")}
             for e in tree if e.get("type") == "file" and e["path"].endswith(".gguf") and (e.get("lfs") or {}).get("oid")]
    if file and not any(x["file"] == file for x in files):
        raise ValueError(f"{file} is not a GGUF in {repo}@{sha[:10]}")
    return {"repo": repo, "revision": sha, "licence": lic, "open": licence_open(lic), "files": files, "chosen": file,
            "gated": bool(m.get("gated"))}


# ── Ollama ────────────────────────────────────────────────────────────────────────────────────────────────
def _ollama_name(name: str) -> tuple:
    n = name.strip().removeprefix("https://ollama.com/").removeprefix("library/")
    n, _, tag = n.partition(":")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*(/[a-z0-9][a-z0-9._-]*)?", n):
        raise ValueError(f"not an Ollama model name: {name}")
    return n, tag or "latest"


def ollama_search(q: str) -> list:
    """ollama.com's search, as [{name, description, sizes, pulls}] (its HTML; there is no JSON search API)."""
    import html
    page = _get("https://ollama.com/search?q=" + urllib.parse.quote(q.strip()), raw=True).decode("utf-8", "replace")
    out = []
    for li in re.findall(r"<li[^>]*>(.*?)</li>", page, flags=re.S):
        m = re.search(r'href="/library/([^"/:]+)"', li)
        if not m:
            continue
        d = re.search(r"<p[^>]*>([^<]{8,})</p>", li)
        sizes = re.findall(r"text-blue-600[^>]*>\s*([0-9.]+[mbk]|[0-9x.]+b)\s*<", li)
        pulls = re.search(r"<span\s*>([0-9.]+[KMB]?)</span>\s*<span[^>]*>&nbsp;Pulls", li)
        cloud = "cloud" in li and ">cloud<" in li
        out.append({"name": m.group(1), "description": html.unescape(d.group(1).strip()) if d else "", "sizes": sizes,
                    "pulls": pulls.group(1) if pulls else "", "cloud": cloud})
    return out


def ollama_tags(name: str) -> list:
    n, _ = _ollama_name(name)
    page = _get(f"https://ollama.com/library/{n}/tags", raw=True).decode("utf-8", "replace")
    return sorted(set(re.findall(rf'href="/library/{re.escape(n)}:([^"]+)"', page)))


def _licence_of_text(t: str) -> str | None:
    s = t[:4000].lower()
    if "gemma terms of use" in s:
        return "gemma (not open source)"
    if "llama" in s and "community license" in s:
        return "llama community licence (not open source)"
    if "apache license" in s and "version 2.0" in s:
        return "apache-2.0"
    if "mit license" in s or "permission is hereby granted, free of charge" in s:
        return "mit"
    if "gnu general public license" in s:
        return "gpl-3.0" if "version 3" in s else "gpl-2.0"
    if "redistribution and use in source and binary forms" in s:
        return "bsd-3-clause"
    return None


def ollama_resolve(name: str) -> dict:
    """{repo, tag, file, bytes, sha256, licence, open, url} from the Ollama registry (manifest + licence layer)."""
    n, tag = _ollama_name(name)
    path = n if "/" in n else f"library/{n}"
    req = urllib.request.Request(f"https://registry.ollama.ai/v2/{path}/manifests/{tag}",
                                 headers={**UA, "Accept": "application/vnd.docker.distribution.manifest.v2+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        man = json.loads(r.read())
    layers = man.get("layers") or []
    model = [l for l in layers if l.get("mediaType") == "application/vnd.ollama.image.model"]
    if not model:
        raise ValueError(f"{n}:{tag} has no model weights in the registry (a cloud-only tag?)")
    lic_layer = [l for l in layers if l.get("mediaType") == "application/vnd.ollama.image.license"]
    lic = None
    if lic_layer:
        lic = _licence_of_text(_get(f"https://registry.ollama.ai/v2/{path}/blobs/{lic_layer[0]['digest']}", raw=True).decode("utf-8", "replace"))
    d = model[0]["digest"].removeprefix("sha256:")
    return {"repo": f"ollama:{path}", "tag": tag, "file": _safe_name(f"{n.replace('/', '-')}-{tag}"), "bytes": model[0]["size"], "sha256": d,
            "licence": lic or "unrecognised", "open": bool(lic and licence_open(lic)), "url": f"https://registry.ollama.ai/v2/{path}/blobs/sha256:{d}",
            "local": str(OLLAMA_STORE / "blobs" / f"sha256-{d}") if (OLLAMA_STORE / "blobs" / f"sha256-{d}").is_file() else None}


def ollama_local() -> list:
    """Models the local Ollama already holds: [{name, tag, sha256, bytes, blob}]."""
    out = []
    root = OLLAMA_STORE / "manifests" / "registry.ollama.ai"
    for f in sorted(root.glob("*/*/*")) if root.is_dir() else []:
        try:
            man = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for l in man.get("layers") or []:
            if l.get("mediaType") == "application/vnd.ollama.image.model":
                blob = OLLAMA_STORE / "blobs" / l["digest"].replace(":", "-")
                if blob.is_file():
                    ns, name = f.parent.parent.name, f.parent.name
                    out.append({"name": name if ns == "library" else f"{ns}/{name}", "tag": f.name, "sha256": l["digest"].removeprefix("sha256:"),
                                "bytes": l["size"], "blob": str(blob)})
    return out


# ── the import job (one at a time; the UI polls JOB) ──────────────────────────────────────────────────────
JOB = {"state": "idle", "what": "", "done": 0, "total": 0, "error": None, "result": None, "started": None}
_LOCK = threading.Lock()


def _hash_file(p: Path, on=None) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
            if on:
                on(len(b))
    return h.hexdigest()


def _download(url: str, dest: Path, nbytes: int, want: str) -> None:
    """Stream url → dest.part, hashing on the way; rename only if the sha256 is the published one. Resumes a .part."""
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    have = part.stat().st_size if part.exists() else 0
    if have > nbytes:
        part.unlink()
        have = 0
    if have:  # resume: hash what is there first
        with open(part, "rb") as f:
            while b := f.read(1 << 22):
                h.update(b)
    JOB["done"] = have
    req = urllib.request.Request(url, headers={**UA, **({"Range": f"bytes={have}-"} if have else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:  # the server ignored the range: start over
            have = 0
            h = hashlib.sha256()
            JOB["done"] = 0
        with open(part, "ab" if have else "wb") as f:
            while b := r.read(1 << 20):
                if JOB.get("cancel"):
                    raise RuntimeError("cancelled (the partial file is kept; importing again resumes it)")
                f.write(b)
                h.update(b)
                JOB["done"] += len(b)
    got = h.hexdigest()
    if got != want:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 {got[:16]}… is not the published {want[:16]}…: discarded")
    os.replace(part, dest)


def import_spec(spec: dict) -> dict:
    """Bring one model in. spec: {file, bytes, sha256, url, repo, revision, licence, pinned_from, local?}."""
    if not licence_open(spec.get("licence")):
        raise PermissionError(f"{spec.get('repo')}: licence {spec.get('licence')!r} is not open source; bankml imports open-source models only")
    if not re.fullmatch(r"[0-9a-f]{64}", spec.get("sha256") or ""):
        raise ValueError("no published sha256 for this file: nothing to pin it to, refused")
    dest = MODELS / _safe_name(spec["file"])
    ok, why = fits(spec["bytes"], have=dest.exists() or bool(spec.get("local")))
    if not ok and not (dest.exists() or spec.get("local")):
        raise RuntimeError(why)
    MODELS.mkdir(parents=True, exist_ok=True)
    JOB.update(total=spec["bytes"], done=0)
    if dest.exists():
        JOB["what"] = f"checking {dest.name} already here"
        if _hash_file(dest, lambda n: JOB.__setitem__("done", JOB["done"] + n)) != spec["sha256"]:
            raise RuntimeError(f"{dest.name} exists but is not the published file (sha256 differs); move it aside first")
    elif spec.get("local"):  # the local Ollama already has it: link, then hash
        JOB["what"] = f"adopting {spec['file']} from the local Ollama (no download)"
        if _hash_file(Path(spec["local"]), lambda n: JOB.__setitem__("done", JOB["done"] + n)) != spec["sha256"]:
            raise RuntimeError("the local Ollama blob does not hash to its digest: refused")
        os.symlink(spec["local"], dest)
    else:
        JOB["what"] = f"downloading {spec['file']} from {spec['repo']}"
        _download(spec["url"], dest, spec["bytes"], spec["sha256"])
    g = guard(dest)
    if g.get("verdict") != "play":
        dest.unlink(missing_ok=True)  # a symlink goes; the Ollama blob it points at stays
        raise RuntimeError(f"bankml's guard refused {dest.name}: {'; '.join(g.get('reasons') or [])}")
    fk = write_fork(dest.name, spec["bytes"], spec["sha256"], spec["repo"], spec.get("page") or spec["url"], spec.get("revision", ""),
                    spec["licence"] if isinstance(spec["licence"], str) else ",".join(spec["licence"]), spec["pinned_from"])
    return {"file": dest.name, "path": str(dest), "fork": str(fk), "arch": g.get("arch"), "types": g.get("types")}


def spec_catalog(cid: str) -> dict:
    c = next(x for x in CATALOG if x["id"] == cid)
    r = hf_resolve(f"https://huggingface.co/{c['repo']}/blob/{c['revision']}/{c['file']}")
    f = next(x for x in r["files"] if x["file"] == c["file"])
    if not (f["sha256"] == c["sha256"] and r["revision"] == c["revision"] and f["bytes"] == c["bytes"]):
        raise RuntimeError(f"{c['repo']} no longer lists the catalogued file (sha256/size/revision changed): refused")
    return spec_hf(r, c["file"])


def spec_hf(r: dict, file: str) -> dict:
    f = next(x for x in r["files"] if x["file"] == file)
    lic = r["licence"] if isinstance(r["licence"], str) else ",".join(t.removeprefix("license:") for t in r["licence"])
    return {"file": Path(file).name, "bytes": f["bytes"], "sha256": f["sha256"], "repo": r["repo"], "revision": r["revision"], "licence": lic,
            "url": f"https://huggingface.co/{r['repo']}/resolve/{r['revision']}/{urllib.parse.quote(file)}",
            "page": f"https://huggingface.co/{r['repo']}/blob/{r['revision']}/{file}", "pinned_from": f"Hugging Face LFS sha256 of {file} at {r['revision']}"}


def spec_ollama(o: dict) -> dict:
    return {"file": o["file"], "bytes": o["bytes"], "sha256": o["sha256"], "repo": o["repo"], "revision": o["tag"], "licence": o["licence"],
            "url": o["url"], "page": f"https://ollama.com/{o['repo'].removeprefix('ollama:').removeprefix('library/')}:{o['tag']}", "local": o.get("local"),
            "pinned_from": "Ollama registry layer digest (sha256 of the GGUF)"}


def start_job(what: str, fn, *a) -> bool:
    """Run fn(*a) in the background as the one import/switch job; False if one is already running."""
    with _LOCK:
        if JOB["state"] == "running":
            return False
        JOB.update(state="running", what=what, done=0, total=0, error=None, result=None, started=time.time(), cancel=False)

    def run():
        try:
            JOB["result"] = fn(*a)
            JOB["state"] = "done"
        except Exception as e:  # noqa: BLE001
            JOB.update(state="error", error=f"{type(e).__name__}: {e}")
    threading.Thread(target=run, daemon=True).start()
    return True


# ── the carrier: bankml serve + llama-server ──────────────────────────────────────────────────────────────
def serve_status(timeout=3) -> dict:
    try:
        with urllib.request.urlopen(f"http://{LISTEN}/bankml", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _listeners() -> dict:
    """port -> pid for the carrier's two ports (from ss), only if the process is llama-server or bankml."""
    out = {}
    try:
        s = subprocess.run(["ss", "-ltnpH"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return out
    for port in (LISTEN.rsplit(":", 1)[1], UPSTREAM.rsplit(":", 1)[1]):
        m = re.search(rf"127\.0\.0\.1:{port}\s.*?pid=(\d+)", s)
        if m:
            pid = int(m.group(1))
            try:
                exe = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[0].decode()
            except OSError:
                continue
            if Path(exe).name in ("llama-server", "bankml"):
                out[port] = pid
    return out


def _stop_carrier():
    pids = set(_listeners().values())
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for _ in range(100):
        if not _listeners():
            return
        time.sleep(0.2)
    for pid in set(_listeners().values()):
        os.kill(pid, signal.SIGKILL)
    time.sleep(0.5)


def _start_carrier(model: Path, fork: Path, threads=3, ctx=4096, wait=900) -> dict:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG, "ab")
    at = log.seek(0, 2)
    log.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} bankml serve {model.name}\n".encode())
    log.flush()
    subprocess.Popen([str(BANKML), "serve", str(model), "--fork", str(fork), "--spawn", str(LLAMA), "--upstream", UPSTREAM, "--listen", LISTEN,
                      "--threads", str(threads), "--ctx", str(ctx)], stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(REPO))
    t0 = time.time()
    while time.time() - t0 < wait:
        st = serve_status(2)
        if "verified" in st:
            return st
        if not _listeners() and time.time() - t0 > 20:  # both gone: it refused or crashed
            break
        time.sleep(1)
    with open(LOG, "rb") as f:
        f.seek(at)
        said = [l for l in f.read().decode(errors="replace").splitlines()[1:] if l.strip()]
    raise RuntimeError(f"bankml serve did not come up with {model.name}: " + (" / ".join(said[-3:])[-400:] or "no output"))


def switch(file: str, busy=lambda: False) -> dict:
    """Make `file` (in MODELS, pinned) the carrier. Rolls back to the previous model if the new one fails."""
    if busy():
        raise RuntimeError("an answer is being written; switch when it is done")
    p = MODELS / file
    pins = forks()
    if file not in pins:
        adopt(file)
        pins = forks()
    prev = serve_status()  # found again by its verified sha256: serve reports canonical paths (a symlink's target)
    prev_sha = (prev.get("verified") or {}).get("model_sha256")
    prev_file = next((f for f, (_, rec) in pins.items() if rec["sha256"] == prev_sha and (MODELS / f).exists()), None)
    JOB["what"] = f"stopping the carrier, starting {file} (verifies the whole file, then loads it)"
    _stop_carrier()
    try:
        return _start_carrier(p, pins[file][0])
    except Exception:
        if prev_file and prev_file != file:
            JOB["what"] = f"{file} failed; restoring {prev_file}"
            _stop_carrier()
            _start_carrier(MODELS / prev_file, pins[prev_file][0])
        raise


def adopt(file: str) -> Path:
    """Pin a file that is already here but has no FORK.json, if the catalogue knows it: hash it and compare to the
    repository's published sha256 (fetched now, not trusted from the catalogue's prefix alone)."""
    c = next((x for x in CATALOG if x["file"] == file), None)
    if not c:
        raise RuntimeError(f"{file} has no FORK.json pin and is not in the catalogue: import it from its source to pin it")
    s = spec_catalog(c["id"])
    JOB.update(what=f"hashing {file} to pin it", total=s["bytes"], done=0)
    if _hash_file(MODELS / file, lambda n: JOB.__setitem__("done", JOB["done"] + n)) != s["sha256"]:
        raise RuntimeError(f"{file} is not the published file (sha256 differs): refused")
    return write_fork(file, s["bytes"], s["sha256"], s["repo"], s["page"], s["revision"], s["licence"], s["pinned_from"])


def first_run(busy=lambda: False) -> dict:
    """Seamless start: if no carrier answers, bring in Bonsai-8B (download only if absent) and start it."""
    st = serve_status()
    if "verified" in st:
        return st
    c = next(x for x in CATALOG if x.get("default"))
    if not (MODELS / c["file"]).exists():
        import_spec(spec_catalog(c["id"]))
    return switch(c["file"], busy)


if __name__ == "__main__":  # python3 ui/models.py [list | catalog | import ID|URL|ollama:NAME:TAG | use FILE | first-run | search Q]
    import sys
    a = sys.argv[1:] or ["list"]
    if a[0] == "list":
        for m in installed():
            print(f"{'pinned ' if m['pinned'] else 'UNPINNED'} {m['bytes'] / 1e9:5.2f} GB  {m['file']}  ({m['source'] or '?'}, {m['licence'] or '?'})")
        print("carrier:", Path(serve_status().get("model", "none")).name)
    elif a[0] == "catalog":
        for c in CATALOG:
            print(f"{c['id']:18} {c['bytes'] / 1e9:5.2f} GB  {c['title']}  [{fits(c['bytes'], (MODELS / c['file']).exists())[1]}]")
    elif a[0] == "search":
        for r in ollama_search(" ".join(a[1:])):
            print(f"{r['name']:24} {','.join(r['sizes']):20} {r['description'][:80]}")
    elif a[0] == "import":
        t = a[1]
        spec = (spec_ollama(ollama_resolve(t.removeprefix("ollama:"))) if t.startswith("ollama:") else
                spec_catalog(t) if any(c["id"] == t for c in CATALOG) else None)
        if spec is None:
            r = hf_resolve(t)
            spec = spec_hf(r, r["chosen"] or (sys.exit("pick a file: " + ", ".join(f["file"] for f in r["files"])) if len(r["files"]) != 1 else r["files"][0]["file"]))
        print(json.dumps(import_spec(spec), indent=1))
    elif a[0] == "use":
        print(json.dumps(switch(a[1]), indent=1))
    elif a[0] == "first-run":
        print(json.dumps(first_run(), indent=1))
