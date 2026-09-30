# SPDX-License-Identifier: MIT OR Apache-2.0
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
EMBEDDING_ARCHS = {"bert", "nomic-bert", "jina-bert-v2", "xlm-roberta", "modern-bert", "neo-bert", "t5encoder"}  # encoders: they embed, they do not chat
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
    # coders (the codephreak and simplecoder agents): open-source only, so StarCoder and WizardCoder are not here
    # (OpenRAIL-M / Llama 2 licences). The newest coder, Qwen3-Coder-Next (80B, Apache-2.0), ships as four files of
    # 48 GB in all; it is named in those agents' .model as the latest, not offered for import here.
    {"id": "qwen2.5-coder-1.5b", "title": "Qwen2.5-Coder-1.5B-Instruct · Q4_K_M · coder", "repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
     "file": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "revision": "f86cb2c1fa58255f8052cc32aeede1b7482d4361", "bytes": 1117320768,
     "sha256": "cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046", "licence": "apache-2.0", "coder": True,
     "note": "the coder agents' default on a small machine: Qwen's own GGUF, 1.1 GB"},
    {"id": "qwen2.5-coder-7b", "title": "Qwen2.5-Coder-7B-Instruct · Q4_K_M · coder", "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
     "file": "qwen2.5-coder-7b-instruct-q4_k_m.gguf", "revision": "13fb94bfda8c8cf22497dc57b78f391a9acb426a", "bytes": 4683073536,
     "sha256": "509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c", "licence": "apache-2.0", "coder": True,
     "note": "the stronger small coder; needs about 5 GB of disk and memory"},
    {"id": "qwen3.8-27b", "title": "Qwen3.8-27B · UD-Q4_K_M · latest Qwen (code and general)", "repo": "unsloth/Qwen3.8-27B-GGUF",
     "file": "Qwen3.8-27B-UD-Q4_K_M.gguf", "revision": "4ca720788d1e01f1bff70c033e0d0028fd02e502", "bytes": 16464440224,
     "sha256": "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482", "licence": "apache-2.0", "coder": True,
     "note": "the newest Qwen (Aug 2026), strong at code; 16.5 GB, for a machine with 24 GB or more"},
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
    """(ok, why): disk for the file plus the margin (skipped when the file is already here), and a model the memory
    can hold (weights ≲ 70 % of RAM; always checked)."""
    if not have and nbytes + DISK_MARGIN > free_bytes():
        return False, f"needs {nbytes / 1e9:.1f} GB + {DISK_MARGIN / 1e9:.1f} GB margin; {free_bytes() / 1e9:.1f} GB free on disk"
    return fits_memory(nbytes)


def fits_memory(nbytes: int) -> tuple:
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
    seg = lambda x: bool(re.fullmatch(r"[A-Za-z0-9._-]+", x)) and x not in (".", "..")
    if len(parts) < 2 or not all(seg(x) for x in parts[:2]):
        raise ValueError("not a Hugging Face model: give https://huggingface.co/OWNER/REPO (optionally …/blob/REV/FILE.gguf)")
    repo = "/".join(parts[:2])
    if len(parts) >= 4 and parts[2] in ("blob", "resolve", "tree"):
        rev, f = urllib.parse.unquote(parts[3]), urllib.parse.unquote("/".join(parts[4:])) if parts[4:] else None
        if not re.fullmatch(r"[A-Za-z0-9._-]+", rev) or rev in (".", "..") or (f and any(x in ("", ".", "..") for x in f.split("/"))):
            raise ValueError("a revision is a branch, tag or commit; a file path has no '.' or '..' segments")
        return repo, rev, f
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
            "gated": bool(m.get("gated"))}  # gated repositories need an account and an accepted agreement: not imported


# ── Ollama ────────────────────────────────────────────────────────────────────────────────────────────────
def _ollama_name(name: str) -> tuple:
    n = name.strip().removeprefix("https://ollama.com/").removeprefix("library/")
    n, _, tag = n.partition(":")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*(/[a-z0-9][a-z0-9._-]*)?", n) or ".." in n:
        raise ValueError(f"not an Ollama model name: {name}")
    if tag and (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag) or ".." in tag):
        raise ValueError(f"not an Ollama tag: {tag}")
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


def ollama_local_spec(name: str) -> dict:
    """The same record as ollama_resolve(), built from the local Ollama's own manifest and licence layer: offline, and
    pinned to what was actually pulled (the registry's tag may have moved since)."""
    n, tag = _ollama_name(name)
    ns, base = (n.split("/", 1) if "/" in n else ("library", n))
    f = OLLAMA_STORE / "manifests" / "registry.ollama.ai" / ns / base / tag
    man = json.loads(f.read_text())
    layers = man.get("layers") or []
    model = [l for l in layers if l.get("mediaType") == "application/vnd.ollama.image.model"]
    if not model:
        raise ValueError(f"{n}:{tag}: the local manifest has no model layer")
    blob = OLLAMA_STORE / "blobs" / model[0]["digest"].replace(":", "-")
    lic_layer = [l for l in layers if l.get("mediaType") == "application/vnd.ollama.image.license"]
    lic = _licence_of_text((OLLAMA_STORE / "blobs" / lic_layer[0]["digest"].replace(":", "-")).read_text(errors="replace")) if lic_layer else None
    d = model[0]["digest"].removeprefix("sha256:")
    return {"repo": f"ollama:{ns}/{base}", "tag": tag, "file": _safe_name(f"{n.replace('/', '-')}-{tag}"), "bytes": model[0]["size"], "sha256": d,
            "licence": lic or "unrecognised", "open": bool(lic and licence_open(lic)), "url": "", "local": str(blob) if blob.is_file() else None}


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
JOB = {"state": "idle", "what": "", "done": 0, "total": 0, "error": None, "result": None, "started": None, "seq": 0, "cancel": False}
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
    """Stream url → dest.part, hashing on the way; rename only if the sha256 is the published one. Resumes a .part;
    never writes past the published size."""
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
    if have < nbytes:
        req = urllib.request.Request(url, headers={**UA, **({"Range": f"bytes={have}-"} if have else {})})
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get("Content-Range") or ""
            if have and (r.status != 206 or not cr.startswith(f"bytes {have}-")):  # the range was ignored or misplaced: start over
                have = 0
                h = hashlib.sha256()
                JOB["done"] = 0
            with open(part, "ab" if have else "wb") as f:
                while b := r.read(1 << 20):
                    if JOB.get("cancel"):
                        raise RuntimeError("cancelled (the partial file is kept; importing again resumes it)")
                    if JOB["done"] + len(b) > nbytes:
                        f.close()
                        part.unlink(missing_ok=True)
                        raise RuntimeError(f"the source sent more than the published {nbytes} bytes: discarded")
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
    if not ok:
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
    return {"file": _safe_name(file.replace("/", "-")), "bytes": f["bytes"], "sha256": f["sha256"], "repo": r["repo"], "revision": r["revision"], "licence": lic,
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
        finally:
            JOB["seq"] += 1  # the UI refreshes its lists when this moves
    threading.Thread(target=run, daemon=True).start()
    return True


def cancel_job() -> bool:
    """Ask a running download to stop at its next chunk (the .part is kept for resume)."""
    if JOB["state"] == "running":
        JOB["cancel"] = True
        return True
    return False


# ── the carrier: bankml serve + llama-server ──────────────────────────────────────────────────────────────
def serve_status(timeout=3) -> dict:
    try:
        with urllib.request.urlopen(f"http://{LISTEN}/bankml", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _listeners() -> dict:
    """port -> pid for the carrier's two ports (from ss), only if the process is this user's llama-server or bankml.
    Matches the configured host (127.0.0.1, 0.0.0.0, [::1] …) exactly."""
    out = {}
    try:
        s = subprocess.run(["ss", "-ltnpH"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return out
    for addr in (LISTEN, UPSTREAM):
        host, port = addr.rsplit(":", 1)
        hosts = {host, f"[{host.strip('[]')}]", "*"} if host in ("0.0.0.0", "::", "[::]") else {host, f"[{host.strip('[]')}]"}
        for line in s.splitlines():
            cols = line.split()
            h, _, pt = cols[3].rpartition(":") if len(cols) >= 4 else ("", "", "")
            if pt != port or h not in hosts:
                continue
            m = re.search(r"pid=(\d+)", line)
            if not m:
                continue
            pid = int(m.group(1))
            try:
                exe = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[0].decode()
                mine = os.stat(f"/proc/{pid}").st_uid == os.getuid()
            except OSError:
                continue
            if mine and Path(exe).name in ("llama-server", "bankml"):
                out[port] = pid
    return out


def _kill(pid: int, sig) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def _stop_carrier():
    for pid in set(_listeners().values()):
        _kill(pid, signal.SIGTERM)
    for _ in range(100):
        if not _listeners():
            return
        time.sleep(0.2)
    for pid in set(_listeners().values()):
        _kill(pid, signal.SIGKILL)
    time.sleep(0.5)
    if _listeners():
        raise RuntimeError(f"the carrier's ports are still held by {sorted(set(_listeners().values()))}; stop them by hand")


CTX = int(os.environ.get("BANKML_CTX", "2048"))  # the engine's context: 2048 keeps an 8B model's KV cache near 0.3 GB
THREADS = int(os.environ.get("BANKML_THREADS_SERVE", "3"))
RESOURCES = LOG.parent / "resources.json"
SLOTS = LOG.parent / "slots"  # the engine's saved KV slots (--slot-save-path): a restart restores the system prompt  # the operator's CPU and RAM choice (the Resources sliders), used by every start
OVERHEAD = 250_000_000  # llama-server's compute buffers and runtime beside weights and KV (measured order of magnitude)
CTX_MIN, CTX_MAX = 512, 32768


def resources() -> dict:
    """{"threads", "ctx", "ram_gb", "spec_ngram", "engine"}: the saved choice, else the defaults (BANKML_THREADS_SERVE,
    BANKML_CTX). engine: "auto" (bankML's own forward pass for the ternary Qwen3 files, where it is about 8x
    llama-server; llama-server otherwise), "native" or "llama.cpp"."""
    r = {"threads": THREADS, "ctx": CTX, "ram_gb": None, "spec_ngram": False, "engine": "auto"}
    try:
        r.update({k: v for k, v in json.loads(RESOURCES.read_text()).items() if k in r})
    except (OSError, ValueError):
        pass
    return r


def plan(model: Path, ram_gb: float) -> dict:
    """What a RAM budget buys for a model: weights stay resident, the rest is KV cache (f16, the guard's
    bytes-per-token for this architecture) after the engine's overhead. ctx is rounded down to 256."""
    g = guard(model)
    kv = int(g.get("kv_f16_bytes_per_token") or 0) or 147_456  # Qwen3-8B's, if the guard cannot say
    w = model.stat().st_size
    room = int(ram_gb * 1e9) - w - OVERHEAD
    ctx = max(0, min(CTX_MAX, room // kv // 256 * 256))
    return {"ctx": ctx, "fits": ctx >= CTX_MIN, "kv_gb": round(ctx * kv / 1e9, 2), "weights_gb": round(w / 1e9, 2),
            "kv_bytes_per_token": kv, "overhead_gb": OVERHEAD / 1e9, "arch": g.get("arch"),
            "min_ram_gb": round((w + OVERHEAD + CTX_MIN * kv) / 1e9, 2)}


def usage() -> dict:
    """What the carrier uses now, from bankml itself (`GET /bankml/usage`, sys.rs reading /proc — bankml's psutil,
    no crates); the Python reading below is the fallback for a serve older than 0.1.8."""
    try:
        with urllib.request.urlopen(f"http://{LISTEN}/bankml/usage", timeout=5) as r:
            u = json.loads(r.read())
        return {"pids": [p["pid"] for p in u["processes"]], "rss_gb": round(u["rss_bytes"] / 1e9, 2), "cpu_pct": round(u["cpu_percent"]),
                "cores": u["cores"], "mem_total_gb": round(u["mem_total_bytes"] / 1e9, 1), "mem_available_gb": round(u["mem_available_bytes"] / 1e9, 2),
                "source": u["source"], "processes": u["processes"]}
    except (OSError, ValueError, KeyError):
        pass
    pids = sorted(set(_listeners().values()))
    def ticks(pid):
        try:
            f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            return int(f[11]) + int(f[12])  # utime + stime
        except (OSError, IndexError, ValueError):
            return 0
    def rss(pid):
        try:
            return int(re.search(r"VmRSS:\s+(\d+)", Path(f"/proc/{pid}/status").read_text()).group(1)) * 1024
        except (OSError, AttributeError):
            return 0
    t0 = {p: ticks(p) for p in pids}
    time.sleep(0.5)
    hz = os.sysconf("SC_CLK_TCK")
    cpu = sum(ticks(p) - t0[p] for p in pids) / hz / 0.5 * 100
    try:
        avail = int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1)) * 1024
    except (OSError, AttributeError):
        avail = 0
    return {"pids": pids, "rss_gb": round(sum(rss(p) for p in pids) / 1e9, 2), "cpu_pct": round(cpu), "cores": os.cpu_count(),
            "mem_total_gb": round(mem_total() / 1e9, 1), "mem_available_gb": round(avail / 1e9, 2), "source": "sAGI/models.py (/proc)"}


def native_for(model: Path, engine: str | None = None) -> bool:
    """Whether the carrier answers from bankML's own forward pass (`bankml serve --native`, 0.3.0) for this model:
    "native" always (bankml refuses a model its forward pass does not run), "llama.cpp" never, "auto" for the
    ternary (Q2_0_g64) files — there bankML is about 8x llama-server with the same tokens; on the 1-bit files
    llama-server is still faster."""
    e = engine or resources().get("engine", "auto")
    return e == "native" or (e == "auto" and "Q2_0" in model.name)


def apply_resources(threads: int, ram_gb: float, busy=lambda: False, spec_ngram: bool = False, engine: str = "auto") -> dict:
    """Save the choice and restart the carrier on the same model with it (a verified switch, with rollback)."""
    threads = max(1, min(int(threads), os.cpu_count() or 1))
    st = serve_status()
    sha = (st.get("verified") or {}).get("model_sha256")
    pins = forks()
    path, fork = _pinned_path(sha, st.get("model"), pins) if sha else (None, None)
    if path is None:
        raise RuntimeError("no verified carrier is running; choose a model first (the settings are saved and used when it starts)")
    pl = plan(path, ram_gb)
    if not pl["fits"]:
        raise RuntimeError(f"{ram_gb:.1f} GB cannot hold {path.name}: it needs at least {pl['min_ram_gb']} GB")
    prev = resources()  # the last settings that ran, restored if these do not
    RESOURCES.parent.mkdir(parents=True, exist_ok=True)
    RESOURCES.write_text(json.dumps({"threads": threads, "ctx": pl["ctx"], "ram_gb": ram_gb, "spec_ngram": bool(spec_ngram),
                                     "engine": engine if engine in ("auto", "native", "llama.cpp") else "auto"}) + "\n")
    if busy():
        raise RuntimeError("saved; an answer is being written, so the engine restarts with these settings on the next switch")
    JOB["what"] = f"restarting {path.name} with {threads} threads and a {pl['ctx']}-token context"
    _stop_carrier()
    try:
        return _start_carrier(path, fork, sha)
    except Exception as first:
        RESOURCES.write_text(json.dumps(prev) + "\n")
        try:
            _stop_carrier()
            _start_carrier(path, fork, sha)
        except Exception as again:  # noqa: BLE001
            raise RuntimeError(f"{first}; restoring the previous settings also failed: {again}") from first
        raise


def _start_carrier(model: Path, fork: Path, want_sha: str | None = None, threads=None, ctx=None, wait=1800) -> dict:
    """Start `bankml serve --spawn` and wait until it answers verified, with `want_sha` when given. bankml hashes the
    whole file before it binds, so the wait is on the process, not on the ports: it fails only when the process exits
    or the wait runs out, and then the process group is killed so no second carrier is left behind."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "ab") as log:
        at = log.seek(0, 2)
        log.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} bankml serve {model.name}\n".encode())
        log.flush()
        SLOTS.mkdir(parents=True, exist_ok=True)
        r = resources()
        if not ctx and r.get("ram_gb"):
            pl = plan(model, float(r["ram_gb"]))
            if not pl["fits"]:
                raise RuntimeError(f"the saved RAM budget ({r['ram_gb']} GB) cannot hold {model.name}: it needs at least {pl['min_ram_gb']} GB")
            ctx = pl["ctx"]
        n_threads, n_ctx = str(threads or resources()["threads"]), str(ctx or resources()["ctx"])
        if native_for(model):
            # bankML's own forward pass answers, on the gateway and on the engine address (0.3.0)
            cmd = [str(BANKML), "serve", str(model), "--fork", str(fork), "--native", "--upstream", UPSTREAM, "--listen", LISTEN, "--ctx", n_ctx]
            env = {**os.environ, "BANKML_THREADS": n_threads}
        else:
            cmd = ([str(BANKML), "serve", str(model), "--fork", str(fork), "--spawn", str(LLAMA), "--upstream", UPSTREAM, "--listen", LISTEN,
                    "--threads", n_threads, "--ctx", n_ctx] + (["--spec-ngram"] if resources().get("spec_ngram") else []) + ["--slot-dir", str(SLOTS)])
            env = None
        proc = subprocess.Popen(cmd, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(REPO), env=env)
    t0 = time.time()
    why = "timed out"  # ctx: per model — the saved RAM budget is re-planned for this model's weights and KV size
    while time.time() - t0 < wait:
        st = serve_status(2)
        v = st.get("verified") or {}
        if v and (want_sha is None or v.get("model_sha256") == want_sha):
            return st
        if v:
            why = f"another carrier answers (sha256 {str(v.get('model_sha256'))[:12]}…), not {model.name}"
            break
        if proc.poll() is not None:
            why = f"exited with {proc.returncode}"
            break
        time.sleep(1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # serve and the llama-server it spawned share the group
    except ProcessLookupError:
        pass
    proc.wait(timeout=10)
    with open(LOG, "rb") as f:
        f.seek(at)
        said = [l for l in f.read().decode(errors="replace").splitlines()[1:] if l.strip()]
    raise RuntimeError(f"bankml serve did not come up with {model.name} ({why}): " + (" / ".join(said[-3:])[-400:] or "no output"))


def _pinned_path(sha: str, model: str | None, pins: dict):
    """(path, fork) for a verified sha256: a file in MODELS first, else the carrier's own path if a pin names it."""
    for f, (fk, rec) in pins.items():
        if rec["sha256"] == sha and (MODELS / f).exists():
            return MODELS / f, fk
    if model:
        m = Path(model)
        rec = pins.get(m.name)
        if rec and rec[1]["sha256"] == sha and m.exists():
            return m, rec[0]
    return None, None


def switch(file: str, busy=lambda: False) -> dict:
    """Make `file` (in MODELS, pinned) the carrier: success only when bankml serve answers verified with this file's
    sha256. If the new one fails, the previous model is restored (found by its verified sha256)."""
    if busy():
        raise RuntimeError("an answer is being written; switch when it is done")
    p = MODELS / file
    if not p.exists():
        raise FileNotFoundError(f"{file} is not in {MODELS}")
    ok, why = fits_memory(p.stat().st_size)
    if not ok:
        raise RuntimeError(why)
    arch = guard(p).get("arch")
    if arch in EMBEDDING_ARCHS:
        raise RuntimeError(f"{file} is an embedding model ({arch}); it cannot be the chat carrier (see docs/embedding.md)")
    pins = forks()
    if file not in pins:
        adopt(file)
        pins = forks()
    want = pins[file][1]["sha256"]
    prev = serve_status()
    prev_sha = (prev.get("verified") or {}).get("model_sha256")
    if prev_sha == want:
        return prev
    prev_path, prev_fork = _pinned_path(prev_sha, prev.get("model"), pins) if prev_sha else (None, None)
    JOB["what"] = f"stopping the carrier, starting {file} (bankml hashes the whole file, then llama-server loads it)"
    try:
        _stop_carrier()
        return _start_carrier(p, pins[file][0], want)
    except Exception:
        if prev_path:
            JOB["what"] = f"{file} failed; restoring {prev_path.name}"
            _stop_carrier()
            _start_carrier(prev_path, prev_fork, prev_sha)
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
    p = MODELS / c["file"]
    if p.is_symlink() and not p.exists():  # a dangling link (the target moved): replace it with the real file
        p.unlink()
    if not p.exists():
        import_spec(spec_catalog(c["id"]))
    return switch(c["file"], busy)


if __name__ == "__main__":  # python3 sAGI/models.py [list | catalog | import ID|URL|ollama:NAME:TAG | use FILE | first-run | search Q]
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
        loc = {f"{m['name']}:{m['tag']}" for m in ollama_local()}
        name = t.removeprefix("ollama:")
        spec = (spec_ollama(ollama_local_spec(name) if name in loc or f"{name}:latest" in loc else ollama_resolve(name)) if t.startswith("ollama:") else
                spec_catalog(t) if any(c["id"] == t for c in CATALOG) else None)
        if spec is None:
            r = hf_resolve(t)
            spec = spec_hf(r, r["chosen"] or (sys.exit("pick a file: " + ", ".join(f["file"] for f in r["files"])) if len(r["files"]) != 1 else r["files"][0]["file"]))
        print(json.dumps(import_spec(spec), indent=1))
    elif a[0] == "use":
        print(json.dumps(switch(a[1]), indent=1))
    elif a[0] == "first-run":
        print(json.dumps(first_run(), indent=1))
