#!/usr/bin/env python3
"""The model importer (ui/models.py) without the network: a tiny GGUF served from a loopback HTTP server stands in for
Hugging Face / the Ollama registry. Checks the pin (a download is kept only if its sha256 is the published one), the
licence gate (open source or refused), resume, the guard, and the URL and search parsers. With BANKML_TEST_CARRIER=1
and Bonsai-1.7B present it also starts, switches and rolls back a real carrier on spare ports.
run: python3 testing/test_models.py"""
import hashlib, http.server, json, os, shutil, struct, sys, tempfile, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp(prefix="bankml-models-"))
os.environ.update(BANKML_MODELS=str(tmp / "models"), BANKML_FORKS=str(tmp / "forks"), BANKML_UI_STATE=str(tmp / "state"),
                  BANKML_SERVE_LISTEN="127.0.0.1:18293", BANKML_UPSTREAM="127.0.0.1:18292")
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "ui"))
import models as M  # noqa: E402

fails = 0


def check(name, ok):
    global fails
    fails += not ok
    print(("ok   " if ok else "FAIL ") + name)


def gguf(arch="qwen3", name="tiny", ty=0, n=64) -> bytes:
    """A minimal valid GGUF v3: two metadata strings, one tensor of `n` elements of type `ty` (F32), aligned to 32."""
    s = lambda x: struct.pack("<Q", len(x.encode())) + x.encode()
    h = b"GGUF" + struct.pack("<IQQ", 3, 1, 2)
    for k, v in (("general.architecture", arch), ("general.name", name)):
        h += s(k) + struct.pack("<I", 8) + s(v)
    h += s("t") + struct.pack("<I", 1) + struct.pack("<Q", n) + struct.pack("<I", ty) + struct.pack("<Q", 0)
    h += b"\0" * (-len(h) % 32)
    return h + b"\0" * (n * 4)


BLOB = gguf()
SHA = hashlib.sha256(BLOB).hexdigest()


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = BLOB
        rng = self.headers.get("Range")
        if rng and self.path != "/norange":
            a = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
            body = BLOB[a:]
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_address[1]}/tiny.gguf"
spec = lambda **k: {"file": "tiny.gguf", "bytes": len(BLOB), "sha256": SHA, "url": url, "repo": "test/tiny", "revision": "r1",
                    "licence": "apache-2.0", "pinned_from": "test", **k}

try:
    # parsers
    check("hf_parse: repo, blob URL, resolve URL, hf:// id", M.hf_parse("https://huggingface.co/Qwen/Qwen3-4B-GGUF") == ("Qwen/Qwen3-4B-GGUF", None, None)
          and M.hf_parse("https://huggingface.co/a/b/blob/main/x.gguf?download=true") == ("a/b", "main", "x.gguf")
          and M.hf_parse("https://hf.co/a/b/resolve/abc123/sub/y%20z.gguf") == ("a/b", "abc123", "sub/y z.gguf")
          and M.hf_parse("hf://a/b") == ("a/b", None, None))
    try:
        M.hf_parse("https://example.com/../../etc/passwd")
        check("hf_parse refuses what is not a repo id", False)
    except ValueError:
        check("hf_parse refuses what is not a repo id", True)
    check("licence gate: OSI licences pass; gemma, llama and none are refused", M.licence_open("apache-2.0") and M.licence_open(["license:mit"])
          and not M.licence_open("gemma") and not M.licence_open("llama3.2") and not M.licence_open(None) and not M.licence_open([]))
    check("Ollama licence text: Apache and MIT recognised; Gemma's terms named and refused",
          M._licence_of_text("Apache License\nVersion 2.0, January 2004") == "apache-2.0"
          and M._licence_of_text("MIT License\n\nPermission is hereby granted, free of charge") == "mit"
          and not M.licence_open(M._licence_of_text("Gemma Terms of Use\nLast modified")))
    check("Ollama names: model[:tag], namespaces; anything else refused", M._ollama_name("qwen3:1.7b") == ("qwen3", "1.7b")
          and M._ollama_name("library/granite3.3") == ("granite3.3", "latest") and M._ollama_name("user/model:q4") == ("user/model", "q4"))
    fx = ('<ul><li class="x"><a href="/library/granite4" class="g"><div><h2><span>granite4</span></h2>'
          '<p class="d">IBM Granite, released under Apache 2.0.</p></div><span class="a text-blue-600 b">1b</span>'
          '<span class="a text-blue-600 b">3b</span><span >12.3K</span>\n<span class="hidden">&nbsp;Pulls</span></a></li></ul>')
    import unittest.mock as um
    with um.patch.object(M, "_get", lambda *a, **k: fx.encode()):
        r = M.ollama_search("granite")
    check("ollama_search parses name, description, sizes, pulls", r == [{"name": "granite4", "description": "IBM Granite, released under Apache 2.0.",
                                                                       "sizes": ["1b", "3b"], "pulls": "12.3K", "cloud": False}])

    # the pin
    out = M.import_spec(spec())
    fk = json.loads(Path(out["fork"]).read_text())
    check("an import is kept when its sha256 is the published one, guarded, and pinned in FORK.json", out["arch"] == "qwen3"
          and (M.MODELS / "tiny.gguf").read_bytes() == BLOB and fk["files"] == [{"path": "tiny.gguf", "bytes": len(BLOB), "sha256": SHA}])
    check("installed() lists it as pinned", [m["pinned"] for m in M.installed() if m["file"] == "tiny.gguf"] == [True])
    try:
        M.import_spec(spec(file="tampered.gguf", sha256="0" * 64))
        check("a download whose sha256 differs is discarded", False)
    except RuntimeError as e:
        check("a download whose sha256 differs is discarded", "discarded" in str(e) and not list(M.MODELS.glob("tampered*")))
    try:
        M.import_spec(spec(file="gemma.gguf", licence="gemma"))
        check("a non-open-source licence is refused before any byte is fetched", False)
    except PermissionError:
        check("a non-open-source licence is refused before any byte is fetched", not (M.MODELS / "gemma.gguf").exists())
    try:
        M.import_spec(spec(file="unpinned.gguf", sha256=None))
        check("no published sha256, no import", False)
    except ValueError:
        check("no published sha256, no import", True)
    (M.MODELS / "resume.gguf.part").write_bytes(BLOB[:100])
    M.import_spec(spec(file="resume.gguf"))
    check("an interrupted download resumes from its .part and still verifies", (M.MODELS / "resume.gguf").read_bytes() == BLOB)
    (M.MODELS / "restart.gguf.part").write_bytes(BLOB[:100])
    M.import_spec(spec(file="restart.gguf", url=url.rsplit("/", 1)[0] + "/norange"))
    check("a server that ignores Range: the download starts over and verifies", (M.MODELS / "restart.gguf").read_bytes() == BLOB)
    bad = gguf(ty=99)
    SHA_BAD = hashlib.sha256(bad).hexdigest()
    BLOB, keep = bad, BLOB
    try:
        M.import_spec(spec(file="unplayable.gguf", sha256=SHA_BAD))
        check("bankml's guard refuses an unplayable file after download, and it is removed", False)
    except RuntimeError as e:
        check("bankml's guard refuses an unplayable file after download, and it is removed", "guard refused" in str(e) and not (M.MODELS / "unplayable.gguf").exists())
    BLOB = keep
    local = tmp / "ollama-blob"
    local.write_bytes(BLOB)
    M.import_spec(spec(file="adopted.gguf", local=str(local), url="http://127.0.0.1:9/unreachable"))
    check("a local Ollama blob is adopted by link after hashing, with no download", (M.MODELS / "adopted.gguf").is_symlink()
          and (M.MODELS / "adopted.gguf").resolve() == local.resolve())
    with um.patch.object(M, "free_bytes", lambda p=None: 10 ** 9):
        ok, why = M.fits(2 * 10 ** 9)
    check("fits(): a file that would leave less than the margin free is refused, with the numbers", not ok and "GB free" in why)
    check("the catalogue: every entry open source, fully pinned (64-hex sha256, 40-hex revision), one default, no Gemma",
          all(M.licence_open(c["licence"]) and len(c["sha256"]) == 64 and len(c["revision"]) == 40 for c in M.CATALOG)
          and sum(bool(c.get("default")) for c in M.CATALOG) == 1 and not any("gemma" in c["id"] for c in M.CATALOG))

    # a real carrier, on spare ports, when asked and when the small Bonsai is here
    b17 = ROOT / ".models" / "Bonsai-1.7B-Q1_0.gguf"
    if os.environ.get("BANKML_TEST_CARRIER") == "1" and b17.exists() and M.LLAMA.exists():
        os.symlink(b17.resolve(), M.MODELS / b17.name)
        c = next(x for x in M.CATALOG if x["file"] == b17.name)
        M.write_fork(b17.name, c["bytes"], c["sha256"], c["repo"], "", c["revision"], c["licence"], "test: the catalogue's pin")
        st = M.switch(b17.name)
        check("switch(): the carrier comes up verified on the chosen model", st["verified"]["model_sha256"] == c["sha256"] and st["verified"]["arch"] == "qwen3")
        try:
            M.switch("tiny.gguf")  # verifies, but llama-server cannot load a 64-float toy: it must roll back
            check("a switch that fails rolls back to the previous model", False)
        except RuntimeError:
            check("a switch that fails rolls back to the previous model", (M.serve_status().get("verified") or {}).get("model_sha256") == c["sha256"])
        M._stop_carrier()
    else:
        print("skip  carrier switch (set BANKML_TEST_CARRIER=1 with .models/Bonsai-1.7B-Q1_0.gguf present)")
finally:
    srv.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)

print("all ok" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
