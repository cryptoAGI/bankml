// SPDX-License-Identifier: MIT OR Apache-2.0
// browseML's worker: bankML's engine runs here, off the page's thread, so the page stays responsive while the
// visitor's CPU reads the prompt and writes the answer.
//
// Threads: on a cross-origin-isolated page (SharedArrayBuffer available) it runs browseml-mt.wasm, the engine
// built for wasm32-wasip1-threads, with its thread pool on Web Workers (browseml-thread.js) sharing its memory —
// the pool's results do not depend on the thread count, so the answers are the same bits. Elsewhere it runs
// browseml.wasm on this one thread.
//
//   page → worker  {type: "open", name, bytes: ArrayBuffer (transferred), fork, threads, ctx}  |  {type: "chat", request}
//   worker → page  {type: "log", level, text} · {type: "opened", rc, out, threads, memory} · {type: "piece", text} ·
//                  {type: "done", rc, out, memory}   (memory: the engine's WebAssembly memory, bytes)
import { wasi } from "./browseml-wasi.js";

let x = null, w = null, memory = null;
const dec = new TextDecoder(), enc = new TextEncoder();
const text = (p, n) => dec.decode(new Uint8Array(memory.buffer).slice(p, p + n));
const put = (s) => { const b = enc.encode(s); const p = x.browseml_alloc(b.length); new Uint8Array(memory.buffer, p, b.length).set(b); return [p, b.length]; };
const out = () => text(x.browseml_out_ptr(), x.browseml_out_len());

/** The limits of the module's imported memory (env.memory), read from its import section. */
function importedMemory(bytes) {
  const b = new Uint8Array(bytes);
  let o = 8;
  const leb = () => { let r = 0, s = 0, c; do { c = b[o++]; r += (c & 0x7f) * 2 ** s; s += 7; } while (c & 0x80); return r; };
  const name = () => { const n = leb(); const t = dec.decode(b.subarray(o, o + n)); o += n; return t; };
  while (o < b.length) {
    const id = b[o++], size = leb(), end = o + size;
    if (id !== 2) { o = end; continue; }
    for (let i = 0, n = leb(); i < n; i++) {
      const mod = name(), field = name(), kind = b[o++];
      if (kind === 0) leb();
      else if (kind === 1) { o++; const f = b[o++]; leb(); if (f & 1) leb(); }
      else if (kind === 3) o += 2;
      else if (kind === 2) {
        const f = b[o++], min = leb(), max = f & 1 ? leb() : undefined;
        if (mod === "env" && field === "memory") return { initial: min, maximum: max, shared: !!(f & 2) };
      }
    }
    return null;
  }
  return null;
}

async function open({ name, bytes, fork, threads, ctx }) {
  const files = { [name]: new Uint8Array(bytes) };
  const log = (fd, s) => postMessage({ type: "log", level: fd === 2 ? 1 : 2, text: s.trimEnd() });
  const browseml = {
    piece: (p, n) => { postMessage({ type: "piece", text: text(p, n) }); return 1; },
    log: (level, p, n) => postMessage({ type: "log", level, text: text(p, n) }),
  };
  const mt = threads > 1 && typeof SharedArrayBuffer !== "undefined" && self.crossOriginIsolated;
  let n = 1;
  if (mt) {
    const wasmBytes = await (await fetch(new URL("./browseml-mt.wasm", import.meta.url))).arrayBuffer();
    const module = await WebAssembly.compile(wasmBytes);
    const lim = importedMemory(wasmBytes);
    if (!lim || !lim.shared) throw new Error("browseml-mt.wasm does not import a shared memory");
    memory = new WebAssembly.Memory({ initial: lim.initial, maximum: lim.maximum, shared: true });
    n = threads;
    // the pool's workers first, all ready before the engine starts: a waiting thread cannot start a worker
    const idle = await Promise.all(Array.from({ length: n - 1 }, () => new Promise((ok, bad) => {
      const t = new Worker(new URL("./browseml-thread.js", import.meta.url), { type: "module" });
      t.onmessage = (e) => {
        if (e.data.type === "ready") ok(t);
        else if (e.data.type === "log") postMessage(e.data);
        else if (e.data.type === "exit" && e.data.error) postMessage({ type: "log", level: 0, text: `a thread stopped: ${e.data.error}` });
      };
      t.onerror = (e) => bad(new Error(e.message || "a thread could not start"));
      t.postMessage({ type: "init", module, memory });
    })));
    let tid = 0;
    w = wasi({ files, env: { BANKML_CACHE_RAM: "0", BANKML_THREADS: String(n) }, write: log });
    ({ exports: x } = await WebAssembly.instantiate(module, {
      ...w.imports, env: { memory }, browseml,
      wasi: { "thread-spawn": (arg) => { const t = idle.shift(); if (!t) return -1; t.postMessage({ type: "start", tid: ++tid, arg }); return tid; } },
    }));
  } else {
    w = wasi({ files, env: { BANKML_CACHE_RAM: "0", BANKML_THREADS: "1" }, write: log });
    const { instance } = await WebAssembly.instantiateStreaming(fetch(new URL("./browseml.wasm", import.meta.url)), { ...w.imports, browseml });
    x = instance.exports;
    memory = x.memory;
  }
  w.bind(memory);
  const rc = x.browseml_open(...put(`/models/${name}`), ...put(fork), ctx || 4096);
  w.release(name);  // the engine holds the model in its own memory now; the downloaded copy can go
  postMessage({ type: "opened", rc, out: out(), threads: n, memory: memory.buffer.byteLength });
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "open") await open(m);
    else if (m.type === "chat") {
      if (!x) throw new Error("no model is open");
      const [p, len] = put(JSON.stringify(m.request));
      const rc = x.browseml_chat(p, len);
      x.browseml_free(p, len);
      postMessage({ type: "done", rc, out: out(), memory: memory.buffer.byteLength });
    }
  } catch (err) {
    // a trap (out of memory, a panic) ends this engine; the page says so and can load it again
    postMessage({ type: m.type === "open" ? "opened" : "done", rc: -99, out: String(err && err.message || err) });
    x = null;
  }
};
