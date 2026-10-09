// SPDX-License-Identifier: MIT OR Apache-2.0
// browseML's bench: the engine as the page runs it — browseml.wasm on one thread, or browseml-mt.wasm with its pool on
// worker threads sharing its memory (as browseml-worker.js and browseml-thread.js do with Web Workers) — timing
// prompt reading and writing, and checking the answer against llama-server b11192's record when one is given.
//   node browseML/testing/bench.mjs WASM MODEL.gguf FORK.json THREADS [oracle.jsonl [turns]]
// With THREADS > 1, WASM must be browseml-mt.wasm. `node --cpu-prof --cpu-prof-dir=DIR` profiles every thread.
import fs from "node:fs";
import path from "node:path";
import { Worker, isMainThread, workerData, parentPort } from "node:worker_threads";
import { wasi as makeWasi } from "../../hf/space/browseml-wasi.js";

const dec = new TextDecoder(), enc = new TextEncoder();

/** The limits of the module's imported memory (env.memory), from its import section. */
function importedMemory(bytes) {
  const b = new Uint8Array(bytes); let o = 8;
  const leb = () => { let r = 0, s = 0, c; do { c = b[o++]; r += (c & 0x7f) * 2 ** s; s += 7; } while (c & 0x80); return r; };
  const name = () => { const n = leb(); const t = dec.decode(b.subarray(o, o + n)); o += n; return t; };
  while (o < b.length) {
    const id = b[o++], size = leb(), end = o + size;
    if (id !== 2) { o = end; continue; }
    for (let i = 0, n = leb(); i < n; i++) {
      const mod = name(), field = name(), kind = b[o++];
      if (kind === 0) leb(); else if (kind === 1) { o++; const f = b[o++]; leb(); if (f & 1) leb(); } else if (kind === 3) o += 2;
      else if (kind === 2) { const f = b[o++], min = leb(), max = f & 1 ? leb() : undefined; if (mod === "env" && field === "memory") return { initial: min, maximum: max }; }
    }
    return null;
  }
  return null;
}

if (!isMainThread) {
  // one thread of the engine's pool: the same module on the same memory, then the pool's loop
  const { module, memory, tid, arg } = workerData;
  const w = makeWasi({ files: {} });
  const instance = await WebAssembly.instantiate(module, { ...w.imports, env: { memory }, wasi: { "thread-spawn": () => -1 }, browseml: { piece: () => 1, log: () => {} } });
  w.bind(memory);
  instance.exports.wasi_thread_start(tid, arg);
  parentPort.postMessage("exit");
} else {
  const [wasmPath, modelPath, forkPath, threadsArg, oraclePath, turnsArg] = process.argv.slice(2);
  const threads = +threadsArg || 1, name = path.basename(modelPath);
  const bytes = fs.readFileSync(wasmPath);
  const module = await WebAssembly.compile(bytes);
  const w = makeWasi({ files: { [name]: new Uint8Array(fs.readFileSync(modelPath)) }, env: { BANKML_CACHE_RAM: "0", BANKML_THREADS: String(threads) }, write: (fd, s) => process.stderr.write(s) });
  let memory, onPiece = () => 1, tid = 0;
  const imports = { ...w.imports, browseml: { piece: (p, n) => onPiece(dec.decode(new Uint8Array(memory.buffer).slice(p, p + n))), log: () => {} } };
  if (threads > 1) {
    const lim = importedMemory(bytes);
    memory = new WebAssembly.Memory({ initial: lim.initial, maximum: lim.maximum, shared: true });
    imports.env = { memory };
    imports.wasi = { "thread-spawn": (arg) => { const t = ++tid; new Worker(new URL(import.meta.url), { workerData: { module, memory, tid: t, arg } }).unref(); return t; } };
  }
  const instance = await WebAssembly.instantiate(module, imports);
  const x = instance.exports; if (!memory) memory = x.memory;
  w.bind(memory);
  const put = (s) => { const b = enc.encode(s); const p = x.browseml_alloc(b.length); new Uint8Array(memory.buffer, p, b.length).set(b); return [p, b.length]; };
  const out = () => dec.decode(new Uint8Array(memory.buffer).slice(x.browseml_out_ptr(), x.browseml_out_ptr() + x.browseml_out_len()));
  const t0 = performance.now();
  const rc = x.browseml_open(...put(`/models/${name}`), ...put(fs.readFileSync(forkPath, "utf8")), 1024);
  if (rc !== 0) { console.log("open failed:", out()); process.exit(1); }
  w.release(name);
  console.log(`open in ${((performance.now() - t0) / 1000).toFixed(1)} s · ${threads} thread${threads > 1 ? "s" : ""}`);
  const reqs = oraclePath
    ? fs.readFileSync(oraclePath, "utf8").trim().split("\n").map(JSON.parse).slice(0, +turnsArg || 1)
    : [{ request: { messages: [{ role: "user", content: "Write a short paragraph about the sea." }], max_tokens: 48, temperature: 0, seed: 1 } }];
  let same = 0;
  for (const r of reqs) {
    let text = ""; onPiece = (s) => { text += s; return 1; };
    const c = x.browseml_chat(...put(JSON.stringify(r.request)));
    const j = JSON.parse(out());
    if (c !== 0) { console.log("chat failed:", out()); process.exit(1); }
    const tm = j.timings, ok = r.text === undefined ? null : text === r.text;
    same += ok === true;
    console.log(`prompt ${tm.prompt_n} tok ${tm.prompt_per_second.toFixed(2)} tok/s · write ${tm.predicted_n} tok ${tm.predicted_per_second.toFixed(2)} tok/s${ok === null ? "" : ok ? " · identical" : " · DIFFERENT"}`);
  }
  if (oraclePath) console.log(`${same} of ${reqs.length} identical to the record`);
  process.exit(0);
}
