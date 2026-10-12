// SPDX-License-Identifier: MIT OR Apache-2.0
// browseML — bankML in this browser. The model is downloaded once from its Hugging Face repository at a pinned
// revision and kept in this browser's Cache Storage; bankML's own engine, compiled to WebAssembly, verifies it
// against its FORK.json (the guard, then the sha256 pin) before it answers, and puts a receipt on every answer —
// the same receipt `bankml serve` gives. The visitor's CPU does the work: no server, no account, no quota.

export const MODEL = {
  name: "Bonsai-1.7B-Q1_0.gguf",
  title: "Bonsai-1.7B · 1-bit (Q1_0)",
  bytes: 248302272,
  url: "https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/210a9e99f79cb184909d49595906526eb2b3dd9a/Bonsai-1.7B-Q1_0.gguf",
  fork: "browseML/Bonsai-1.7B-Q1_0.gguf.FORK.json",
};
const CACHE = "browseml-models";

let worker = null, ready = null, busy = null;
/** The engine's WebAssembly memory after its last open or answer, in bytes (0 when not loaded). */
export let memoryBytes = 0;

/** The most threads the engine takes, however many cores the browser reports. */
export const MAX_THREADS = 8;
/** Threads for the engine: every core this browser reports (up to MAX_THREADS), when the page is cross-origin
 *  isolated (the Space sends COOP and COEP); otherwise one. Also the most a visitor can choose. */
export const threads = () => (globalThis.crossOriginIsolated && typeof SharedArrayBuffer !== "undefined" ? Math.max(1, Math.min(MAX_THREADS, navigator.hardwareConcurrency || 1)) : 1);

/** The model in this browser's cache, or null. */
async function cached() {
  try { const c = await caches.open(CACHE); const r = await c.match(MODEL.url); return r ? await r.arrayBuffer() : null; } catch { return null; }
}

/** The download in pieces: one long response can be cut mid-body (seen in Brave on Linux: "TypeError: network error"
 *  at 57 MB and at 219 MB of 248, where curl on the same machine got the whole file), and a cut late in one stream
 *  would cost the whole download. Each piece is an HTTP Range request, retried on its own, so a cut costs at most one
 *  piece's remainder (asked for again from the byte where it was cut) and the download goes on. */
export const PIECE = 16 * 1024 * 1024;
const TRIES = 6;
const pause = (ms) => new Promise((ok) => setTimeout(ok, ms));

/** Download the model (or take it from the cache), with progress as (bytes, total, phase); phase "retry" when a cut
 *  piece is asked for again (bytes: where it resumes). */
async function fetchModel(progress, { signal } = {}) {
  const kept = await cached();
  if (kept && kept.byteLength === MODEL.bytes) { progress(kept.byteLength, MODEL.bytes, "cache"); return kept; }
  const buf = await download(MODEL.url, MODEL.bytes, progress, { signal });
  // kept for the next visit (put finishes reading the buffer before it is handed to the engine); a private window or
  // a full disk just means downloading again
  try { const c = await caches.open(CACHE); await c.put(MODEL.url, new Response(buf, { headers: { "Content-Type": "application/octet-stream", "Content-Length": String(MODEL.bytes) } })); } catch {}
  return buf.buffer;
}

/** `total` bytes from `url` in pieces, each retried from the byte where it was cut; progress as fetchModel's. */
export async function download(url, total, progress, { signal, piece = PIECE, pauseMs = 500 } = {}) {
  const buf = new Uint8Array(total);
  let got = 0;
  const seen = (n) => { got = n; progress(n, total, "download"); };
  while (got < total) {
    const end = Math.min(total, got + piece) - 1;
    for (let tryNo = 1; ; tryNo++) {
      // a retry asks from the exact byte: what arrived of the cut piece is already in buf
      const from = got;
      try {
        const r = await fetch(url, { headers: { Range: `bytes=${from}-${end}` }, signal });
        if (r.status === 200 && from === 0) {
          // a server that ignores Range sends the whole file: take it as one stream
          await readInto(r, buf, 0, total, seen);
          if (got !== total) throw new Error(`the download stopped at ${got} of ${total} bytes`);
          break;
        }
        if (r.status !== 206) throw Object.assign(new Error(`the model could not be downloaded: HTTP ${r.status}`), { final: r.status >= 400 && r.status < 500 });
        await readInto(r, buf, from, end + 1, seen);
        if (got !== end + 1) throw new Error(`a piece stopped at ${got} of ${end + 1} bytes`);
        break;
      } catch (e) {
        if ((signal && signal.aborted) || e.final) throw e;  // stopped, or refused (a 4xx is not a cut)
        if (tryNo >= TRIES) throw new Error(`the download failed at ${got} of ${total} bytes after ${TRIES} tries: ${e.message || e}`);
        progress(got, total, "retry");
        await pause(Math.min(8000, pauseMs * 2 ** (tryNo - 1)));
      }
    }
  }
  return buf;
}

/** Read a response's body into buf from `at`, never past `limit`; `seen` gets the running end. */
async function readInto(r, buf, at, limit, seen) {
  const reader = r.body.getReader();
  let n = at;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return;
      if (n + value.length > limit) throw new Error("the download is longer than the model");
      buf.set(value, n); n += value.length;
      seen(n);
    }
  } catch (e) {
    reader.cancel().catch(() => {});
    throw e;
  }
}

export const isReady = () => !!ready;
export const isBusy = () => !!busy;
/** Stop the engine and free its memory (the model stays in the browser's cache): new settings apply on the next load. */
export function unload() {
  if (busy) throw new Error("browseML is writing an answer: wait for it to finish");
  if (worker) worker.terminate();
  worker = null; ready = null; memoryBytes = 0;
}
/** The model's bytes in this browser's cache, or 0. */
export async function cachedBytes() { try { const r = await (await caches.open(CACHE)).match(MODEL.url); return r ? +(r.headers.get("Content-Length") || MODEL.bytes) : 0; } catch { return 0; } }
export async function isCached() { try { return !!(await (await caches.open(CACHE)).match(MODEL.url)); } catch { return false; } }
export async function forget() { try { await caches.delete(CACHE); } catch {} }

/**
 * Load bankML in this browser: the model (downloaded or cached), the engine, and bankML's verification.
 * @param {(bytes:number, total:number, phase:string) => void} progress
 * @param {(level:number, text:string) => void} log
 * @returns {Promise<{model, sha256, guard, engine, arch, version, threads}>} what bankML verified, and its threads
 */
let loading = null;
/** A load in progress (a download, the verification), or null. */
export const isLoading = () => !!loading;
export const whenLoaded = () => loading;
export function load(progress, log, opts = {}) {
  if (ready) return Promise.resolve(ready);
  if (loading) return loading;  // one download at a time: a second caller waits for the first
  loading = loadOnce(progress, log, opts).finally(() => { loading = null; });
  return loading;
}
/** The last load, measured: when each phase ended (ms from the start), the bytes fetched (0: from this browser's
 *  cache), retried pieces, and the engine's WebAssembly memory once open. null before a load finishes. */
export let loadStats = null;
async function loadOnce(progress, log, { threads: want, ctx = 4096, cacheRam = 0 } = {}) {
  const t0 = performance.now(), st = { fetched: 0, retries: 0, source: "download" };
  const seen = (got, total, phase) => {
    if (phase === "retry") st.retries++;
    else if (phase === "cache") st.source = "cache";
    else if (phase === "download") st.fetched = got;
    progress(got, total, phase);
  };
  const [bytes, fork] = await Promise.all([fetchModel(seen), fetch(MODEL.fork).then((r) => { if (!r.ok) throw new Error("the model's FORK.json is missing"); return r.text(); })]);
  st.modelMs = Math.round(performance.now() - t0);
  progress(MODEL.bytes, MODEL.bytes, "verify");
  worker = new Worker(new URL("./browseml-worker.js", import.meta.url), { type: "module" });
  const verified = await new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "log") log(m.level, m.text);
      else if (m.type === "opened") { memoryBytes = m.memory || 0; m.rc === 0 ? resolve({ ...JSON.parse(m.out), threads: m.threads, ctx, build: m.build }) : reject(new Error(m.out)); }
    };
    worker.onerror = (e) => reject(new Error(e.message || "the engine could not start"));
    worker.postMessage({ type: "open", name: MODEL.name, bytes, fork, threads: Math.max(1, Math.min(threads(), want || threads())), ctx, cacheRam }, [bytes]);
  });
  loadStats = { ...st, openMs: Math.round(performance.now() - t0) - st.modelMs, totalMs: Math.round(performance.now() - t0), memory: memoryBytes, threads: verified.threads, ctx, cacheRam, build: verified.build };
  ready = verified;
  return verified;
}

/**
 * One answer: an OpenAI-shaped request (messages, max_tokens, …), the pieces as they are written, and the
 * `/v1/chat/completions` object at the end (usage, timings, bankml_receipt).
 */
export function chat(request, onPiece, log) {
  if (!ready) return Promise.reject(new Error("browseML is not loaded"));
  if (busy) return Promise.reject(new Error("browseML is still writing the last answer"));
  busy = new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "piece") onPiece(m.text);
      else if (m.type === "log") log(m.level, m.text);
      else if (m.type === "done") {
        busy = null;
        memoryBytes = m.memory || memoryBytes;
        if (m.rc === -99) { ready = null; worker.terminate(); worker = null; }  // the engine trapped: load it again
        const j = (() => { try { return JSON.parse(m.out); } catch { return { error: { message: m.out } }; } })();
        m.rc === 0 ? resolve(j) : reject(new Error(j.error ? j.error.message : m.out));
      }
    };
    worker.postMessage({ type: "chat", request });
  });
  return busy;
}
