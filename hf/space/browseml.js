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

/** The model in this browser's cache, or null. */
async function cached() {
  try { const c = await caches.open(CACHE); const r = await c.match(MODEL.url); return r ? await r.arrayBuffer() : null; } catch { return null; }
}

/** Download the model (or take it from the cache), with progress as (bytes, total, phase). */
async function fetchModel(progress) {
  const kept = await cached();
  if (kept && kept.byteLength === MODEL.bytes) { progress(kept.byteLength, MODEL.bytes, "cache"); return kept; }
  const r = await fetch(MODEL.url);
  if (!r.ok) throw new Error(`the model could not be downloaded: HTTP ${r.status}`);
  const total = +r.headers.get("Content-Length") || MODEL.bytes;
  const buf = new Uint8Array(total);
  const reader = r.body.getReader();
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (got + value.length > buf.length) throw new Error("the download is longer than the model");
    buf.set(value, got); got += value.length;
    progress(got, total, "download");
  }
  if (got !== total) throw new Error(`the download stopped at ${got} of ${total} bytes`);
  // kept for the next visit (put finishes reading the buffer before it is handed to the engine); a private window or
  // a full disk just means downloading again
  try { const c = await caches.open(CACHE); await c.put(MODEL.url, new Response(buf, { headers: { "Content-Type": "application/octet-stream" } })); } catch {}
  return buf.buffer;
}

export const isReady = () => !!ready;
export async function isCached() { try { return !!(await (await caches.open(CACHE)).match(MODEL.url)); } catch { return false; } }
export async function forget() { try { await caches.delete(CACHE); } catch {} }

/**
 * Load bankML in this browser: the model (downloaded or cached), the engine, and bankML's verification.
 * @param {(bytes:number, total:number, phase:string) => void} progress
 * @param {(level:number, text:string) => void} log
 * @returns {Promise<{model, sha256, guard, engine, arch, version}>} what bankML verified
 */
export async function load(progress, log) {
  if (ready) return ready;
  const [bytes, fork] = await Promise.all([fetchModel(progress), fetch(MODEL.fork).then((r) => { if (!r.ok) throw new Error("the model's FORK.json is missing"); return r.text(); })]);
  progress(MODEL.bytes, MODEL.bytes, "verify");
  worker = new Worker(new URL("./browseml-worker.js", import.meta.url), { type: "module" });
  const verified = await new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "log") log(m.level, m.text);
      else if (m.type === "opened") m.rc === 0 ? resolve(JSON.parse(m.out)) : reject(new Error(m.out));
    };
    worker.onerror = (e) => reject(new Error(e.message || "the engine could not start"));
    worker.postMessage({ type: "open", name: MODEL.name, bytes, fork }, [bytes]);
  });
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
        if (m.rc === -99) { ready = null; worker.terminate(); worker = null; }  // the engine trapped: load it again
        const j = (() => { try { return JSON.parse(m.out); } catch { return { error: { message: m.out } }; } })();
        m.rc === 0 ? resolve(j) : reject(new Error(j.error ? j.error.message : m.out));
      }
    };
    worker.postMessage({ type: "chat", request });
  });
  return busy;
}
