// SPDX-License-Identifier: MIT OR Apache-2.0
// browseML's model download, under the visitor's control: nothing is fetched until the visitor presses "Download and
// verify" on the page's consent card. Progress (MB done of total, MB/s, time left), Pause or a dropped connection (the
// bytes so far stay in this tab's memory only, and Resume or Retry asks for the rest with an HTTP Range request), Cancel (aborts the fetch and
// keeps nothing). The finished file goes into this browser's Cache Storage, where browseml.js finds it; bankML then
// checks its sha256 against the pin before it answers (browseml.load). Also what browseML keeps in this browser
// (Cache Storage, the engine files) and the browser's storage controls (estimate, persisted, persist).
import * as browseml from "./browseml.js";

export const CACHE = "browseml-models";  // as browseml.js keeps it
const M = browseml.MODEL;

/** phase: idle · downloading · paused · saving · failed · cancelled · done (in the cache; verification is the page's) */
let st = { phase: "idle", got: 0, total: M.bytes, rate: null, eta: null, error: null, resumable: false };
const fns = new Set();
const set = (patch) => { st = { ...st, ...patch }; fns.forEach((f) => { try { f(st); } catch { /* a listener's own failure */ } }); };
export const state = () => st;
export function onChange(f) { fns.add(f); return () => fns.delete(f); }

let abort = null, partial = null;

/** Bytes per second over the last few seconds of samples [{t, got}]. */
export function rateOf(samples) {
  if (samples.length < 2) return null;
  const a = samples[0], b = samples[samples.length - 1], dt = (b.t - a.t) / 1000;
  return dt > 0.4 ? (b.got - a.got) / dt : null;
}

/** Download the model into this browser's Cache Storage. Resolves true when it is there; false if paused, cancelled
 *  or failed (the reason is in state().error). Resumes a paused download with a Range request. */
export async function download() {
  if (st.phase === "downloading" || st.phase === "saving") return false;
  if (await browseml.isCached()) { set({ phase: "done", got: M.bytes, error: null }); return true; }
  abort = new AbortController();
  // a pause, or a connection that dropped mid-file, kept the bytes so far: ask for the rest with a Range request
  const from = partial && (st.phase === "paused" || (st.phase === "failed" && st.resumable)) ? st.got : 0;
  set({ phase: "downloading", error: null, resumable: false, rate: null, eta: null, got: from, total: M.bytes });
  let got = from, streaming = false;  // the exact bytes in hand; a failure while streaming the body is resumable
  try {
    const r = await fetch(M.url, { signal: abort.signal, headers: from ? { Range: `bytes=${from}-` } : undefined });
    if (!r.ok) throw new Error(`the model could not be downloaded: HTTP ${r.status}`);
    got = r.status === 206 ? from : 0;  // a server that ignores the range sends the whole file again
    if (!partial || got === 0) partial = new Uint8Array(M.bytes);
    const buf = partial, reader = r.body.getReader(), samples = [{ t: performance.now(), got }];
    let shown = 0;
    streaming = true;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (got + value.length > buf.length) { streaming = false; throw new Error("the download is longer than the model"); }
      buf.set(value, got); got += value.length;
      const t = performance.now();
      samples.push({ t, got });
      while (samples.length > 2 && t - samples[0].t > 5000) samples.shift();
      if (t - shown > 150) { shown = t; const rate = rateOf(samples); set({ got, rate, eta: rate ? (M.bytes - got) / rate : null }); }
    }
    if (got !== M.bytes) throw new Error(`the download stopped at ${got} of ${M.bytes} bytes`);
    streaming = false;
    set({ got, phase: "saving", rate: null, eta: null });
    try {
      const c = await caches.open(CACHE);
      await c.put(M.url, new Response(buf, { headers: { "Content-Type": "application/octet-stream", "Content-Length": String(M.bytes) } }));
    } catch (e) {
      throw new Error(`the model could not be kept in this browser (${String(e.message || e)}): free some storage, or leave a private window, and retry`);
    }
    partial = null;
    set({ phase: "done" });
    return true;
  } catch (e) {
    // pause() or cancel() already set the state; a pause resumes from the exact byte
    if (abort && abort.signal.aborted) { if (st.phase === "paused") set({ got }); return false; }
    // the connection dropped mid-file (a phone, Wi-Fi that changes): keep the bytes so far; Retry resumes there
    const resumable = streaming && !!partial && got > 0 && got < M.bytes;
    if (!resumable) partial = null;
    set({ phase: "failed", error: String(e.message || e), resumable, got: resumable ? got : st.got, rate: null, eta: null });
    return false;
  } finally {
    abort = null;
  }
}
/** Pause: stop the fetch; the bytes so far stay in this tab's memory (a reload loses them). */
export function pause() {
  if (st.phase !== "downloading" || !abort) return;
  set({ phase: "paused", rate: null, eta: null });
  abort.abort();
}
/** Cancel: stop the fetch and keep nothing. */
export function cancel() {
  if (st.phase === "downloading" && abort) abort.abort();
  partial = null;
  set({ phase: "cancelled", got: 0, rate: null, eta: null, error: null, resumable: false });
}
/** Say the download is over (the page verified, or the visitor removed the model). */
export function reset(phase = "idle", error = null) { partial = null; set({ phase, got: phase === "idle" ? 0 : st.got, rate: null, eta: null, error, resumable: false }); }

// ── what browseML keeps in this browser ─────────────────────────────────────────────────────────────────────────
/** Every file browseML keeps: the model in Cache Storage (exact bytes), and the engine's files as this page loaded
 *  them (the browser's HTTP cache, which it manages; sizes as it reports them). */
export async function keptFiles() {
  const out = [];
  try {
    for (const name of await caches.keys()) {
      if (!name.startsWith("browseml")) continue;
      const c = await caches.open(name);
      for (const req of await c.keys()) {
        const r = await c.match(req);
        const len = Number(r && r.headers.get("Content-Length")) || (r ? (await r.clone().blob()).size : 0);
        out.push({ name: decodeURIComponent(req.url.split("?")[0].split("/").pop() || req.url), bytes: len, where: "Cache Storage" });
      }
    }
  } catch { /* Cache Storage off (a private window) */ }
  try {
    const seen = new Set();
    for (const e of performance.getEntriesByType("resource")) {
      const f = e.name.split("?")[0].split("/").pop() || "";
      if (!/^browseml.*\.(wasm|js)$/.test(f) || seen.has(f)) continue;
      seen.add(f);
      if (e.decodedBodySize) out.push({ name: f, bytes: e.decodedBodySize, where: "HTTP cache" });
    }
  } catch { /* no resource timing */ }
  return out;
}
/** Remove browseML's Cache Storage (the model): the bytes it freed. A loaded engine keeps running until unloaded. */
export async function clearCache() {
  const freed = (await keptFiles()).filter((f) => f.where === "Cache Storage").reduce((n, f) => n + f.bytes, 0);
  try { for (const n of await caches.keys()) if (n.startsWith("browseml")) await caches.delete(n); } catch { /* off */ }
  reset("idle");
  return freed;
}
export async function persisted() { try { return navigator.storage && navigator.storage.persisted ? await navigator.storage.persisted() : null; } catch { return null; } }
export async function askPersist() { try { return navigator.storage && navigator.storage.persist ? await navigator.storage.persist() : null; } catch { return null; } }
export async function estimate() { try { const e = navigator.storage && navigator.storage.estimate ? await navigator.storage.estimate() : null; return e && e.quota ? { usage: e.usage || 0, quota: e.quota } : null; } catch { return null; } }

/** "Remove the model after this session": the session is this tab's (sessionStorage), so a reload or a link within the
 *  tab keeps the model, and the next visit in a new tab or window starts by removing what the last one kept. Nothing
 *  is removed on pagehide: it fires on a reload too (the 248 MB would be thrown away and downloaded again), and a
 *  closing tab gives no reliable time to delete them anyway. */
const MARK = "browseml:session";
export function sessionCleanup(on) {
  let fresh = false;
  try { fresh = !sessionStorage.getItem(MARK); sessionStorage.setItem(MARK, "1"); } catch { /* off */ }
  if (fresh && on()) { try { caches.delete(CACHE); } catch { /* off */ } }
}
