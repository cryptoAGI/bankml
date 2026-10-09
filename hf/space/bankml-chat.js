// SPDX-License-Identifier: MIT OR Apache-2.0
// Ask bankML, from this static page: three ways, never confused.
//  · browseML — bankML in this browser: the engine compiled to WebAssembly (browseml.js), the model downloaded once
//    and verified by bankML before it answers, a receipt on every answer. The visitor's CPU; nothing to install. Free.
//  · Your own bankML — this browser talks to `bankml serve` on the visitor's machine (started with
//    --allow-origin for this page): bankML's verified arithmetic, the visitor's CPU and RAM, a receipt on every
//    answer whose sha256 this page checks. Free.
//  · A Hugging Face provider — sign in with Hugging Face; a provider-hosted model answers under bankML's persona,
//    billed to the visitor's inference quota. NOT bankML's arithmetic, no receipt; said on every answer.
// Who speaks is chosen apart from who answers: bankML or Savante, each from its own .persona and .context.
// The questions are asked in the ultimate input field (uif-space.js): each response window is its own conversation;
// this module is its engine (window.bankmlSpaceEngine), and it reports every answer's measurements to the page's
// controls and diagnostics (space-dashboard.js) through window.bankmlStats and the "bankml:stats" event.
// Nothing an engine or a provider returns becomes markup: text only.
import * as browseml from "./browseml.js";
import { oauthLoginUrl, oauthHandleRedirectIfPresent } from "https://cdn.jsdelivr.net/npm/@huggingface/hub@2.17.1/+esm";
import { InferenceClient } from "https://cdn.jsdelivr.net/npm/@huggingface/inference@4.13.28/+esm";

const $ = (id) => document.getElementById(id);
const SESSION = "bankml.oauth";
const DEFAULT_ENDPOINT = "http://127.0.0.1:18093";
const DEFAULT_MODEL = "Qwen/Qwen3-8B";
let persona = null;
let oauth = null;

const mode = () => document.querySelector('input[name="mode"]:checked').value;
const log = (kind, text) => window.bankmlLog && window.bankmlLog(kind, text);  // the Logs tab (space-tabs.js)
// The conversation sent with a question: at least the last 12 messages, trimmed four exchanges at a time, so the
// engine's cached prompt holds from turn to turn (a window that slid every turn made it read everything again).
const windowed = (h, keep = 12, step = 8) => h.slice(Math.floor(Math.max(0, h.length - keep) / step) * step);

// ── settings the visitor controls (the page's controls, space-dashboard.js), kept in this browser ─────────────────
const SETTINGS_KEY = "bankml.space.settings";
const DEFAULTS = { threads: 0 /* 0: every core */, ctx: 4096, maxTokens: 256, lean: true };
export const settings = (() => { try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") }; } catch { return { ...DEFAULTS }; } })();
export function setSettings(change) {
  Object.assign(settings, change);
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch {}
  stats.emit();
}
// ── what the page measured: every answer, the engine, the model — read by the diagnostics ───────────────────────
export const stats = {
  answers: [],        // {at, mode, speaker, prompt_n, predicted_n, cache_n, prompt_tps, gen_tps, ttft_ms, total_ms, ok}
  browse: null,       // what bankML verified in this browser, and its threads and context
  local: null,        // the visitor's own bankml serve, when connected
  emit() { window.dispatchEvent(new CustomEvent("bankml:stats")); },
};
window.bankmlStats = stats;

// ── what is happening now: one line at the top of the page (and in the dashboard), with a bar when it can measure ───
// downloading, verifying, starting the engine, reading the prompt, writing — never a silent wait
const activity = (() => {
  let timer = null;
  const show = (text, frac) => {
    const box = $("activity"); if (!box) return;
    box.hidden = !text;
    $("activity-text").textContent = text || "";
    const bar = $("activity-bar");
    bar.parentElement.hidden = frac === undefined || frac === null;
    if (frac !== undefined && frac !== null) bar.style.width = `${Math.max(0, Math.min(1, frac)) * 100}%`;
    stats.activity = text || null; window.dispatchEvent(new CustomEvent("bankml:activity"));
  };
  return {
    set(text, frac) { clearInterval(timer); timer = null; show(text, frac); },
    /** A line that ticks every second: `f(elapsedSeconds)` returns [text, fraction?]. */
    tick(f) {
      clearInterval(timer);
      const t0 = performance.now(), step = () => { const [t, fr] = f((performance.now() - t0) / 1000); show(t, fr); };
      step(); timer = setInterval(step, 1000);
    },
    clear() { clearInterval(timer); timer = null; show(null); },
  };
})();
const secs = (s) => (s < 90 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);
const ONE_THREAD_HINT = "one thread here — the full page (pythai-bankml.static.hf.space) uses every core";
window.bankmlSettings = { get: () => settings, set: setSettings };

// ── the persona: who speaks — the same files the bankML console and Savante speak from ───────────────────────────
// bankML's ships with this Space; Savante's is read from her own repository at a pinned commit and must hash to
// the sha256 recorded here, or she does not speak.
const PERSONAS = {
  bankml: { label: "bankML", persona: "sAGI/personas/bankml.persona", context: "sAGI/personas/bankml.context",
            heading: "your own codebase, generated from the repository" },
  savante: { label: "Savante",
             persona: "https://raw.githubusercontent.com/cryptoAGI/savante/eb4b8509decc09e61cf5e2b01d875d7ee6819d61/savante.persona",
             sha256: "d96556b1198940c73eb7b0456c29b9cdfd9eb7f83a3a88a65f8a880b95807b7e",
             context: "sAGI/personas/savante.context", heading: "who designed you and your family of repositories, from GitHub" },
};
let speaker = "bankml";
async function loadPersona(key = speaker) {
  const P = PERSONAS[key];
  try {
    const r = await fetch(P.persona, { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const raw = await r.text();
    if (P.sha256 && (await sha256hex(raw)) !== P.sha256) throw new Error(`${P.label}'s .persona does not match its pinned sha256: refused`);
    persona = JSON.parse(raw);
    speaker = key;
    log("persona", `${P.label} — ${key === "bankml" ? "sAGI/personas/bankml.persona" : "savante.persona (sha256 " + P.sha256.slice(0, 12) + "…, pinned)"}`);
  } catch (e) {
    persona = null;
    log("error", `${P.label}'s persona could not be read: ${e.message || e}`);
    return;
  }
  codebase = null;
  try {  // .context: generated from the repositories (tools/context.py); optional
    const r = await fetch(P.context, { cache: "no-store" });
    if (r.ok) { codebase = await r.json(); log("context", `${codebase.chunks.length} passages for ${P.label}${codebase.version ? " (bankML " + codebase.version + ")" : ""}`); }
  } catch (e) { codebase = null; }
  if (persona.mantra) $("mantra").textContent = "“" + persona.mantra + "”";
}

// ── .context: the summary after the persona (the cached prefix), the passages that match a question after it ──────
let codebase = null;
const baseSystem = () => persona.system_prompt + (codebase && codebase.summary ? `\n\nCONTEXT — ${PERSONAS[speaker].heading}:\n${codebase.summary}` : "");
const STOP = new Set(("the and for are was were this that with from what which who whom how why when where does did doing done can could would should will shall may might must have has had having been being into onto over under about your yours you you're its it's they them their there here then than also just only very more most some any all not but yes our ours ask tell me my mine please").split(" "));
const toks = (t) => (t.toLowerCase().match(/[a-z0-9]+/g) || []);
// BM25 as the console scores it (savante._BM25), so the same question brings the same passages on both
function passages(q, k = 2) {
  const chunks = (codebase && codebase.chunks) || [];
  const words = (q.toLowerCase().match(/[a-z0-9_.]+/g) || []).filter((w) => !STOP.has(w) && w.length > 2).join(" ");
  if (!chunks.length || !words) return { text: "", ids: [] };
  const docs = chunks.map((c) => { const tf = {}; for (const w of toks(`${c.title} ${c.title} ${c.text}`)) tf[w] = (tf[w] || 0) + 1; return { c, tf, len: Object.values(tf).reduce((a, b) => a + b, 0) }; });
  const df = {}; for (const d of docs) for (const w in d.tf) df[w] = (df[w] || 0) + 1;
  const n = docs.length, avg = docs.reduce((a, d) => a + d.len, 0) / n;
  const qw = [...new Set(toks(words))];
  const scored = docs.map((d) => ({ c: d.c, sc: qw.reduce((s, w) => d.tf[w] ? s + Math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) * d.tf[w] * 2.2 / (d.tf[w] + 1.2 * (0.25 + 0.75 * d.len / avg)) : s, 0) }))
    .filter((x) => x.sc >= 2.5).sort((a, b) => b.sc - a.sc).slice(0, k);
  if (!scored.length) return { text: "", ids: [] };
  return { ids: scored.map((x) => x.c.id),
    text: "CONTEXT — passages that match this question:\n" + scored.map((x) => `- [${x.c.source}] ${x.c.title}: ${x.c.text}` +
      ([x.c.github && `GitHub ${x.c.github}`, x.c.huggingface && `Hugging Face ${x.c.huggingface}`].filter(Boolean).join(" · ").replace(/^(.+)$/, " ($1)"))).join("\n") };
}

// ── the SELF block, as the console builds it: one sentence per measurement, "not measured" when it was not ──────
const get = async (ep, path) => {
  try {
    const r = await fetch(ep + path, { cache: "no-store" });
    return r.ok ? await r.json() : null;
  } catch { return null; }
};
async function selfText(ep) {
  const [b, u, m] = [await get(ep, "/bankml") || {}, await get(ep, "/bankml/usage") || {}, await get(ep, "/bankml/metrics") || {}];
  const last = (m.records || [{}]).slice(-1)[0] || {};
  const v = (x, unit = "", scale = 1, d = 1) => (x === null || x === undefined ? "not measured" : (x * scale).toFixed(d) + unit);
  const ver = b.verified || {};
  return [
    `- the model I am running: ${ver.name || b.resident || "none"} (sha256 ${String(ver.model_sha256 || "").slice(0, 16)}…), bankML ${ver.bankml}`,
    `- tokens I have read in total (prompts): ${v(m.prompt_tokens, "", 1, 0)}`,
    `- tokens I have written in total (answers): ${v(m.completion_tokens, "", 1, 0)}`,
    `- time to first token of my last answer: ${v(last.ttft_ms, " milliseconds", 1, 0)}`,
    `- prompt reading speed of my last answer: ${v(last.prompt_tps, " tokens per second")}`,
    `- generation speed of my last answer: ${v(last.eval_tps, " tokens per second")}`,
    `- CPU use right now: ${v(u.cpu_percent, " percent of one core")}`,
    `- memory I hold (RSS): ${v(u.rss_bytes, " GB", 1e-9, 2)}; memory still available on this machine: ${v(u.mem_available_bytes, " GB", 1e-9, 2)}`,
    `- power the CPU package draws: ${v(u.package_watts, " watts")}; energy per token I write: ${v(m.joules_per_token, " joules", 1, 3)}`,
  ].join("\n");
}

// a thinking model may still put its reasoning in the content as <think>…</think>: show the answer only
const answerOnly = (t) => t.replace(/<think>[\s\S]*?(<\/think>|$)/g, "").replace(/^\s+/, "");
async function sha256hex(text) {
  const h = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
/** A stream from callbacks: `start(push, end, fail)` feeds it; the field reads it piece by piece. */
function streamOf(start) {
  const q = []; let wake = null, done = false, err = null;
  const poke = () => { if (wake) { const w = wake; wake = null; w(); } };
  start((p) => { q.push(p); poke(); }, () => { done = true; poke(); }, (e) => { err = e; done = true; poke(); });
  return (async function* () {
    for (;;) {
      while (q.length) yield q.shift();
      if (err) throw err;
      if (done) return;
      await new Promise((r) => (wake = r));
    }
  })();
}
// the system message: the persona, and (unless lean) the summary of its .context — the part the engine caches
const system = () => (settings.lean ? persona.system_prompt : baseSystem());
function record(entry) {
  stats.answers.push({ at: Date.now(), speaker: PERSONAS[speaker].label, ...entry });
  if (stats.answers.length > 200) stats.answers.shift();
  stats.emit();
}
const receiptLine = (where, rc, ok, extra) => rc
  ? `${ok ? "✓" : "✗"} receipt — bankML ${rc.bankml}${where} · model sha256 ${String(rc.model_sha256 || "").slice(0, 12)}… · answer sha256 ${ok ? "matches the text received" : "does NOT match"}${extra}`
  : "✗ no receipt came with this answer";

// ── your own bankML ───────────────────────────────────────────────────────────────────────────────────────────
const endpoint = () => $("endpoint").value.trim().replace(/\/+$/, "") || DEFAULT_ENDPOINT;
async function connect() {
  const ep = endpoint();
  $("localstatus").textContent = "connecting…";
  log("connect", ep);
  const t0 = performance.now();
  const b = await get(ep, "/bankml");
  if (!b) {
    stats.local = null; stats.emit();
    $("localstatus").textContent = "not reachable — is bankml serve running with --allow-origin " + location.origin + " ? (see below)";
    log("error", `${ep} not reachable (bankml serve not running, or started without --allow-origin ${location.origin})`);
    return false;
  }
  const v = b.verified || {};
  stats.local = { endpoint: ep, rtt_ms: performance.now() - t0, verified: v, resident: b.resident }; stats.emit();
  log("connect", `${ep} answered in ${(performance.now() - t0).toFixed(0)} ms`);
  $("localstatus").textContent = v.guard === "play"
    ? `✓ connected: ${v.name || b.resident || "model"} · bankML ${v.bankml} · sha256 ${String(v.model_sha256 || "").slice(0, 12)}…`
    : "connected, but no verified model is loaded yet";
  log(v.guard === "play" ? "ok" : "warn", v.guard === "play"
    ? `verified ${v.name || b.resident || "model"} · bankML ${v.bankml} · sha256 ${String(v.model_sha256 || "").slice(0, 16)}…`
    : "connected, but no verified model is loaded");
  return v.guard === "play";
}
async function* askLocal(hist, message, done, progress = () => {}) {
  const ep = endpoint();
  // the persona first and unchanged, so the engine keeps it cached; SELF (measured now) just before the question
  const self = "SELF (measured by bankML just now):\n" + await selfText(ep);
  const ctx = passages(message);
  if (ctx.ids.length) log("context", "passages: " + ctx.ids.join(", "));
  const t0 = performance.now();
  let first = null, text = "", receipt = null, timings = {};
  activity.tick((s) => { const t = `your bankML at ${ep} is reading the prompt · ${secs(s)}`; progress(t); return ["📖 " + t]; });
  const r = await fetch(ep + "/v1/chat/completions", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "system", content: system() }, ...windowed(hist),
      { role: "system", content: self + (ctx.text ? "\n\n" + ctx.text : "") }, { role: "user", content: message }],
      stream: true, max_tokens: settings.maxTokens, timings_per_token: false }),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 300)}`);
  const reader = r.body.getReader(), dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done: fin, value } = await reader.read();
    if (fin) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (!line.startsWith("data:") || line === "data: [DONE]") continue;
      const d = JSON.parse(line.slice(5));
      if (d.timings) timings = d.timings;
      if (d.bankml_receipt) { receipt = d.bankml_receipt; continue; }
      const piece = d.choices?.[0]?.delta?.content;
      if (piece) { if (first === null) { first = performance.now(); activity.set("✍ your bankML is writing"); progress(`your bankML read the prompt in ${secs((first - t0) / 1000)} · writing`); log("answer", `first token after ${((first - t0) / 1000).toFixed(1)} s`); } text += piece; yield piece; }
    }
  }
  activity.clear();
  if (timings.prompt_n !== undefined) progress(`your bankML read ${timings.prompt_n} tokens (${Number(timings.prompt_per_second || 0).toFixed(1)} tok/s) · wrote ${timings.predicted_n} (${Number(timings.predicted_per_second || 0).toFixed(1)} tok/s)`);
  const ok = !!receipt && (await sha256hex(text)) === receipt.response_sha256;
  record({ mode: "local", prompt_n: timings.prompt_n ?? receipt?.prompt_tokens, predicted_n: timings.predicted_n ?? receipt?.completion_tokens, cache_n: timings.cache_n,
           prompt_tps: timings.prompt_per_second, gen_tps: timings.predicted_per_second, ttft_ms: first === null ? null : first - t0, total_ms: performance.now() - t0, ok });
  log(receipt ? (ok ? "ok" : "error") : "warn", receipt ? `receipt ${ok ? "✓" : "✗"} · ${receipt.prompt_tokens} + ${receipt.completion_tokens} tokens · ${((performance.now() - t0) / 1000).toFixed(1)} s` : "no receipt came with this answer");
  done({ ok, line: receiptLine(" on your machine", receipt, ok, ` · ${((performance.now() - t0) / 1000).toFixed(1)} s`) });
  hist.push({ role: "user", content: message }, { role: "assistant", content: text });
}

// ── browseML: bankML in this browser ─────────────────────────────────────────────────────────────────────────
let lastBrowse = null;  // the last answer's measurements, for the SELF block
function browseSelf(v) {
  const t = lastBrowse && lastBrowse.timings || {};
  const n = (x, unit, d = 1) => (x === undefined || x === null ? "not measured" : Number(x).toFixed(d) + unit);
  return "SELF (measured by bankML just now):\n" + [
    `- the model I am running: ${v.model} (sha256 ${v.sha256.slice(0, 16)}…), bankML ${v.version}, verified (guard ${v.guard})`,
    `- where I am running: in the visitor's own web browser, compiled to WebAssembly (browseML), ${v.threads} thread${v.threads > 1 ? "s" : ""} of their CPU — no server`,
    `- prompt reading speed of my last answer: ${n(t.prompt_per_second, " tokens per second")}`,
    `- generation speed of my last answer: ${n(t.predicted_per_second, " tokens per second")}`,
  ].join("\n");
}
export async function loadBrowse(progressLine) {
  if (browseml.isReady()) return true;
  $("browseload").disabled = true;
  const say = (t, frac) => { $("browsestatus").textContent = t; const pb = $("browseprogress"); pb.hidden = frac === undefined; if (frac !== undefined) pb.value = frac; if (progressLine) progressLine(t); };
  try {
    const t0 = performance.now();
    let dl0 = null, verifyClock = null;
    const v = await browseml.load((got, total, phase) => {
      if (phase === "download") {
        if (!dl0) dl0 = { t: performance.now(), got };
        const dt = (performance.now() - dl0.t) / 1000, rate = dt > 0.5 ? (got - dl0.got) / dt : 0;
        const left = rate > 0 ? (total - got) / rate : null;
        const t = `downloading the model (${browseml.MODEL.title}): ${(got / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB`
          + (rate ? ` · ${(rate / 1e6).toFixed(1)} MB/s · about ${secs(left)} left` : "") + " — once; it stays in this browser's cache";
        say(t, got / total); activity.set("⬇ " + t, got / total);
      } else if (phase === "cache") {
        const t = `the model is in this browser's cache (${(got / 1e6).toFixed(0)} MB): no download`;
        say(t); activity.set("◉ " + t);
      } else if (phase === "verify" && !verifyClock) {
        verifyClock = true;
        // the engine starts in a worker, checks the GGUF header, hashes all 248 MB against the pin, then lays out the
        // weights — about 4 to 10 s; the clock runs until it says it is verified
        activity.tick((s) => {
          const t = `bankML is verifying the model in this browser: the guard, then the sha256 of all ${(browseml.MODEL.bytes / 1e6).toFixed(0)} MB against its pin, then the weights · ${secs(s)}`;
          say(t); return ["🔒 " + t];
        });
      }
    }, (level, text) => log(level === 0 ? "error" : level === 1 ? "warn" : "engine", text),
    { threads: settings.threads || undefined, ctx: settings.ctx });
    stats.browse = { ...v, load_ms: performance.now() - t0 }; stats.emit();
    const done = `✓ ${v.model} verified by bankML ${v.version} · sha256 ${v.sha256.slice(0, 12)}… · ${v.threads} thread${v.threads > 1 ? "s" : ""} · ready in ${secs((performance.now() - t0) / 1000)}`;
    say(done); $("browseprogress").hidden = true;
    activity.set(done); setTimeout(() => { if (stats.activity === done) activity.clear(); }, 6000);
    log("ok", `browseML: ${v.model} verified · sha256 ${v.sha256.slice(0, 16)}… · ${v.threads} thread${v.threads > 1 ? "s" : ""}${v.threads > 1 ? "" : " (this page is not cross-origin isolated, or the browser has no SharedArrayBuffer)"} · context ${v.ctx} · engine ${v.engine}`);
    $("browseload").hidden = true;
    return true;
  } catch (e) {
    say("browseML could not load: " + (e.message || e)); $("browseprogress").hidden = true;
    activity.set("✗ browseML could not load: " + (e.message || e));
    log("error", "browseML could not load: " + (e.message || e));
    return false;
  } finally {
    $("browseload").disabled = false;
  }
}
/** Stop browseML and load it again with the current settings (threads, context); the model stays cached. */
export async function reloadBrowse() {
  browseml.unload();
  stats.browse = null; stats.emit();
  $("browseload").hidden = false;
  return loadBrowse();
}
async function* askBrowse(hist, message, done, progress = () => {}) {
  if (!browseml.isReady()) {
    // a first download waits for the visitor's Load (248 MB is their choice); one under way is waited for, and a
    // cached model loads in seconds — either way the window shows what the page shows at its top
    if (!browseml.isLoading() && !(await browseml.isCached())) throw new Error(`press “Load bankML in this browser” first: it downloads ${(browseml.MODEL.bytes / 1e6).toFixed(0)} MB once (the progress shows at the top of the page)`);
    const mirror = () => stats.activity && progress(stats.activity.replace(/^\S+ /, ""));
    window.addEventListener("bankml:activity", mirror);
    progress(browseml.isLoading() ? "waiting for bankML to finish loading…" : "starting bankML in this browser…");
    try {
      const ok = browseml.isLoading() ? await browseml.whenLoaded().then(() => true, () => false) : await loadBrowse();
      if (!ok || !browseml.isReady()) throw new Error("browseML could not load (the reason is in the Logs tab)");
    } finally { window.removeEventListener("bankml:activity", mirror); }
  }
  const v = stats.browse || await browseml.load(() => {}, () => {});
  const ctx = passages(message);
  if (ctx.ids.length) log("context", "passages: " + ctx.ids.join(", "));
  const msgs = [{ role: "system", content: system() }, ...windowed(hist),
    { role: "system", content: browseSelf(v) + (ctx.text ? "\n\n" + ctx.text : "") }, { role: "user", content: message }];
  // what it will read: everything new since this window's last answer (the persona and the earlier turns are kept in
  // the engine's cache while the same window goes on); an estimate, about 3.6 characters a token, said as one
  const chars = msgs.reduce((a, m) => a + m.content.length + 12, 0) - (hist.length ? system().length + windowed(hist).slice(0, -2).reduce((a, m) => a + m.content.length + 12, 0) : 0);
  const estTokens = Math.max(8, Math.round(chars / 3.6));
  const lastB = stats.answers.filter((a) => a.mode === "browse" && a.prompt_tps).slice(-1)[0];
  const rate = lastB ? lastB.prompt_tps : v.threads > 1 ? 2.2 : 1.0;
  const est = estTokens / rate;
  const thr = `${v.threads} thread${v.threads > 1 ? "s" : ""}`;
  const t0 = performance.now();
  let first = null, text = "", result = null, n = 0, lastShown = 0;
  const reading = (s) => {
    const over = s > est * 1.05;  // past the estimate: say so, and keep the clock running rather than a stuck bar
    const t = `reading the prompt in this browser · ${secs(s)} ${over ? `— longer than the ${secs(est)} estimated (the computer may be busy); still reading` : `of about ${secs(est)}`} (≈${estTokens} tokens at ${rate.toFixed(1)}/s) · ${thr}` + (v.threads > 1 ? "" : ` · ${ONE_THREAD_HINT}`);
    progress(t); return ["📖 " + t, over ? null : Math.min(0.97, s / est)];
  };
  activity.tick(reading);
  const pieces = streamOf((push, end, fail) => browseml.chat({ messages: msgs, max_tokens: settings.maxTokens },
    (piece) => {
      n++;
      if (first === null) { first = performance.now(); log("answer", `first token after ${((first - t0) / 1000).toFixed(1)} s`); }
      const now = performance.now();
      if (now - lastShown > 400) {
        lastShown = now;
        const tps = n > 1 ? (n - 1) / ((now - first) / 1000) : 0;
        const t = `read the prompt in ${secs((first - t0) / 1000)} · writing: ${n} tokens${tps ? ` · ${tps.toFixed(2)} tok/s` : ""} · ${settings.maxTokens} at most`;
        progress(t); activity.set("✍ " + t, n / settings.maxTokens);
      }
      text += piece; push(piece);
    },
    (level, t) => log(level === 0 ? "error" : "engine", t)).then((d) => { result = d; end(); }, (e) => { activity.clear(); fail(e); }));
  for await (const p of pieces) yield p;
  lastBrowse = result;
  const receipt = result.bankml_receipt, tm = result.timings || {};
  const ok = !!receipt && (await sha256hex(text)) === receipt.response_sha256;
  progress(`read ${tm.prompt_n ?? "?"} tokens in ${secs((tm.prompt_ms || 0) / 1000)} (${Number(tm.prompt_per_second || 0).toFixed(1)} tok/s${tm.cache_n ? `, ${tm.cache_n} reused` : ""}) · wrote ${tm.predicted_n ?? n} in ${secs((tm.predicted_ms || 0) / 1000)} (${Number(tm.predicted_per_second || 0).toFixed(2)} tok/s) · ${thr}`);
  activity.clear();
  record({ mode: "browse", prompt_n: tm.prompt_n, predicted_n: tm.predicted_n, cache_n: tm.cache_n, prompt_tps: tm.prompt_per_second,
           gen_tps: tm.predicted_per_second, ttft_ms: first === null ? null : first - t0, total_ms: performance.now() - t0, ok, memory: browseml.memoryBytes });
  log(receipt ? (ok ? "ok" : "error") : "warn", receipt
    ? `receipt ${ok ? "✓" : "✗"} · ${result.usage.prompt_tokens} + ${result.usage.completion_tokens} tokens · ${((performance.now() - t0) / 1000).toFixed(1)} s · ${Number(tm.predicted_per_second || 0).toFixed(2)} tok/s`
    : "no receipt came with this answer");
  done({ ok, line: receiptLine(" in this browser", receipt, ok, ` · ${((performance.now() - t0) / 1000).toFixed(1)} s`) });
  hist.push({ role: "user", content: message }, { role: "assistant", content: text });
}

// ── a Hugging Face provider (not bankML) ──────────────────────────────────────────────────────────────────────
function readSession() { try { return JSON.parse(sessionStorage.getItem(SESSION) || "null"); } catch { return null; } }
function writeSession(v) { try { v ? sessionStorage.setItem(SESSION, JSON.stringify(v)) : sessionStorage.removeItem(SESSION); } catch {} }
const signedIn = () => !!(oauth && oauth.accessToken && new Date(oauth.accessTokenExpiresAt) > new Date());
function renderAuth() {
  $("signin").hidden = signedIn();
  $("signout").hidden = !signedIn();
  $("who").textContent = signedIn()
    ? `signed in as ${oauth.userInfo?.preferred_username || oauth.userInfo?.name || "you"} — answers spend your inference quota`
    : "not signed in";
}
async function* askProvider(hist, message, done, progress = () => {}) {
  if (!signedIn()) throw new Error("sign in with Hugging Face first (who answers ▸ a provider); a free account needs purchased credits");
  const model = $("model").value.trim() || DEFAULT_MODEL;
  if (!/^[\w.-]+\/[\w.-]+$/.test(model)) throw new Error("the model must be a Hugging Face repository id, owner/name");
  // the persona, told the truth about where it is running
  const ctx = passages(message);
  if (ctx.ids.length) log("context", "passages: " + ctx.ids.join(", "));
  const who = PERSONAS[speaker].label;
  const sys = system() + (ctx.text ? "\n\n" + ctx.text : "") + `\n\nWHERE THIS REPLY COMES FROM: you speak as ${who} — the design, the rule, the voice — but this particular reply is generated by ${model} through a Hugging Face inference provider, not by bankML's engine: there is no bankML arithmetic, no receipt and no SELF block behind it. Answer as ${who} would; when asked about your speed, use, receipt, verification or what is running right now, say plainly that this reply comes from ${model}, not from bankML, and that bankML in the browser (browseML) or the visitor's own bankML gives verified answers.`;
  const t0 = performance.now();
  let text = "", shown = "", first = null;
  const client = new InferenceClient(oauth.accessToken);
  activity.tick((s) => { const t = `asking ${model} through a Hugging Face provider (not bankML) · ${secs(s)}`; progress(t); return ["☁ " + t]; });
  // Qwen3 and other thinking models: /no_think in the prompt (honoured by the model itself) and enable_thinking off
  // (honoured by some providers) — otherwise the whole budget can go to reasoning and no answer arrives
  const stream = client.chatCompletionStream({ provider: "auto", model, max_tokens: Math.max(settings.maxTokens, 512),
    chat_template_kwargs: { enable_thinking: false },
    messages: [{ role: "system", content: sys + "\n/no_think" }, ...hist.slice(-12), { role: "user", content: message }] });
  for await (const chunk of stream) {
    const d = chunk?.choices?.[0]?.delta || {};
    if (!d.content) continue;
    text += d.content;
    const now = answerOnly(text);
    if (now.length > shown.length && now.startsWith(shown)) { if (first === null) { first = performance.now(); activity.set("✍ the provider is writing"); } yield now.slice(shown.length); shown = now; }
  }
  activity.clear();
  if (!shown) yield "The provider returned no answer (perhaps only the model's reasoning) — ask again, or choose a model that does not think aloud.";
  record({ mode: "hf", ttft_ms: first === null ? null : first - t0, total_ms: performance.now() - t0, ok: false });
  done({ ok: false, line: `not bankML — ${model} via a Hugging Face provider · no receipt · your quota` });
  hist.push({ role: "user", content: message }, { role: "assistant", content: shown });
}

window.bankmlLoadBrowse = loadBrowse;
window.bankmlReloadBrowse = reloadBrowse;

// ── the engine the input field asks (uif-space.js) ─────────────────────────────────────────────────────────────
const statusListeners = new Set();
const announce = () => statusListeners.forEach((f) => f());
window.addEventListener("bankml:stats", announce);
window.bankmlSpaceEngine = {
  async *ask(hist, message, done, _window, progress) {
    if (!persona) throw new Error("the persona could not be read: reload the page");
    window.bankmlDismissHighlights && window.bankmlDismissHighlights();  // the highlights give way to the conversation
    const m = mode(), how = { browse: "browseML", local: "your bankML", hf: "provider " + $("model").value }[m];
    log("ask", `${PERSONAS[speaker].label} via ${how}: ${message.length > 80 ? message.slice(0, 79) + "…" : message}`);
    try {
      yield* (m === "browse" ? askBrowse(hist, message, done, progress) : m === "local" ? askLocal(hist, message, done, progress) : askProvider(hist, message, done, progress));
    } catch (e) {
      activity.clear();
      const who = { browse: "bankML in your browser did not answer: ", local: "your bankML did not answer: ", hf: "the provider did not answer: " }[m];
      log("error", who + String(e.message || e).slice(0, 300));
      throw new Error(who + String(e.message || e).slice(0, 300) + (m === "local" ? " — is bankml serve running with --allow-origin " + location.origin + " ?" : ""));
    }
  },
  status() {
    const m = mode(), who = PERSONAS[speaker].label;
    if (m === "browse") return stats.browse ? `${who} · ✓ ${stats.browse.model.replace(/\.gguf$/, "")} · ${stats.browse.threads} thr` : `${who} · browseML not loaded`;
    if (m === "local") return stats.local?.verified?.guard === "play" ? `${who} · ✓ ${stats.local.verified.name || "your bankML"}` : `${who} · your bankML: connect`;
    return `${who} · provider (not bankML)`;
  },
  onStatus(f) { statusListeners.add(f); },
  async ping() {
    const m = mode();
    if (m === "local") {
      const ts = [];
      for (let i = 0; i < 3; i++) { const t0 = performance.now(); const b = await get(endpoint(), "/health"); if (b === null && !(await fetch(endpoint() + "/bankml").then((r) => r.ok, () => false))) return { ok: false, line: `ping ${endpoint()}: not reachable` }; ts.push(performance.now() - t0); }
      return { ok: true, line: `ping your bankML at ${endpoint()}: ${Math.min(...ts).toFixed(1)}/${(ts.reduce((a, b) => a + b, 0) / ts.length).toFixed(1)}/${Math.max(...ts).toFixed(1)} ms (min/avg/max of 3)` };
    }
    if (m === "browse") return { ok: browseml.isReady(), line: browseml.isReady() ? `browseML runs in this page (no network): ${stats.browse.threads} thread${stats.browse.threads > 1 ? "s" : ""}, engine memory ${(browseml.memoryBytes / 1e6).toFixed(0)} MB` : "browseML is not loaded" };
    return { ok: signedIn(), line: signedIn() ? "the provider answers through router.huggingface.co (not timed here)" : "not signed in to Hugging Face" };
  },
  async diag() { return window.bankmlDiag ? window.bankmlDiag() : { title: "diag", sections: [] }; },
};

// ── wiring ─────────────────────────────────────────────────────────────────────────────────────────────────────
function renderMode() {
  const m = mode();
  $("browserow").hidden = m !== "browse";
  $("answererline").innerHTML = "";
  $("answererline").append("· answered by ", Object.assign(document.createElement("b"), { textContent: { browse: "bankML in this browser", local: "your own bankML", hf: "a provider (not bankML)" }[m] }),
    m === "browse" ? " (browseML)" : "", " · ", Object.assign(document.createElement("a"), { href: "#", textContent: "change", onclick: (e) => { e.preventDefault(); document.querySelector("details.who").open = true; document.querySelector("details.who").scrollIntoView({ block: "nearest" }); } }));
  $("localrow").hidden = m !== "local";
  $("hfrow").hidden = m !== "hf";
  log("mode", m === "browse" ? "browseML: bankML in this browser" : m === "local" ? "your own bankML" : "a Hugging Face provider (not bankML)");
  $("modenote").textContent = m === "browse"
    ? `bankML's own engine in this page, on your CPU: it downloads ${(browseml.MODEL.bytes / 1e6).toFixed(0)} MB once (${browseml.MODEL.title}), verifies it, and puts a receipt on every answer. Nothing is sent anywhere.`
    : m === "local"
    ? "bankML's own arithmetic on your machine: verified model, receipt checked here, nothing sent anywhere else."
    : "Not bankML: a provider-hosted model speaks with the chosen persona, without bankML's arithmetic or a receipt. Free Hugging Face accounts have no inference credits: this needs PRO or purchased credits.";
  stats.mode = m; stats.emit();
}
document.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", renderMode));
$("connect").addEventListener("click", connect);
$("browseload").addEventListener("click", () => loadBrowse());
$("browseforget").addEventListener("click", async () => { await browseml.forget(); $("browsestatus").textContent = "the model is removed from this browser's cache"; log("mode", "browseML: the cached model was removed"); stats.emit(); });
document.querySelectorAll('input[name="speaker"]').forEach((r) => r.addEventListener("change", async () => {
  // a new speaker: the response windows keep their conversations; the next question is answered as the new persona
  if (r.checked) { await loadPersona(r.value); log("persona", `${PERSONAS[r.value].label} speaks from now on`); stats.emit(); }
}));
$("signin").addEventListener("click", async () => {
  if (!window.huggingface?.variables?.OAUTH_CLIENT_ID) { $("who").textContent = "sign-in works only on the Hugging Face Space itself"; return; }
  window.location.href = await oauthLoginUrl({ scopes: window.huggingface.variables.OAUTH_SCOPES });
});
$("signout").addEventListener("click", () => { writeSession(null); oauth = null; renderAuth(); });
$("origin").textContent = location.origin;

(async () => {
  $("endpoint").value = DEFAULT_ENDPOINT;
  $("model").value = DEFAULT_MODEL;
  renderMode();
  await loadPersona("bankml");
  if (await browseml.isCached()) $("browsestatus").textContent = "the model is already in this browser's cache: loading takes a few seconds, then bankML verifies it";
  try { const res = await oauthHandleRedirectIfPresent(); if (res) writeSession(res); } catch (e) { $("who").textContent = "sign-in failed: " + String(e).slice(0, 120); }
  oauth = readSession();
  renderAuth();
  // the input field: mounted once its engine (above) exists
  const mountUIF = () => window.BankmlSpaceUIF && window.BankmlSpaceUIF.mount($("uif-root"), window.bankmlSpaceEngine,
    { placeholder: "Ask bankML about itself, its exactness, its speed… (Enter)", terminalPlaceholder: "T mode: help · diag · ping · tile · close all" });
  if (window.BankmlSpaceUIF) mountUIF(); else window.addEventListener("load", mountUIF, { once: true });
})();
