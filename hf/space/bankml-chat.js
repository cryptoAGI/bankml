// SPDX-License-Identifier: MIT OR Apache-2.0
// Ask bankML, from this static page: who answers is chosen with the tabs over the conversation, and named on every answer.
//  · bankML · free CPU (the default) — answers at once, as PYTHAI/savante renders from Bonsai: through @gradio/client
//    to the Space Gregory-L/mindXhfgradio, endpoint /mindx (uif-space.js askSpace). HONEST: that is Bonsai in llama.cpp
//    on Hugging Face's free CPU, not bankML's engine, and there is no receipt; every answer and the tab's note say so.
//    What is typed is processed on a public Space. When that Space offers a /bankml endpoint that returns a
//    bankml_receipt, askSpace uses it and the receipt is checked here.
//  · browseML — bankML in this browser: the engine compiled to WebAssembly (browseml.js), the model (Bonsai-1.7B
//    Q1_0, 248 MB) downloaded only after the visitor presses "Download and verify" on the in-page consent card
//    (browseml-fetch.js: progress, pause, cancel), verified by bankML before it answers, a receipt on every answer.
//  · your own bankML — `bankml serve` on the visitor's machine (started with --allow-origin for this page), a receipt
//    on every answer whose sha256 this page checks. When it does not answer, one card offers the ways forward.
//  · Hugging Face — a provider-hosted model answers under bankML's persona, billed to the visitor's own account. NOT
//    bankML's arithmetic, no receipt; said on every answer. The sign-in and the token belong to the separate,
//    GPL-3.0-only module bankml-creds.js (window.bankmlCreds): this file never holds the token, never writes the
//    token into a request; it asks bankmlCreds.fetchWithToken, which sends only to huggingface.co hosts.
// Who speaks is chosen apart from who answers: bankML or Savante, each from its own .persona and .context.
// The questions are asked in the ultimate input field (uif-space.js): each response window is its own conversation;
// this module is its engine (window.bankmlSpaceEngine), and it reports every answer's measurements to the page's
// controls and diagnostics (space-dashboard.js) through window.bankmlStats and the "bankml:stats" event.
// What a window shows is kept in three parts: the ANSWER (only the model's text, streamed piece by piece), the
// page's progress line while it works, and the DELIVERY line after it (the engine's token counts, time to first
// token, speed, where it ran, the receipt's verdict). No completion tokens counted is "No delivery from the model",
// never text; a turn joins the conversation only when an answer was delivered.
// Nothing an engine or a provider returns becomes markup: text only.
import * as browseml from "./browseml.js";
import * as fetcher from "./browseml-fetch.js";

const $ = (id) => document.getElementById(id);
const DEFAULT_ENDPOINT = "http://127.0.0.1:18093";
const DEFAULT_MODEL = "Qwen/Qwen3-8B";
let persona = null;

// who answers: the tab chosen over the conversation; the free CPU first and the default
const MODES = ["space", "browse", "local", "hf"];
let current = "space";
const mode = () => current;
const U = () => window.BankmlSpaceUIF || {};            // uif-space.js: askSpace, the device checks, loadCreds
const creds = () => window.bankmlCreds || null;          // bankml-creds.js (GPL-3.0-only), when the page loaded it
const log = (kind, text) => window.bankmlLog && window.bankmlLog(kind, text);  // the Logs tab (space-tabs.js)
// The conversation sent with a question: at least the last 12 messages, trimmed four exchanges at a time, so the
// engine's cached prompt holds from turn to turn (a window that slid every turn made it read everything again).
const windowed = (h, keep = 12, step = 8) => h.slice(Math.floor(Math.max(0, h.length - keep) / step) * step);

// ── settings the visitor controls (the page's controls, space-dashboard.js), kept in this browser ─────────────────
const SETTINGS_KEY = "bankml.space.settings";
// the context a device that reports little memory is advised (the input field's bundle uses the same rule, caps.ctxCapFor),
// and the default there: a phone that reports 4 GB starts at 2048, not 4096
const ctxCap = () => { const m = navigator.deviceMemory; return typeof m !== "number" ? null : m <= 2 ? 1024 : m <= 4 ? 2048 : null; };
const DEFAULTS = { threads: 0 /* 0: every core */, ctx: Math.min(4096, ctxCap() || 4096), maxTokens: 256, lean: true, removeAfterSession: false, spaceModel: "ternary" };
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
    // working (downloading, reading, writing) in orange; done and verified in green; failed in red
    box.classList.toggle("is-ok", /^✓/.test(text || ""));
    box.classList.toggle("is-bad", /^✗/.test(text || ""));
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
  // what the speaker's prompt costs to read, lean and full: the dashboard's estimate follows who speaks
  stats.prompt = { speaker: P.label, lean: persona.system_prompt.length, full: baseSystem().length };
  stats.emit();
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
const receiptVerdict = (rc, ok) => rc
  ? `${ok ? "✓" : "✗"} receipt — bankML ${rc.bankml} · model sha256 ${String(rc.model_sha256 || "").slice(0, 12)}… · answer sha256 ${ok ? "matches the text received" : "does NOT match"}`
  : "✗ no receipt came with this answer";

// ── the session counter: the engines' own token counts, delivered answers only; while one streams, its pieces ─────
const tally = { in: 0, out: 0, turns: 0 };
const live = (t) => { const e = $("c_live"); if (e) e.textContent = t || ""; };
function counted(promptN, completionN, last) {
  tally.in += Number(promptN) || 0; tally.out += Number(completionN) || 0; tally.turns += 1;
  for (const [id, v] of [["c_in", tally.in], ["c_out", tally.out], ["c_turns", tally.turns]]) { const e = $(id); if (e) e.textContent = String(v); }
  live(""); const l = $("c_last"); if (l) l.textContent = last || "";
}
// ── the DELIVERY line, under the answer and apart from it ──────────────────────────────────────────────────────
const sec1 = (ms) => (ms === null || ms === undefined ? "?" : (ms / 1000).toFixed(1) + " s");
function delivery({ who, prompt, completion, ttft, tps, total, where, verdict }) {
  return `delivered by the model — ${who}: ${prompt ?? "?"} prompt → ${completion ?? "?"} completion tokens` +
    ` · first token ${sec1(ttft)}` + (tps ? ` · ${Number(tps).toFixed(2)} tok/s` : "") + ` · total ${sec1(total)} · ${where} · ${verdict}`;
}
const noDelivery = (who, why = "counted no completion tokens") =>
  `No delivery from the model: ${who} ${why}, so there is no answer to show. Nothing was added to this conversation.`;
/** Nothing delivered: thrown by an answerer, said by the engine (below) in place of any text that streamed. */
class NoDelivery extends Error { constructor(line) { super(line); this.line = line; } }
// errors, said as what to do about them
function hint(m, msg) {
  if (/\b402\b|credit|payment required/i.test(msg)) return " — Hugging Face reports no inference credit left (402): free accounts have no included inference credit since 2026-10-07; this needs PRO or purchased credits. The free CPU tab and browseML are free.";
  if (/\b404\b|not supported|no provider|model_not_supported/i.test(msg)) return " — no live provider serves that model (404): try Qwen/Qwen3-8B, or another answerer.";
  if (m === "local" && /failed to fetch|networkerror|load failed|not reachable|TypeError/i.test(msg))
    return ` — the browser did not reach ${endpoint()}: the card under “your own bankML” shows the ways forward (browseML now, install steps for this device, the free CPU tab, Hugging Face).`;
  if (m === "space" && /connection|network|failed to fetch|errored/i.test(msg)) return " — the free CPU Space could not be reached or dropped the connection: ask again, or use browseML.";
  if (m === "browse" && /download|HTTP \d|stopped at|longer than/i.test(msg)) return " — the model's download failed: check the connection and press Retry on the browseML card (after a dropped connection it continues from the bytes that arrived).";
  return "";
}

// ── your own bankML ───────────────────────────────────────────────────────────────────────────────────────────
const endpoint = () => $("endpoint").value.trim().replace(/\/+$/, "") || DEFAULT_ENDPOINT;
async function connect() {
  const ep = endpoint();
  $("localstatus").textContent = "connecting…"; $("localstatus").className = "status is-working";
  log("connect", ep);
  const t0 = performance.now();
  const b = await get(ep, "/bankml");
  if (!b) {
    stats.local = null; stats.emit();
    $("localstatus").textContent = `not reachable at ${ep}: the ways forward are below`;
    $("localstatus").className = "status is-bad";
    log("error", `${ep} not reachable (bankml serve not running, started without --allow-origin ${location.origin}, or the browser's local-network permission)`);
    showNotFound();
    return false;
  }
  const v = b.verified || {};
  stats.local = { endpoint: ep, rtt_ms: performance.now() - t0, verified: v, resident: b.resident }; stats.emit();
  log("connect", `${ep} answered in ${(performance.now() - t0).toFixed(0)} ms`);
  hideNotFound();
  $("localstatus").textContent = v.guard === "play"
    ? `✓ connected: ${v.name || b.resident || "model"} · bankML ${v.bankml} · sha256 ${String(v.model_sha256 || "").slice(0, 12)}…`
    : "connected, but no verified model is loaded yet";
  $("localstatus").className = "status " + (v.guard === "play" ? "is-ok" : "is-warn");
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
  let first = null, text = "", receipt = null, timings = {}, pieces = 0;
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
      if (piece) { if (first === null) { first = performance.now(); activity.set("✍ your bankML is writing"); progress(`your bankML read the prompt in ${secs((first - t0) / 1000)} · writing`); log("answer", `first token after ${((first - t0) / 1000).toFixed(1)} s`); } text += piece; live(`streaming · ${++pieces} pieces`); yield piece; }
    }
  }
  activity.clear(); live("");
  if (timings.prompt_n !== undefined) progress(`your bankML read ${timings.prompt_n} tokens (${Number(timings.prompt_per_second || 0).toFixed(1)} tok/s) · wrote ${timings.predicted_n} (${Number(timings.predicted_per_second || 0).toFixed(1)} tok/s)`);
  const ok = !!receipt && (await sha256hex(text)) === receipt.response_sha256;
  const promptN = timings.prompt_n ?? receipt?.prompt_tokens, completion = timings.predicted_n ?? receipt?.completion_tokens ?? 0;
  const total = performance.now() - t0, ttft = first === null ? null : first - t0;
  record({ mode: "local", prompt_n: promptN, predicted_n: completion, cache_n: timings.cache_n,
           prompt_tps: timings.prompt_per_second, gen_tps: timings.predicted_per_second, ttft_ms: ttft, total_ms: total, ok });
  if (!completion || !text.trim()) { log("warn", "no delivery: your bankML counted no completion tokens"); throw new NoDelivery(noDelivery("your bankML")); }
  log(receipt ? (ok ? "ok" : "error") : "warn", receipt ? `receipt ${ok ? "✓" : "✗"} · ${receipt.prompt_tokens} + ${receipt.completion_tokens} tokens · ${(total / 1000).toFixed(1)} s` : "no receipt came with this answer");
  const name = stats.local?.verified?.name || "the model it verified";
  counted(promptN, completion, `last: ${completion} tokens · your own bankML`);
  done({ ok, line: delivery({ who: `your own bankML (${name})`, prompt: promptN, completion, ttft, tps: timings.predicted_per_second, total,
                              where: `on your machine (bankml serve at ${ep})`, verdict: receiptVerdict(receipt, ok) }) });
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
  // never a download without the visitor's consent: a model not yet kept here goes through the consent card
  if (!browseml.isLoading() && !(await browseml.isCached())) { showConsent(); return false; }
  const lb = $("browseload"); if (lb) lb.disabled = true;
  const say = (t, frac) => { $("browsestatus").textContent = t; $("browsestatus").className = "status is-working"; const pb = $("browseprogress"); pb.hidden = frac === undefined; if (frac !== undefined) pb.value = frac; if (progressLine) progressLine(t); };
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
    say(done); $("browseprogress").hidden = true; $("browsestatus").className = "status is-ok";
    activity.set(done); setTimeout(() => { if (stats.activity === done) activity.clear(); }, 6000);
    log("ok", `browseML: ${v.model} verified · sha256 ${v.sha256.slice(0, 16)}… · ${v.threads} thread${v.threads > 1 ? "s" : ""}${v.threads > 1 ? "" : " (this page is not cross-origin isolated, or the browser has no SharedArrayBuffer)"} · context ${v.ctx} · engine ${v.engine}`);
    if (lb) lb.hidden = true;
    fetcher.reset("done"); renderConsent();
    return true;
  } catch (e) {
    say("browseML could not load: " + (e.message || e) + hint("browse", String(e.message || e))); $("browseprogress").hidden = true; $("browsestatus").className = "status is-bad";
    activity.set("✗ browseML could not load: " + (e.message || e));
    log("error", "browseML could not load: " + (e.message || e));
    return false;
  } finally {
    if (lb) lb.disabled = false;
    renderConsent();
  }
}
/** Stop browseML and load it again with the current settings (threads, context); the model stays cached. */
export async function reloadBrowse() {
  browseml.unload();
  stats.browse = null; stats.emit();
  return loadBrowse();
}
async function* askBrowse(hist, message, done, progress = () => {}) {
  if (!browseml.isReady()) {
    // a first download is the visitor's choice (248 MB): the question waits while the in-page consent card (the
    // browseML tab) says what will happen and what it needs; nothing downloads until "Download and verify" is
    // pressed, and the question is asked again once bankML has verified the model. A cached model loads in seconds.
    if (!browseml.isLoading() && !(await browseml.isCached())) {
      const mb = (browseml.MODEL.bytes / 1e6).toFixed(0);
      pendingQuestion = message;
      showConsent(true);
      log("mode", "browseML: the consent card is shown; nothing is downloaded until Download and verify is pressed");
      throw new NoDelivery(noDelivery("bankML in this browser", `has no model yet`) + ` The browseML card above says what the ${mb} MB download needs: press “Download and verify” there and this question is asked again once bankML has verified the model — or choose the free CPU tab.`);
    }
    const mirror = () => stats.activity && progress(stats.activity.replace(/^\S+ /, ""));
    window.addEventListener("bankml:activity", mirror);
    progress(browseml.isLoading() ? "waiting for bankML to finish loading…" : "starting bankML in this browser (the model is in this browser's cache; bankML verifies it again)…");
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
      text += piece; live(`streaming · ${n} pieces`); push(piece);
    },
    (level, t) => log(level === 0 ? "error" : "engine", t)).then((d) => { result = d; end(); }, (e) => { activity.clear(); live(""); fail(e); }));
  for await (const p of pieces) yield p;
  live("");
  lastBrowse = result;
  const receipt = result.bankml_receipt, tm = result.timings || {};
  const ok = !!receipt && (await sha256hex(text)) === receipt.response_sha256;
  progress(`read ${tm.prompt_n ?? "?"} tokens in ${secs((tm.prompt_ms || 0) / 1000)} (${Number(tm.prompt_per_second || 0).toFixed(1)} tok/s${tm.cache_n ? `, ${tm.cache_n} reused` : ""}) · wrote ${tm.predicted_n ?? n} in ${secs((tm.predicted_ms || 0) / 1000)} (${Number(tm.predicted_per_second || 0).toFixed(2)} tok/s) · ${thr}`);
  activity.clear();
  const promptN = tm.prompt_n ?? result.usage?.prompt_tokens, completion = tm.predicted_n ?? result.usage?.completion_tokens ?? 0;
  const total = performance.now() - t0, ttft = first === null ? null : first - t0;
  record({ mode: "browse", prompt_n: promptN, predicted_n: completion, cache_n: tm.cache_n, prompt_tps: tm.prompt_per_second,
           gen_tps: tm.predicted_per_second, ttft_ms: ttft, total_ms: total, ok, memory: browseml.memoryBytes });
  if (!completion || !text.trim()) { log("warn", "no delivery: bankML in this browser counted no completion tokens"); throw new NoDelivery(noDelivery("bankML in this browser")); }
  log(receipt ? (ok ? "ok" : "error") : "warn", receipt
    ? `receipt ${ok ? "✓" : "✗"} · ${promptN} + ${completion} tokens · ${(total / 1000).toFixed(1)} s · ${Number(tm.predicted_per_second || 0).toFixed(2)} tok/s`
    : "no receipt came with this answer");
  counted(promptN, completion, `last: ${completion} tokens · bankML in this browser`);
  done({ ok, line: delivery({ who: `bankML in this browser (${String(v.model).replace(/\.gguf$/, "")}, browseML)`, prompt: promptN, completion, ttft,
                              tps: tm.predicted_per_second, total, where: `in this browser on your CPU (WebAssembly, ${thr})`, verdict: receiptVerdict(receipt, ok) }) });
  hist.push({ role: "user", content: message }, { role: "assistant", content: text });
}

// ── a Hugging Face provider (not bankML), through the credential module ─────────────────────────────────────────
// The token never reaches this file: bankml-creds.js (GPL-3.0-only) keeps it and attaches it, and only to
// https://huggingface.co or https://router.huggingface.co. Without that module the page offers no sign-in.
const signedIn = () => { try { return !!(creds() && creds().status().signedIn); } catch { return false; } };
const ROUTER = "https://router.huggingface.co/v1/chat/completions";
async function* askProvider(hist, message, done, progress = () => {}) {
  if (!creds()) throw new Error("the credential module (bankml-creds.js) is not loaded on this page, so there is no Hugging Face sign-in here");
  if (!signedIn()) throw new Error("sign in to Hugging Face first (the Hugging Face tab); free accounts have no included inference credit since 2026-10-07");
  const model = $("model").value.trim() || DEFAULT_MODEL;
  if (!/^[\w.-]+\/[\w.-]+(:[\w.-]+)?$/.test(model)) throw new Error("the model must be a Hugging Face repository id, owner/name");
  // the persona, told the truth about where it is running
  const ctx = passages(message);
  if (ctx.ids.length) log("context", "passages: " + ctx.ids.join(", "));
  const who = PERSONAS[speaker].label;
  const sys = system() + (ctx.text ? "\n\n" + ctx.text : "") + `\n\nWHERE THIS REPLY COMES FROM: you speak as ${who} — the design, the rule, the voice — but this particular reply is generated by ${model} through a Hugging Face inference provider, not by bankML's engine: there is no bankML arithmetic, no receipt and no SELF block behind it. Answer as ${who} would; when asked about your speed, use, receipt, verification or what is running right now, say plainly that this reply comes from ${model}, not from bankML, and that bankML in the browser (browseML) or the visitor's own bankML gives verified answers.`;
  const t0 = performance.now();
  let text = "", shown = "", first = null;
  activity.tick((s) => { const t = `asking ${model} through a Hugging Face provider (not bankML) · ${secs(s)}`; progress(t); return ["☁ " + t]; });
  // Qwen3 and other thinking models: /no_think in the prompt (honoured by the model itself) and enable_thinking off
  // (honoured by some providers) — otherwise the whole budget can go to reasoning and no answer arrives
  const r = await creds().fetchWithToken(ROUTER, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, stream: true, max_tokens: Math.max(settings.maxTokens, 512),
      chat_template_kwargs: { enable_thinking: false }, stream_options: { include_usage: true },
      messages: [{ role: "system", content: sys + "\n/no_think" }, ...hist.slice(-12), { role: "user", content: message }] }),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text().catch(() => "")).slice(0, 300)}`);
  let usage = null, pieces = 0, buf = "";
  const reader = r.body.getReader(), dec = new TextDecoder();
  for (;;) {
    const { done: fin, value } = await reader.read();
    if (fin) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (!line.startsWith("data:") || line === "data: [DONE]") continue;
      let chunk; try { chunk = JSON.parse(line.slice(5)); } catch { continue; }
      if (chunk?.error) throw new Error(String(chunk.error.message || chunk.error));
      if (chunk?.usage) usage = chunk.usage;
      const d = chunk?.choices?.[0]?.delta || {};
      if (!d.content) continue;
      text += d.content;
      const now = answerOnly(text);
      if (now.length > shown.length && now.startsWith(shown)) { if (first === null) { first = performance.now(); activity.set("✍ the provider is writing"); } live(`streaming · ${++pieces} pieces`); yield now.slice(shown.length); shown = now; }
    }
  }
  activity.clear(); live("");
  const total = performance.now() - t0, ttft = first === null ? null : first - t0;
  const label = "not bankML · no receipt · your quota";
  record({ mode: "hf", prompt_n: usage?.prompt_tokens, predicted_n: usage?.completion_tokens, ttft_ms: ttft, total_ms: total, ok: false });
  // with no text (perhaps only the model's reasoning), or no completion tokens counted, nothing was delivered
  if (!shown.trim() || (usage && !usage.completion_tokens)) {
    throw new NoDelivery(noDelivery(`${model} via a Hugging Face provider`, usage && !usage.completion_tokens ? "counted no completion tokens" : "sent no answer text (perhaps only its reasoning)") + ` · ${label}`);
  }
  counted(usage?.prompt_tokens, usage?.completion_tokens, `last: ${usage ? usage.completion_tokens + " tokens" : pieces + " pieces (the provider sent no token count)"} · ${model} (not bankML)`);
  done({ ok: false, receipt: false, line: delivery({ who: `${model} via a Hugging Face provider`, prompt: usage?.prompt_tokens, completion: usage?.completion_tokens,
                                     ttft, total, where: "at a Hugging Face Inference Provider" + (usage ? "" : ` (it sent no token count; ${pieces} pieces)`), verdict: label }) });
  hist.push({ role: "user", content: message }, { role: "assistant", content: shown });
}

// ── bankML · free CPU: Bonsai on Hugging Face's free CPU, as PYTHAI/savante renders it (uif-space.js askSpace) ────
// The answer is the model's text only; the Space's own footer ("— sAGI …") is split off and shown as the Space's; the
// delivery line is the Space's own stats. Until bankML runs on that Space every answer carries the honest label.
const SPACE_NAME = "Gregory-L/mindXhfgradio";
const spaceHonest = () => U().SPACE_HONEST || "Bonsai on Hugging Face's free CPU — llama.cpp, not bankML's engine, no receipt";
async function* askFree(hist, message, done, progress = () => {}) {
  const askSpace = U().askSpace;
  if (!askSpace) throw new Error("the input field's bundle (uif-space.js) did not load, so the free CPU cannot be asked: reload the page");
  const ctx = passages(message);
  if (ctx.ids.length) log("context", "passages: " + ctx.ids.join(", "));
  const t0 = performance.now();
  let first = null, text = "", verdict = null;
  activity.tick((s) => [`☁ asking the free CPU (${SPACE_NAME}) · ${secs(s)} — a public Space: what you type is processed there`]);
  const say = (t) => { progress(t); };
  try {
    for await (const piece of askSpace(system() + (ctx.text ? "\n\n" + ctx.text : ""), hist, message, (v) => { verdict = v; },
                                       settings.spaceModel === "bonsai" ? "bonsai" : "ternary", settings.maxTokens, say)) {
      if (first === null) { first = performance.now(); activity.set("✍ the free CPU is writing"); log("answer", `first words after ${((first - t0) / 1000).toFixed(1)} s`); }
      text += piece; live(`streaming · ${text.length} characters`); yield piece;
    }
  } finally { activity.clear(); live(""); }
  const v = verdict || {}, u = v.usage || {}, total = performance.now() - t0, ttft = first === null ? null : first - t0;
  const bankmlThere = !!(U().spaceIsBankml && U().spaceIsBankml());
  record({ mode: "space", prompt_n: u.prompt_tokens, predicted_n: u.completion_tokens, ttft_ms: ttft, total_ms: total, ok: !!v.ok });
  if (!u.completion_tokens || !text.trim()) {
    log("warn", "no delivery: the free CPU counted no completion tokens");
    throw new NoDelivery(noDelivery(`the free CPU (${SPACE_NAME})`) + (bankmlThere ? "" : ` · ${spaceHonest()}`));
  }
  log(bankmlThere ? (v.ok ? "ok" : "error") : "warn", bankmlThere ? `receipt ${v.ok ? "✓" : "✗"} from bankML on the free CPU` : `free CPU: ${u.completion_tokens} tokens · ${spaceHonest()}`);
  counted(u.prompt_tokens, u.completion_tokens, `last: ${u.completion_tokens} tokens · the free CPU${bankmlThere ? "" : " (not bankML)"}`);
  // three lines under the answer: the Space's delivery, the honest label, and the Space's footer (not the model's words)
  const lines = [v.delivery || `delivered by the model — ${u.prompt_tokens ?? "?"} prompt → ${u.completion_tokens} completion tokens · free public CPU (${SPACE_NAME})`,
    bankmlThere ? (U().SPACE_BANKML || "bankML on Hugging Face's free CPU — a receipt on the answer, checked here") : spaceHonest()];
  if (v.footer) lines.push(`footer added by the Space (not the model's words): ${v.footer}`);
  // no receipt by design (llama.cpp on the free CPU) is not a failed receipt: the pill then reads a neutral "no receipt"
  done({ ok: !!v.ok, receipt: bankmlThere, line: lines.join("\n") });
  hist.push({ role: "user", content: message }, { role: "assistant", content: text });
}

window.bankmlLoadBrowse = loadBrowse;
window.bankmlReloadBrowse = reloadBrowse;

// ── the engine the input field asks (uif-space.js) ─────────────────────────────────────────────────────────────
const statusListeners = new Set();
const announce = () => statusListeners.forEach((f) => f());
window.addEventListener("bankml:stats", announce);
// A failure (nothing delivered, or an error) is the page's line, never the answer's: it takes the place of the window's
// progress line, in red (markWindows, below), and the receipt pill goes back to "receipt —" — there was no answer to
// check. Text that already streamed is taken back the only way the field allows, by throwing: the field then puts
// "Error: …" in the answer's place (hidden by the page, the line says it) and keeps the turn out of its .history.
function failed(line, done, progress, streamed) {
  progress(line);
  try { done(null); } catch { /* the field sets the pill from null ("receipt —") before it reads .line, which throws */ }
  if (streamed) throw new Error(line);
}
const HOW = { space: () => "the free CPU (not bankML)", browse: () => "browseML", local: () => "your bankML", hf: () => "provider " + $("model").value };
const WHO_FAILED = { space: "the free CPU did not answer: ", browse: "bankML in this browser did not answer: ", local: "your bankML did not answer: ", hf: "the provider did not answer (not bankML): " };
let pendingQuestion = null;   // a question asked before browseML was there: asked again once bankML has verified the model
window.bankmlSpaceEngine = {
  async *ask(hist, message, done, _window, progress = () => {}) {
    const m = mode();
    let streamed = "";
    try {
      if (!persona) throw new Error("the persona could not be read: reload the page");
      window.bankmlDismissHighlights && window.bankmlDismissHighlights();  // the highlights give way to the conversation
      log("ask", `${PERSONAS[speaker].label} via ${HOW[m]()}: ${message.length > 80 ? message.slice(0, 79) + "…" : message}`);
      const answer = m === "space" ? askFree(hist, message, done, progress) : m === "browse" ? askBrowse(hist, message, done, progress)
        : m === "local" ? askLocal(hist, message, done, progress) : askProvider(hist, message, done, progress);
      for await (const piece of answer) { streamed += piece; yield piece; }
    } catch (e) {
      activity.clear(); live("");
      if (e instanceof NoDelivery) return failed(e.line, done, progress, streamed);
      const msg = String(e.message || e).slice(0, 300);
      log("error", WHO_FAILED[m] + msg);
      // your own bankML did not answer: one card with the ways forward, instead of a bare error
      if (m === "local" && !streamed && /failed to fetch|networkerror|load failed|not reachable|TypeError|HTTP 403/i.test(msg)) showNotFound(message);
      return failed("✗ " + WHO_FAILED[m] + msg + hint(m, msg), done, progress, streamed);
    }
  },
  status() {
    const m = mode(), who = PERSONAS[speaker].label;
    if (m === "space") return `${who} · free CPU · not bankML, no receipt`;
    if (m === "browse") return stats.browse ? `${who} · ✓ ${stats.browse.model.replace(/\.gguf$/, "")} · ${stats.browse.threads} thr` : `${who} · browseML not loaded`;
    if (m === "local") return stats.local?.verified?.guard === "play" ? `${who} · ✓ ${stats.local.verified.name || "your bankML"}` : `${who} · your bankML: connect`;
    return `${who} · Hugging Face (not bankML)`;
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
    if (m === "space") {
      const t0 = performance.now(), ok = await fetch(`https://huggingface.co/api/spaces/${SPACE_NAME}`, { cache: "no-store" }).then((r) => r.ok, () => false);
      return { ok, line: ok ? `the free CPU's Space ${SPACE_NAME}: the Hub answered in ${(performance.now() - t0).toFixed(0)} ms (the Space itself may still be asleep)` : `the Hub did not answer for ${SPACE_NAME}` };
    }
    return { ok: signedIn(), line: signedIn() ? "the provider answers through router.huggingface.co (not timed here)" : "not signed in to Hugging Face" };
  },
  async diag() { return window.bankmlDiag ? window.bankmlDiag() : { title: "diag", sections: [] }; },
};

// ── wiring ─────────────────────────────────────────────────────────────────────────────────────────────────────
const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids.filter((k) => k !== null && k !== undefined && k !== false)); return e; };
const MB = (b) => `${(b / 1e6).toFixed(b < 1e7 ? 1 : 0)} MB`;
// the carrier note under the tabs: where it runs · what it costs · what is private
const CARRIER = {
  space: () => `${spaceHonest()}. Answers at once: ${settings.spaceModel === "bonsai" ? "Bonsai-8B (1-bit)" : "Ternary-Bonsai-8B"} on the public Space ${SPACE_NAME}, as PYTHAI/savante asks it — the answer, the Space's footer apart, and the delivery line from the Space's own stats. Free. Public: what you type is processed on that public Space. For bankML's own verified answers use browseML or your own bankML.`,
  browse: () => `bankML in this browser, on your CPU: bankML compiled to WebAssembly, with ${browseml.MODEL.title} downloaded once (${(browseml.MODEL.bytes / 1e6).toFixed(0)} MB, only when you press Download and verify) and checked against its sha256 pin; a receipt on every answer, checked here. Free. Private: nothing you ask leaves this computer. Once loaded it stays available; the free CPU tab stays too.`,
  local: () => {
    const host = endpoint().replace(/^https?:\/\//, ""), loop = /^(127\.|localhost(:|$)|\[::1\])/.test(host);
    return `Runs on ${loop ? "your machine" : "the machine you named"}: your bankml serve on ${host}, started with --allow-origin ${location.origin}; a receipt on every answer, checked here. Free: your CPU, your electricity. Private: this page talks only to ${loop ? "that loopback address" : host}.`;
  },
  hf: () => `A Hugging Face Inference Provider (${DEFAULT_MODEL} unless you name another) under your own account, speaking with the chosen persona: not bankML · no receipt · your quota. Free accounts have no included inference credit since 2026-10-07 (PRO or purchased credits). Your question goes to that provider.`,
};
// the environment line, server side: where the model runs
const SERVER_ENV = {
  space: () => `Hugging Face's free public CPU — ${SPACE_NAME}, llama.cpp (not bankML)`,
  browse: () => `this browser — bankML in WebAssembly on your CPU, ${browseml.MODEL.title}`,
  local: () => `your machine — bankml serve at ${endpoint()}`,
  hf: () => "a Hugging Face Inference Provider's servers — not bankML",
};
const TAB_NAME = { space: "bankML · free CPU", browse: "browseML", local: "your own bankML", hf: "Hugging Face" };
function setMode(m, focus = false) {
  if (!MODES.includes(m)) return;
  current = m;
  try { sessionStorage.setItem("bankml.answerer", m); } catch { /* off */ }
  renderMode(focus);
}
// the chosen tab in view inside the scrolled strip (a phone), however it was chosen (a click, the not-found card's
// buttons, the keys): only the strip scrolls, never the page
function inStrip(t) {
  const s = t.parentElement; if (!s || s.scrollWidth <= s.clientWidth) return;
  const sr = s.getBoundingClientRect(), tr = t.getBoundingClientRect();
  if (tr.left < sr.left) s.scrollLeft -= sr.left - tr.left + 8;
  else if (tr.right > sr.right) s.scrollLeft += tr.right - sr.right + 8;
}
function renderMode(focus = false) {
  const m = mode();
  for (const t of document.querySelectorAll("#anstabs [role=tab]")) {
    const on = t.dataset.mode === m;
    t.setAttribute("aria-selected", String(on)); t.tabIndex = on ? 0 : -1;
    if (on) { $("anspanel").setAttribute("aria-labelledby", t.id); if (focus) t.focus(); inStrip(t); }
  }
  $("spacerow").hidden = m !== "space";
  $("browserow").hidden = m !== "browse";
  $("localrow").hidden = m !== "local";
  $("hfrow").hidden = m !== "hf";
  log("mode", TAB_NAME[m] + (m === "space" || m === "hf" ? " (not bankML)" : ""));
  $("modenote").textContent = CARRIER[m]();
  $("env_server").textContent = SERVER_ENV[m]();
  if (m === "browse") { refreshConsent(); }
  stats.mode = m; stats.emit();
}
// a real tablist: click, the arrow keys, Home and End
$("anstabs").addEventListener("click", (e) => { const t = e.target.closest("[role=tab]"); if (t) setMode(t.dataset.mode); });
$("anstabs").addEventListener("keydown", (e) => {
  const all = [...document.querySelectorAll("#anstabs [role=tab]")], i = all.findIndex((t) => t.dataset.mode === mode());
  const j = e.key === "ArrowRight" ? (i + 1) % all.length : e.key === "ArrowLeft" ? (i - 1 + all.length) % all.length : e.key === "Home" ? 0 : e.key === "End" ? all.length - 1 : -1;
  if (j >= 0) { e.preventDefault(); setMode(all[j].dataset.mode, true); }
});

// ── ask a question again (after browseML verified, or from the not-found card): the field sends it as if typed ─────
function reask(q) {
  if (!q) return;
  putInField(q);
  setTimeout(() => {
    const send = document.querySelector(".uif-send");
    if (send && !send.disabled) send.click();
    else log("warn", "the question is in the field: press Enter to ask it");
  }, 60);
}

// ── browseML: the consent card and its controls (nothing downloads until "Download and verify") ──────────────────
const caps = () => (U().readCaps ? U().readCaps() : { wasm: typeof WebAssembly === "object", simd: true, relaxedSimd: false, isolated: !!globalThis.crossOriginIsolated, cores: navigator.hardwareConcurrency || null, memoryGB: navigator.deviceMemory || null, storage: !!navigator.storage?.estimate, persist: !!navigator.storage?.persist });
let consentInfo = { estimate: null, persisted: null, cached: false, platform: "unknown" };
let consentShown = false;   // the card in full (a question waits, or the browseML tab is open with no model)
async function refreshConsent() {
  const [estimate, persisted, cached] = await Promise.all([fetcher.estimate(), fetcher.persisted(), browseml.isCached()]);
  consentInfo = { ...consentInfo, estimate, persisted, cached };
  renderConsent(); renderConsentControls();
}
function showConsent(question = false) {
  consentShown = true;
  if (mode() !== "browse") setMode("browse");
  refreshConsent().then(() => { const c = $("browsecard"); if (c && question) c.scrollIntoView({ block: "center", behavior: "smooth" }); });
}
const isPhone = () => consentInfo.platform === "phone" || consentInfo.platform === "tablet";
function renderConsent() {
  const box = $("browsestate"); if (!box) return;
  const c = caps(), d = fetcher.state(), need = browseml.MODEL.bytes;
  const avail = consentInfo.estimate ? consentInfo.estimate.quota - consentInfo.estimate.usage : null;
  const kids = [];
  if (!c.wasm || !c.simd) {
    kids.push(el("p", { className: "cs-line is-bad", textContent: "browseML is unavailable in this browser: it needs WebAssembly SIMD, which this browser does not have." }),
      el("div", { className: "cs-row" }, el("button", { type: "button", className: "load", textContent: "Use the free CPU tab", onclick: () => setMode("space") })));
    box.replaceChildren(...kids); return;
  }
  if (browseml.isReady()) {
    const b = stats.browse || {};
    kids.push(el("p", { className: "cs-line is-ok", textContent: `verified ✓ — ${String(b.model || browseml.MODEL.name).replace(/\.gguf$/, "")} · bankML ${b.version || ""} · sha256 ${String(b.sha256 || "").slice(0, 12)}… · ${b.threads || 1} thread${(b.threads || 1) > 1 ? "s" : ""} · context ${b.ctx || settings.ctx}. Ask in the field; it stays loaded while this tab is open.` }));
  } else if (browseml.isLoading() || verifying) {
    kids.push(el("p", { className: "cs-line is-working", textContent: "checking sha256… bankML is verifying the model in this browser (the guard, the sha256 of every byte against its pin, then the weights)" }));
  } else if (d.phase === "downloading" || d.phase === "paused" || d.phase === "saving") {
    const pct = d.total ? d.got / d.total : 0;
    const line = d.phase === "saving" ? `keeping the model in this browser's cache… (${MB(d.got)})`
      : d.phase === "paused" ? `paused at ${MB(d.got)} of ${MB(d.total)} — the bytes so far are kept in this tab's memory only (a reload loses them); Resume asks for the rest`
      : `downloading ${MB(d.got)} of ${MB(d.total)}${d.rate ? ` · ${(d.rate / 1e6).toFixed(1)} MB/s · about ${secs(d.eta)} left` : ""} — from huggingface.co, at a pinned revision`;
    kids.push(el("p", { className: "cs-line is-working", textContent: line }),
      el("progress", { className: "cs-bar", max: 1, value: pct, ariaLabel: "the model's download" }),
      el("div", { className: "cs-row" },
        d.phase === "downloading" ? el("button", { type: "button", textContent: "Pause", onclick: () => { fetcher.pause(); log("mode", "browseML: download paused"); } }) : null,
        d.phase === "paused" ? el("button", { type: "button", className: "load", textContent: "Resume", onclick: () => startDownload() }) : null,
        d.phase !== "saving" ? el("button", { type: "button", className: "cs-stop", textContent: "Cancel", onclick: () => { fetcher.cancel(); log("mode", "browseML: download cancelled, nothing kept"); } }) : null));
  } else if (d.phase === "failed") {
    kids.push(el("p", { className: "cs-line is-bad", textContent: `✗ ${d.error || "the download or the verification failed"} — ${d.resumable ? `the ${MB(d.got)} that arrived are kept in this tab's memory (a reload loses them): Retry asks for the rest` : "Retry starts it again"}.` }),
      el("div", { className: "cs-row" }, el("button", { type: "button", className: "load", textContent: d.resumable ? `Retry from ${MB(d.got)}` : "Retry", onclick: () => startDownload() }),
        d.resumable ? el("button", { type: "button", className: "cs-stop", textContent: "Cancel", onclick: () => { fetcher.cancel(); log("mode", "browseML: download cancelled, nothing kept"); } }) : null,
        el("button", { type: "button", textContent: "Use the free CPU tab", onclick: () => setMode("space") })));
  } else if (consentInfo.cached) {
    kids.push(el("p", { className: "cs-line", textContent: `The model is already in this browser's cache (${MB(need)}): starting takes a few seconds, and bankML checks its sha256 again first. Nothing is downloaded.` }),
      el("div", { className: "cs-row" }, el("button", { type: "button", className: "load", textContent: "Start bankML in this browser", onclick: () => verifyAndAsk() })));
  } else {
    kids.push(el("p", { className: "cs-what", textContent: `What happens when you press Download and verify: this browser downloads ${browseml.MODEL.title}, ${MB(need)}, from huggingface.co at a pinned revision; bankML checks its sha256 against the pin before it answers; the model is kept in this browser (Cache Storage) for the next visit, until you remove it. Nothing is downloaded before you press it.` }));
    if (d.phase === "cancelled") kids.push(el("p", { className: "cs-line", textContent: "Cancelled: the download was stopped and nothing was kept." }));
    kids.push(el("dl", { className: "dkv cs-kv" },
      el("dt", { textContent: "storage it needs" }), el("dd", { textContent: MB(need) }),
      el("dt", { textContent: "available here" }), el("dd", { className: avail === null ? "warn" : avail >= need * 1.1 ? "ok" : "warn", textContent: avail === null ? "this browser does not say" : `${MB(avail)} (navigator.storage.estimate)` }),
      el("dt", { textContent: "kept under pressure" }), el("dd", { textContent: consentInfo.persisted === null ? "this browser does not say" : consentInfo.persisted ? "yes (persistent storage granted)" : "not yet — Download asks the browser to keep it (navigator.storage.persist)" })));
    const verdicts = U().browseVerdicts ? U().browseVerdicts(c, { phone: isPhone(), available: avail, ctx: settings.ctx }) : [];
    if (verdicts.length) kids.push(el("ul", { className: "dadvice cs-verdicts" }, ...verdicts.map((v) => el("li", { className: v.level, textContent: v.text }))));
    const tooSmall = avail !== null && avail < need;
    kids.push(el("div", { className: "cs-row" },
      el("button", { type: "button", className: "load", textContent: `Download and verify (${MB(need)})`, disabled: tooSmall, onclick: () => startDownload() }),
      el("button", { type: "button", textContent: "Use the free CPU tab instead", onclick: () => setMode("space") })));
  }
  if (pendingQuestion && !browseml.isReady()) kids.push(el("p", { className: "cs-line", textContent: `Waiting to ask: “${pendingQuestion.length > 90 ? pendingQuestion.slice(0, 89) + "…" : pendingQuestion}” — it is asked again once bankML has verified the model.` }));
  box.replaceChildren(...kids);
}
fetcher.onChange(() => { renderConsent(); const d = fetcher.state(); if (d.phase === "downloading") activity.set(`⬇ downloading the model: ${MB(d.got)} of ${MB(d.total)}${d.rate ? ` · ${(d.rate / 1e6).toFixed(1)} MB/s` : ""}`, d.got / d.total); else if (d.phase === "failed") activity.set("✗ " + (d.error || "the download failed")); else if (d.phase === "paused" || d.phase === "cancelled") activity.clear(); });
async function startDownload() {
  const p = await fetcher.askPersist();   // ask the browser to keep it under storage pressure
  consentInfo.persisted = p;
  log("mode", `browseML: download started by the visitor${p === null ? "" : p ? " · persistent storage granted" : " · persistent storage not granted (the browser may evict it under pressure)"}`);
  if (await fetcher.download()) await verifyAndAsk();
  else renderConsent();
}
let verifying = false;   // between the download's end and bankML's verdict: "checking sha256…"
async function verifyAndAsk() {
  verifying = true; renderConsent();
  let ok = false;
  try { ok = await loadBrowse(); } finally { verifying = false; }
  if (!ok) {
    const why = ($("browsestatus") && $("browsestatus").textContent) || "bankML could not verify or start the model";
    // a model that fails its pin is not kept: Retry downloads it again rather than checking the same bytes
    if (/sha256|guard|pin|does not match/i.test(why)) { await browseml.forget(); log("warn", "browseML: the kept model failed verification and was removed"); }
    fetcher.reset("failed", why);
  }
  consentInfo.cached = await browseml.isCached();
  renderConsent(); renderConsentControls();
  if (ok && pendingQuestion) { const q = pendingQuestion; pendingQuestion = null; if (mode() !== "browse") setMode("browse"); reask(q); }
}
// the controls on the card (CPU, RAM): the same settings as the dashboard below the conversation, kept in this browser
function renderConsentControls() {
  const box = $("browsectl"); if (!box) return;
  if (box.contains(document.activeElement) && /^(INPUT|SELECT)$/.test(document.activeElement.tagName)) return;
  const c = caps(), cores = navigator.hardwareConcurrency || 1, max = browseml.threads(), b = stats.browse;
  const want = Math.max(1, Math.min(max, settings.threads || cores));
  const out = el("output", { textContent: String(want) });
  const thr = el("input", { type: "range", min: 1, max, value: want, disabled: !c.isolated, ariaLabel: "threads",
    oninput: (e) => { out.textContent = e.target.value; }, onchange: (e) => { setSettings({ threads: +e.target.value }); renderConsentControls(); } });
  const est = (ctx) => (U().ramEstimate ? U().ramEstimate(ctx) : browseml.MODEL.bytes + ctx * 28 * 1024 * 2 * 2 + 96e6);
  const ctx = el("select", { ariaLabel: "context size", onchange: (e) => { setSettings({ ctx: +e.target.value }); renderConsentControls(); refreshConsent(); } },
    ...[1024, 2048, 4096].map((n) => el("option", { value: n, selected: settings.ctx === n, textContent: `${n} tokens · about ${MB(est(n))} of RAM (estimate)` })));
  const mo = el("output", { textContent: String(settings.maxTokens) });
  const maxT = el("input", { type: "range", min: 32, max: 512, step: 32, value: settings.maxTokens, ariaLabel: "answer length",
    oninput: (e) => { mo.textContent = e.target.value; }, onchange: (e) => setSettings({ maxTokens: +e.target.value }) });
  const stale = b && (b.threads !== want || b.ctx !== settings.ctx);
  box.replaceChildren(...[
    el("label", { className: "dctl" }, el("span", { textContent: "CPU threads" }), thr, out,
      el("small", { textContent: c.isolated ? `1 to ${max} of ${cores} cores (bankML takes at most ${browseml.MAX_THREADS}) · applies on the next load` : "one: this page is not cross-origin isolated here (no SharedArrayBuffer), so threads are not possible — open pythai-bankml.static.hf.space directly for every core" })),
    el("label", { className: "dctl" }, el("span", { textContent: "context (RAM)" }), ctx,
      el("small", { textContent: `the model ${MB(browseml.MODEL.bytes)} + its cache per token of context; estimates · applies on the next load${b ? ` · WebAssembly memory in use now: ${MB(browseml.memoryBytes)}` : ""}${ctxCap() && settings.ctx > ctxCap() ? ` · this device reports about ${navigator.deviceMemory} GB: keep the context at ${ctxCap()} or less, or the browser may close the tab` : ""}` })),
    el("label", { className: "dctl" }, el("span", { textContent: "answer length" }), maxT, mo, el("small", { textContent: "tokens at most, from the next answer" })),
    stale ? el("div", { className: "cs-row" }, el("span", { className: "cs-line is-working", textContent: `the engine runs ${b.threads} thread${b.threads > 1 ? "s" : ""}, context ${b.ctx}:` }),
      el("button", { type: "button", textContent: "Reload the engine to apply", onclick: () => reloadBrowse().then(() => { renderConsent(); renderConsentControls(); }) })) : null].filter(Boolean));
}

// ── your own bankML did not answer: the not-found card (device recognition here, nothing sent) ───────────────────
let notFoundFor = null;
function hideNotFound() { const c = $("notfound"); if (c) { c.hidden = true; c.replaceChildren(); } notFoundFor = null; }
async function showNotFound(question = null) {
  const c = $("notfound"); if (!c) return;
  notFoundFor = question || notFoundFor;
  const u = U(), ep = endpoint();
  let platform = "unknown";
  try { platform = u.recognise ? u.recognise(await u.readHints()) : "unknown"; } catch { /* hints refused */ }
  consentInfo.platform = platform;
  const plan = u.installPlan ? u.installPlan(platform, location.origin) : { lede: "On Linux x86-64:", commands: ["git clone https://github.com/cryptoAGI/bankml && cd bankml && ./install.sh"], notes: [], suggest: [] };
  const reasons = u.unreachableReasons ? u.unreachableReasons(location.origin, ep) : [];
  const q = notFoundFor;
  const copy = (text) => el("button", { type: "button", className: "linkish", textContent: "copy", onclick: async (e) => { try { await navigator.clipboard.writeText(text); e.target.textContent = "copied"; } catch { e.target.textContent = "select and copy"; } setTimeout(() => (e.target.textContent = "copy"), 1500); } });
  c.replaceChildren(
    el("h3", { textContent: `Your own bankML did not answer at ${ep.replace(/^https?:\/\//, "")}` }),
    el("p", { className: "nf-lede", textContent: `Either bankML is not running here, or this browser kept the page from reaching it (the local-network permission, or serve started without --allow-origin ${location.origin}). Three ways forward${q ? " — your question is kept" : ""}:` }),
    el("ol", { className: "nf-ways" },
      el("li", {}, el("b", { textContent: "Use browseML now" }), " — bankML in this browser: one click downloads the model (248 MB), verifies it, then asks your question again.", " ",
        el("button", { type: "button", className: "load", textContent: "Use browseML now", onclick: () => { pendingQuestion = q; setMode("browse"); refreshConsent().then(() => (consentInfo.cached ? verifyAndAsk() : startDownload())); } })),
      el("li", {}, el("b", { textContent: `Install bankML on this machine` }), ` — ${plan.lede}`,
        plan.commands.length ? el("div", { className: "nf-code" }, el("pre", { textContent: plan.commands.join("\n") }), copy(plan.commands.join("\n"))) : null,
        ...plan.notes.map((n) => el("p", { className: "nf-note", textContent: n })),
        reasons.length ? el("details", {}, el("summary", { textContent: "Installed, but not reachable?" }), el("ul", {}, ...reasons.map((r) => el("li", { textContent: r })))) : null,
        el("p", { className: "nf-note", textContent: "Recognised from what this browser reports about itself (navigator.userAgentData, platform, cores, memory), read here and never sent; no IP address is read." })),
      el("li", {}, el("b", { textContent: "Sign in to Hugging Face, or create an account" }), " — a provider answers under bankML's persona: not bankML, no receipt, your quota.", " ",
        el("button", { type: "button", textContent: "Hugging Face", onclick: () => setMode("hf") }))),
    el("div", { className: "cs-row" },
      el("button", { type: "button", textContent: q ? "Use the free CPU tab now (asks again)" : "Use the free CPU tab now", onclick: () => { setMode("space"); if (q) reask(q); } }),
      el("button", { type: "button", className: "linkish", textContent: "try again", onclick: () => connect() })));
  c.hidden = false;
}

// ── Hugging Face: the credential module's own panel (bankml-creds.js, GPL-3.0-only), or a plain note without it ────
async function mountCreds() {
  const box = $("credspanel"); if (!box) return;
  const m = U().loadCreds ? await U().loadCreds() : creds();
  if (!m) {
    box.replaceChildren(el("p", { className: "status is-warn", textContent: "Hugging Face sign-in is not offered on this page: the credential module (bankml-creds.js) did not load. Everything else works." }));
    return;
  }
  m.mountPanel(box);
  m.onChange(() => { announce(); stats.emit(); });
  log("mode", `credential module bankml-creds.js ${m.version} (${m.licence}) loaded: ${m.status().signedIn ? "signed in" : "not signed in"}`);
}

// the environment line, this side: what this device offers. The browser reports cores, RAM rounded and capped at
// 8 GB (Chromium only) and the GPU's name (WebGL, WebGPU). Read here, never sent.
async function clientEnv() {
  const bits = [`${navigator.hardwareConcurrency || "?"} CPU threads`,
    navigator.deviceMemory ? `${navigator.deviceMemory >= 8 ? "≥ 8" : navigator.deviceMemory} GB RAM` : "RAM not reported by this browser"];
  let gpu = "";
  try {
    const gl = document.createElement("canvas").getContext("webgl");
    const ext = gl && gl.getExtension("WEBGL_debug_renderer_info");
    gpu = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : (gl ? gl.getParameter(gl.RENDERER) : "");
  } catch { /* no WebGL */ }
  const g = String(gpu).replace(/^ANGLE \((.*)\)$/, "$1");
  bits.push(gpu ? `GPU ${g.length > 80 ? g.slice(0, 80).replace(/[\s,(]+\S*$/, "") + "…" : g}` : "GPU not reported");
  $("env_client").title = gpu ? `GPU ${g}` : "";
  let webgpu = "WebGPU no";
  try { const ad = navigator.gpu && await navigator.gpu.requestAdapter(); if (ad) webgpu = "WebGPU yes"; } catch { /* adapter refused */ }
  bits.push(webgpu);
  const c = caps();
  bits.push(c.simd ? "WebAssembly SIMD yes" : "WebAssembly SIMD no (browseML unavailable)", c.isolated ? "threads yes" : "one thread (not isolated)");
  $("env_client").textContent = bits.join(" · ");
}
// the response windows, coloured by what a line says (the field gives every page line one class): a failure or "No
// delivery" in red, a delivery line in green; an answer the engine took back ("Error: ✗…" / "Error: No delivery…",
// see failed()) is hidden — the red line under it already says what happened
const FAIL_LINE = /^(✗|No delivery from the model)/;
function markWindows() {
  for (const e of document.querySelectorAll(".uif-output .uif-msg-system")) {
    const t = e.textContent;
    e.classList.toggle("bk-fail", FAIL_LINE.test(t));
    e.classList.toggle("bk-delivered", t.startsWith("delivered by the model"));
  }
  for (const e of document.querySelectorAll(".uif-output .uif-msg-assistant")) {
    const t = e.textContent, gone = /^Error: (✗|No delivery from the model)/.test(t);
    e.classList.toggle("bk-withdrawn", gone);
    e.classList.toggle("bk-fail", !gone && t.startsWith("Error: "));
    if (e.parentElement && e.parentElement.classList.contains("uif-msg-row")) e.parentElement.classList.toggle("bk-withdrawn", gone);
  }
}
let markQueued = false;
new MutationObserver(() => { if (!markQueued) { markQueued = true; requestAnimationFrame(() => { markQueued = false; markWindows(); }); } })
  .observe(document.body, { subtree: true, childList: true, characterData: true });
// the other Spaces: a link once the Space is public (Hub API); until then its name, marked "not yet public"
document.querySelectorAll("a[data-space]").forEach(async (a) => {
  try {
    const r = await fetch(`https://huggingface.co/api/spaces/${a.dataset.space}`, { cache: "no-store" });
    if (!r.ok) return;
    a.href = `https://huggingface.co/spaces/${a.dataset.space}`; a.rel = "noopener"; a.classList.remove("unpublished"); a.removeAttribute("title");
    const tag = a.nextElementSibling; if (tag && tag.classList.contains("pub-tag")) tag.remove();
  } catch { /* offline, or the Hub refused: the name stays unlinked */ }
});
// an example question: into the input field (it stays the visitor's to send, Enter)
function putInField(text) {
  const ta = document.querySelector(".ultimate-input-field textarea, textarea.ultimate-textarea");
  if (!ta) return;
  Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set.call(ta, text);  // the field's own state
  ta.dispatchEvent(new Event("input", { bubbles: true }));
  ta.focus();
}
document.querySelectorAll("[data-example]").forEach((b) => b.addEventListener("click", () => putInField(b.dataset.example)));
$("connect").addEventListener("click", connect);
$("endpoint").addEventListener("change", () => { if (mode() === "local") { $("env_server").textContent = SERVER_ENV.local(); $("modenote").textContent = CARRIER.local(); } });
$("spacemodel").addEventListener("change", (e) => { setSettings({ spaceModel: e.target.value }); $("modenote").textContent = CARRIER.space(); });
document.querySelectorAll('input[name="speaker"]').forEach((r) => r.addEventListener("change", async () => {
  // a new speaker: the response windows keep their conversations; the next question is answered as the new persona
  if (r.checked) { await loadPersona(r.value); log("persona", `${PERSONAS[r.value].label} speaks from now on`); stats.emit(); }
}));
$("origin").textContent = location.origin;
// "/" or Ctrl/Cmd+K: to the input field (not while typing elsewhere)
document.addEventListener("keydown", (e) => {
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "") || document.activeElement?.isContentEditable;
  if ((e.key === "/" && !typing && !e.ctrlKey && !e.metaKey && !e.altKey) || ((e.ctrlKey || e.metaKey) && (e.key === "k" || e.key === "K"))) {
    const ta = document.querySelector(".ultimate-input-field textarea, textarea.ultimate-textarea");
    if (ta && (e.key !== "/" || document.activeElement !== ta)) { e.preventDefault(); ta.focus(); }
  }
});
// "remove the model after this session" (the dashboard's disk controls): a new visit removes what the last one kept
fetcher.sessionCleanup(() => !!settings.removeAfterSession);
window.bankmlBrowseCard = { refresh: refreshConsent, controls: renderConsentControls, show: () => showConsent(true) };

(async () => {
  try { const m = sessionStorage.getItem("bankml.answerer"); if (MODES.includes(m)) current = m; } catch { /* off */ }
  $("endpoint").value = DEFAULT_ENDPOINT;
  $("model").value = DEFAULT_MODEL;
  $("spacemodel").value = settings.spaceModel === "bonsai" ? "bonsai" : "ternary";
  renderMode();
  clientEnv();
  try { if (U().recognise) consentInfo.platform = U().recognise(await U().readHints()); } catch { /* hints refused */ }
  await loadPersona("bankml");
  refreshConsent();
  window.addEventListener("bankml:stats", () => { if (mode() === "browse") renderConsentControls(); });
  mountCreds();
  // the input field: mounted once its engine (above) exists
  const mountUIF = () => window.BankmlSpaceUIF && window.BankmlSpaceUIF.mount($("uif-root"), window.bankmlSpaceEngine,
    { placeholder: "Ask bankML… Enter sends · Shift+Enter for a new line", terminalPlaceholder: "T mode: help · diag · ping · tile · close all" });
  if (window.BankmlSpaceUIF) mountUIF(); else window.addEventListener("load", mountUIF, { once: true });
})();
