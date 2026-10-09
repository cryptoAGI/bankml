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
// Nothing an engine or a provider returns becomes markup: text only.
import * as browseml from "./browseml.js";
import { oauthLoginUrl, oauthHandleRedirectIfPresent } from "https://cdn.jsdelivr.net/npm/@huggingface/hub@2.17.1/+esm";
import { InferenceClient } from "https://cdn.jsdelivr.net/npm/@huggingface/inference@4.13.28/+esm";

const $ = (id) => document.getElementById(id);
const SESSION = "bankml.oauth";
const DEFAULT_ENDPOINT = "http://127.0.0.1:18093";
const DEFAULT_MODEL = "Qwen/Qwen3-8B";
const history = [];
let persona = null;
let oauth = null;

const mode = () => document.querySelector('input[name="mode"]:checked').value;
const log = (kind, text) => window.bankmlLog && window.bankmlLog(kind, text);  // the Logs tab (space-tabs.js)
// The conversation sent with a question: at least the last 12 messages, trimmed four exchanges at a time, so the
// engine's cached prompt holds from turn to turn (a window that slid every turn made it read everything again).
const windowed = (h, keep = 12, step = 8) => h.slice(Math.floor(Math.max(0, h.length - keep) / step) * step);

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
    $("askstatus").textContent = `${P.label}'s persona could not be read: ${e.message || e}`;
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

// ── the conversation ──────────────────────────────────────────────────────────────────────────────────────────
function bubble(role, text) {
  const d = document.createElement("div");
  d.className = "msg " + role;
  d.textContent = text;
  $("chat").append(d);
  d.scrollIntoView({ block: "nearest" });
  return d;
}
function note(el, text, cls) {
  const p = document.createElement("div");
  p.className = "meta " + (cls || "");
  p.textContent = text;
  el.after(p);
  return p;
}
// "…" alone looks like no reply: say what is happening and how long it has taken, until the first piece arrives
let pending = null;  // the answer being waited for: a failure stops its clock and says so in its bubble
function waiting(out, what) {
  const t0 = Date.now();
  const tick = () => { if (out.dataset.started !== "1") out.textContent = `… ${what} · ${Math.round((Date.now() - t0) / 1000)} s`; };
  tick();
  const id = setInterval(tick, 1000);
  const stop = () => { out.dataset.started = "1"; clearInterval(id); if (pending && pending.out === out) pending = null; };
  pending = { out, stop };
  return stop;
}
// a thinking model may still put its reasoning in the content as <think>…</think>: show the answer only
const answerOnly = (t) => t.replace(/<think>[\s\S]*?(<\/think>|$)/g, "").replace(/^\s+/, "");
async function sha256hex(text) {
  const h = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// ── your own bankML ───────────────────────────────────────────────────────────────────────────────────────────
async function connect() {
  const ep = $("endpoint").value.trim().replace(/\/+$/, "") || DEFAULT_ENDPOINT;
  $("localstatus").textContent = "connecting…";
  log("connect", ep);
  const t0 = performance.now();
  const b = await get(ep, "/bankml");
  if (!b) {
    $("localstatus").textContent = "not reachable — is bankml serve running with --allow-origin " + location.origin + " ? (see below)";
    log("error", `${ep} not reachable (bankml serve not running, or started without --allow-origin ${location.origin})`);
    return false;
  }
  log("connect", `${ep} answered in ${(performance.now() - t0).toFixed(0)} ms`);
  const v = b.verified || {};
  $("localstatus").textContent = v.guard === "play"
    ? `✓ connected: ${v.name || b.resident || "model"} · bankML ${v.bankml} · sha256 ${String(v.model_sha256 || "").slice(0, 12)}…`
    : "connected, but no verified model is loaded yet";
  log(v.guard === "play" ? "ok" : "warn", v.guard === "play"
    ? `verified ${v.name || b.resident || "model"} · bankML ${v.bankml} · sha256 ${String(v.model_sha256 || "").slice(0, 16)}…`
    : "connected, but no verified model is loaded");
  return v.guard === "play";
}
async function askLocal(message) {
  const ep = $("endpoint").value.trim().replace(/\/+$/, "") || DEFAULT_ENDPOINT;
  // the persona first and unchanged, so the engine keeps it cached; SELF (measured now) just before the question
  const self = "SELF (measured by bankML just now):\n" + await selfText(ep);
  const ctx = passages(message);
  if (ctx.ids.length) log("context", "passages: " + ctx.ids.join(", "));
  const out = bubble("assistant", "…");
  const t0 = performance.now();
  let first = null;
  const started = waiting(out, "your bankML is reading the prompt (the first answer of a session reads the whole persona; it can take a few minutes on a busy CPU)");
  let text = "", receipt = null;
  const r = await fetch(ep + "/v1/chat/completions", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "system", content: baseSystem() }, ...windowed(history),
      { role: "system", content: self + (ctx.text ? "\n\n" + ctx.text : "") }, { role: "user", content: message }],
      stream: true, max_tokens: 384 }),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 300)}`);
  const reader = r.body.getReader(), dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (!line.startsWith("data:") || line === "data: [DONE]") continue;
      const d = JSON.parse(line.slice(5));
      if (d.bankml_receipt) { receipt = d.bankml_receipt; continue; }
      const piece = d.choices?.[0]?.delta?.content;
      if (piece) { started(); if (first === null) { first = performance.now(); log("answer", `first token after ${((first - t0) / 1000).toFixed(1)} s`); } text += piece; out.textContent = text; }
    }
  }
  started();
  const ok = receipt && (await sha256hex(text)) === receipt.response_sha256;
  log(receipt ? (ok ? "ok" : "error") : "warn", receipt
    ? `receipt ${ok ? "✓ the answer's sha256 matches" : "✗ the answer's sha256 does NOT match"} · ${receipt.prompt_tokens} + ${receipt.completion_tokens} tokens · ${((performance.now() - t0) / 1000).toFixed(1)} s`
    : "no receipt came with this answer");
  note(out, receipt
    ? `${ok ? "✓" : "✗"} receipt — bankML ${receipt.bankml} · model sha256 ${String(receipt.model_sha256 || "").slice(0, 12)}… · answer sha256 ${ok ? "matches the text received" : "does NOT match the text received"}`
    : "no receipt came with this answer", ok ? "ok" : "bad");
  history.push({ role: "user", content: message }, { role: "assistant", content: text });
}

// ── browseML: bankML in this browser ─────────────────────────────────────────────────────────────────────────
let lastBrowse = null;  // the last answer's measurements, for the SELF block
function browseSelf(v) {
  const t = lastBrowse && lastBrowse.timings || {};
  const n = (x, unit, d = 1) => (x === undefined || x === null ? "not measured" : Number(x).toFixed(d) + unit);
  return "SELF (measured by bankML just now):\n" + [
    `- the model I am running: ${v.model} (sha256 ${v.sha256.slice(0, 16)}…), bankML ${v.version}, verified (guard ${v.guard})`,
    `- where I am running: in the visitor's own web browser, compiled to WebAssembly (browseML), one thread of their CPU — no server`,
    `- prompt reading speed of my last answer: ${n(t.prompt_per_second, " tokens per second")}`,
    `- generation speed of my last answer: ${n(t.predicted_per_second, " tokens per second")}`,
  ].join("\n");
}
async function loadBrowse() {
  if (browseml.isReady()) return true;
  $("browseload").disabled = true;
  try {
    const t0 = performance.now();
    const v = await browseml.load((got, total, phase) => {
      $("browsestatus").textContent = phase === "verify" ? "bankML is verifying the model (guard, then sha256 of all 248 MB)…"
        : phase === "cache" ? "the model is in this browser's cache"
        : `downloading the model: ${(got / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB`;
    }, (level, text) => log(level === 0 ? "error" : level === 1 ? "warn" : "engine", text));
    $("browsestatus").textContent = `✓ ${v.model} verified by bankML ${v.version} · sha256 ${v.sha256.slice(0, 12)}… · ready in ${((performance.now() - t0) / 1000).toFixed(0)} s`;
    log("ok", `browseML: ${v.model} verified · sha256 ${v.sha256.slice(0, 16)}… · engine ${v.engine}`);
    $("browseload").hidden = true;
    return true;
  } catch (e) {
    $("browsestatus").textContent = "browseML could not load: " + (e.message || e);
    log("error", "browseML could not load: " + (e.message || e));
    return false;
  } finally {
    $("browseload").disabled = false;
  }
}
async function askBrowse(message) {
  if (!browseml.isReady()) throw new Error("load bankML in this browser first (the button above the question box’s answers)");
  const v = await browseml.load(() => {}, () => {});
  const ctx = passages(message);
  if (ctx.ids.length) log("context", "passages: " + ctx.ids.join(", "));
  const out = bubble("assistant", "…");
  const t0 = performance.now();
  let first = null, text = "";
  const started = waiting(out, "bankML is reading the prompt in your browser (the first answer reads the whole persona; on one thread this takes a while)");
  const d = await browseml.chat({ messages: [{ role: "system", content: baseSystem() }, ...windowed(history),
      { role: "system", content: browseSelf(v) + (ctx.text ? "\n\n" + ctx.text : "") }, { role: "user", content: message }],
      max_tokens: 256 },
    (piece) => { started(); if (first === null) { first = performance.now(); log("answer", `first token after ${((first - t0) / 1000).toFixed(1)} s`); } text += piece; out.textContent = text; },
    (level, t) => log(level === 0 ? "error" : "engine", t));
  started();
  lastBrowse = d;
  const receipt = d.bankml_receipt;
  const ok = receipt && (await sha256hex(text)) === receipt.response_sha256;
  const tm = d.timings || {};
  log(receipt ? (ok ? "ok" : "error") : "warn", receipt
    ? `receipt ${ok ? "✓ the answer's sha256 matches" : "✗ the answer's sha256 does NOT match"} · ${d.usage.prompt_tokens} + ${d.usage.completion_tokens} tokens · ${((performance.now() - t0) / 1000).toFixed(1)} s · ${Number(tm.predicted_per_second || 0).toFixed(2)} tok/s`
    : "no receipt came with this answer");
  note(out, receipt
    ? `${ok ? "✓" : "✗"} receipt — bankML ${receipt.bankml} in this browser · model sha256 ${String(receipt.model_sha256 || "").slice(0, 12)}… · answer sha256 ${ok ? "matches the text received" : "does NOT match the text received"}`
    : "no receipt came with this answer", ok ? "ok" : "bad");
  history.push({ role: "user", content: message }, { role: "assistant", content: text });
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
async function askProvider(message) {
  const model = $("model").value.trim() || DEFAULT_MODEL;
  if (!/^[\w.-]+\/[\w.-]+$/.test(model)) throw new Error("the model must be a Hugging Face repository id, owner/name");
  // the persona, told the truth about where it is running
  const ctx = passages(message);
  if (ctx.ids.length) log("context", "passages: " + ctx.ids.join(", "));
  const who = PERSONAS[speaker].label;
  const system = baseSystem() + (ctx.text ? "\n\n" + ctx.text : "") + `\n\nWHERE THIS REPLY COMES FROM: you speak as ${who} — the design, the rule, the voice — but this particular reply is generated by ${model} through a Hugging Face inference provider, not by bankML's engine: there is no bankML arithmetic, no receipt and no SELF block behind it. Answer as ${who} would; when asked about your speed, use, receipt, verification or what is running right now, say plainly that this reply comes from ${model}, not from bankML, and that bankML in the browser (browseML) or the visitor's own bankML gives verified answers.`;
  const out = bubble("assistant", "…");
  const started = waiting(out, `asking ${model} through a Hugging Face provider`);
  let text = "", reasoning = "";
  const client = new InferenceClient(oauth.accessToken);
  // Qwen3 and other thinking models: /no_think in the prompt (honoured by the model itself) and enable_thinking off
  // (honoured by some providers) — otherwise the whole budget can go to reasoning and no answer arrives
  const stream = client.chatCompletionStream({ provider: "auto", model, max_tokens: 512,
    chat_template_kwargs: { enable_thinking: false },
    messages: [{ role: "system", content: system + "\n/no_think" }, ...history.slice(-12), { role: "user", content: message }] });
  for await (const chunk of stream) {
    const d = chunk?.choices?.[0]?.delta || {};
    if (d.reasoning_content) reasoning += d.reasoning_content;
    if (d.content) { started(); text += d.content; out.textContent = answerOnly(text) || "…"; }
  }
  started();
  text = answerOnly(text);
  if (!text) {
    out.textContent = reasoning
      ? "The provider returned only the model's reasoning and no answer — ask again, or choose a model that does not think aloud."
      : "The provider returned an empty answer.";
  }
  note(out, `not bankML — ${model} via a Hugging Face provider · no receipt · your quota`, "warn");
  history.push({ role: "user", content: message }, { role: "assistant", content: text });
}

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
}
document.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", renderMode));
$("connect").addEventListener("click", connect);
$("browseload").addEventListener("click", loadBrowse);
$("browseforget").addEventListener("click", async () => { await browseml.forget(); $("browsestatus").textContent = "the model is removed from this browser's cache"; log("mode", "browseML: the cached model was removed"); });
document.querySelectorAll('input[name="speaker"]').forEach((r) => r.addEventListener("change", async () => {
  if (r.checked) { history.length = 0; await loadPersona(r.value); log("persona", `a new conversation with ${PERSONAS[r.value].label}`); }
}));
$("signin").addEventListener("click", async () => {
  if (!window.huggingface?.variables?.OAUTH_CLIENT_ID) { $("who").textContent = "sign-in works only on the Hugging Face Space itself"; return; }
  window.location.href = await oauthLoginUrl({ scopes: window.huggingface.variables.OAUTH_SCOPES });
});
$("signout").addEventListener("click", () => { writeSession(null); oauth = null; renderAuth(); });
$("send").addEventListener("click", async () => {
  const message = $("message").value.trim();
  if (!message || !persona) return;
  if (mode() === "hf" && !signedIn()) { $("askstatus").textContent = "sign in with Hugging Face first"; return; }
  $("message").value = "";
  $("send").disabled = true;
  $("askstatus").textContent = "";
  window.bankmlDismissHighlights && window.bankmlDismissHighlights();  // the highlights give way to the conversation
  const how = { browse: "browseML", local: "your bankML", hf: "provider " + $("model").value }[mode()];
  log("ask", `${PERSONAS[speaker].label} via ${how}: ${message.length > 80 ? message.slice(0, 79) + "…" : message}`);
  bubble("user", message);
  try {
    await (mode() === "browse" ? askBrowse(message) : mode() === "local" ? askLocal(message) : askProvider(message));
  } catch (e) {
    if (pending) { const { out, stop } = pending; stop(); if (!out.textContent || out.textContent.startsWith("…")) { out.textContent = "— no answer (the reason is just below)"; out.classList.add("failed"); } }
    const m = mode(), who = { browse: "bankML in your browser did not answer: ", local: "your bankML did not answer: ", hf: "the provider did not answer: " }[m];
    log("error", who + String(e.message || e).slice(0, 300));
    $("askstatus").textContent = who + String(e.message || e).slice(0, 300)
      + (m === "local" ? " — is bankml serve running with --allow-origin " + location.origin + " ?" : "");
  } finally {
    $("send").disabled = false;
  }
});
$("message").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("send").click(); });
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
})();
