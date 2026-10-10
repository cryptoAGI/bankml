// SPDX-License-Identifier: MIT OR Apache-2.0
// browseML's page: one answerer here, bankML in this browser (browseml.js); the chooser names the others honestly (two
// on the full bankML Space, one coming) without calling them. The input → response pattern of the Savante Space, with every model word written with textContent, never innerHTML. Each answer keeps three things apart:
//   the answer   — only the text the model generated
//   a page note  — anything this page adds (a hidden <think> section, a length cut), labelled as the page's
//   the delivery — the engine's own token counts, time to the first token, tok/s, where it ran, the receipt's verdict
// No completion tokens counted is "No delivery from the model", not text; only a delivered answer joins the history.
import * as browseml from "./browseml.js";

const $ = (id) => document.getElementById(id);
const MB = (b) => (b / 1e6).toFixed(0);
const secs = (s) => (s === null || s === undefined || !isFinite(s) ? "?" : s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);
const WHO = "bankML in this browser";
const plural = (n, one, many = one + "s") => `${n} ${n === 1 ? one : many}`;
// cut a long string at a word boundary near `max`, and say it was cut
const clip = (t, max) => t.length <= max ? t : t.slice(0, max).replace(/[\s,;(]+\S*$/, "") + "…";
const HISTORY_TURNS = 3;  // the last three delivered turns go back to the model; the engine reuses what it has read
const history = [];       // delivered turns only
let busy = false;

// ── the engine log (a details drawer) ─────────────────────────────────────────────────────────────────────────
const logLines = [];
function log(kind, text) {
  logLines.push(`${new Date().toLocaleTimeString()} ${kind.padEnd(6)} ${text}`);
  if (logLines.length > 300) logLines.shift();
  $("englog").textContent = logLines.join("\n");
}
const engineLog = (level, text) => log(level === 0 ? "error" : level === 1 ? "warn" : "engine", text);

// ── the session counter: the engine's own token counts, delivered answers only; while one streams, its pieces ──
const tally = { in: 0, out: 0, turns: 0 };
const live = (t) => { $("c_live").textContent = t || ""; };
function counted(promptN, completionN, last) {
  tally.in += Number(promptN) || 0; tally.out += Number(completionN) || 0; tally.turns += 1;
  $("c_in").textContent = tally.in; $("c_out").textContent = tally.out; $("c_turns").textContent = tally.turns;
  live(""); $("c_last").textContent = last || "";
}

// ── the environment line, this side: what this device offers. Read here, never sent. ─────────────────────────
async function clientEnv() {
  const iso = !!(globalThis.crossOriginIsolated && typeof SharedArrayBuffer !== "undefined");
  const bits = [`${navigator.hardwareConcurrency || "?"} CPU threads`,
    navigator.deviceMemory ? `${navigator.deviceMemory >= 8 ? "≥ 8" : navigator.deviceMemory} GB RAM` : "RAM not reported by this browser"];
  let gpu = "";
  try {
    const gl = document.createElement("canvas").getContext("webgl");
    const ext = gl && gl.getExtension("WEBGL_debug_renderer_info");
    gpu = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : (gl ? gl.getParameter(gl.RENDERER) : "");
  } catch { /* no WebGL */ }
  bits.push(gpu ? `GPU ${clip(String(gpu).replace(/^ANGLE \((.*)\)$/, "$1"), 80)} (not used: the engine runs on the CPU)` : "GPU not reported");
  const n = browseml.threads();
  bits.push(iso ? `cross-origin isolated: bankML takes ${n} thread${n > 1 ? "s" : ""}` : "not cross-origin isolated: bankML takes one thread");
  $("env_client").textContent = bits.join(" · ");
}

// ── loading: the download (once), then bankML's verification, said as it happens ─────────────────────────────
const listeners = new Set();  // answer windows waiting on the load mirror its line
function say(text, cls = "is-working", frac) {
  $("status").textContent = text; $("status").className = "status " + cls;
  const bar = $("bar"); bar.hidden = frac === undefined; if (frac !== undefined) bar.value = frac;
  listeners.forEach((f) => f(text));
}
let verified = null;
async function load() {
  if (browseml.isReady() && verified) return verified;
  if (browseml.isLoading()) return browseml.whenLoaded();  // one download at a time: wait for the one under way
  $("load").disabled = true;
  const t0 = performance.now();
  let dl0 = null, clock = null;
  try {
    verified = await browseml.load((got, total, phase) => {
      if (phase === "download") {
        if (!dl0) dl0 = { t: performance.now(), got };
        const dt = (performance.now() - dl0.t) / 1000, rate = dt > 0.5 ? (got - dl0.got) / dt : 0;
        say(`downloading the model (${browseml.MODEL.title}): ${MB(got)} of ${MB(total)} MB` +
          (rate ? ` · ${(rate / 1e6).toFixed(1)} MB/s · about ${secs((total - got) / rate)} left` : "") + " · once; it stays in this browser's cache",
        "is-working", got / total);
      } else if (phase === "cache") {
        say(`the model is in this browser's cache (${MB(got)} MB): no download`);
      } else if (phase === "verify" && !clock) {
        // the engine starts in a worker, checks the GGUF header, hashes all 248 MB against the pin, then lays out the
        // weights: seconds, not minutes; the clock runs until it says it is verified
        const v0 = performance.now();
        const tick = () => say(`bankML is verifying the model in this browser: the guard, then the sha256 of all ${MB(browseml.MODEL.bytes)} MB against its pin · ${secs((performance.now() - v0) / 1000)}`);
        tick(); clock = setInterval(tick, 500);
      }
    }, engineLog, { threads: browseml.threads(), ctx: 4096 });
    const v = verified;
    say(`✓ ${v.model} verified by bankML ${v.version} · sha256 ${v.sha256.slice(0, 12)}… · ${v.threads} thread${v.threads > 1 ? "s" : ""} · ${v.build || "browseml.wasm"} · ready in ${secs((performance.now() - t0) / 1000)}`, "is-ok");
    log("ok", `verified ${v.model} · sha256 ${v.sha256} · guard ${v.guard} · engine ${v.engine} · ${v.threads} thread(s) · ${v.build || "browseml.wasm"}`);
    $("load").hidden = true;
    return v;
  } catch (e) {
    const m = String(e && e.message || e);
    say("bankML could not load: " + m + hint(m), "is-bad");
    log("error", "load: " + m);
    throw e;
  } finally {
    if (clock) clearInterval(clock);
    $("load").disabled = false;
  }
}

// errors, said as what to do about them
function hint(m) {
  if (/download|HTTP \d|stopped at|longer than|failed to fetch|networkerror|load failed/i.test(m))
    return " — the model's download failed: check the connection and ask again. A partial download is not kept.";
  if (/FORK\.json/.test(m)) return " — the model's pin did not load with the page: reload the page.";
  if (/sha256|guard|pin/i.test(m)) return " — the model failed bankML's check, so it will not answer from it. Clear this site's data to download it again.";
  if (/memory|unreachable|RuntimeError|trap|-99/i.test(m)) return " — the engine stopped (most often out of memory). It loads again on the next question; closing other tabs helps.";
  if (/compile|WebAssembly|\.wasm/i.test(m)) return " — this browser could not start the engine's WebAssembly: a current Chrome, Edge or Firefox runs it.";
  return "";
}

// ── the receipt, checked here ────────────────────────────────────────────────────────────────────────────────
async function sha256hex(text) {
  const h = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
const verdictText = (rc, ok) => rc
  ? `${ok ? "✓" : "✗"} receipt — bankML ${rc.bankml} · model sha256 ${String(rc.model_sha256 || "").slice(0, 12)}… · answer sha256 ${ok ? "matches the text received" : "does NOT match"}`
  : "✗ no receipt came with this answer";
// a thinking model may put its reasoning in <think>…</think>: the answer is shown without it, and the page says so
const answerOnly = (t) => t.replace(/<think>[\s\S]*?(<\/think>|$)/g, "").replace(/^\s+/, "");

// ── the conversation ─────────────────────────────────────────────────────────────────────────────────────────
function bubble(role, label) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  const who = document.createElement("div"); who.className = "who"; who.textContent = label;
  div.append(who);
  $("log").append(div);
  return div;
}
function userBubble(text) {
  const div = bubble("user", "you");
  const body = document.createElement("div"); body.className = "body"; body.textContent = text;
  div.append(body);
  div.scrollIntoView({ block: "nearest" });
}
function answerWindow() {
  const div = bubble("assistant", WHO);
  const card = document.createElement("div"); card.className = "card";
  const el = (cls) => { const e = document.createElement("div"); e.className = cls; card.append(e); return e; };
  const w = { div, card, answer: el("answer"), progress: el("progress"), note: null, delivery: null, actions: el("actions") };
  div.append(card);
  w.show = () => div.scrollIntoView({ block: "nearest" });
  w.fail = (cls, text) => { w.progress.textContent = ""; w.answer.textContent = ""; const e = el(cls); e.textContent = text; w.show(); };
  return w;
}
// the first question offers the one-time download in the answer window, rather than failing or starting it unasked
function offer(w) {
  w.progress.textContent = `bankML in this browser needs its model first: ${browseml.MODEL.title}, ${MB(browseml.MODEL.bytes)} MB from its Hugging Face repository at a pinned revision. Downloaded once, checked against its sha256 pin, and kept in this browser's cache.`;
  return new Promise((resolve) => {
    const yes = document.createElement("button"); yes.className = "go"; yes.textContent = `Download ${MB(browseml.MODEL.bytes)} MB and answer`;
    const no = document.createElement("button"); no.textContent = "Not now";
    yes.onclick = () => { w.actions.replaceChildren(); resolve(true); };
    no.onclick = () => { w.actions.replaceChildren(); resolve(false); };
    w.actions.replaceChildren(yes, no); w.show(); yes.focus();
  });
}

const SYSTEM = (v) => `You are bankML, answering in the visitor's own web browser: bankML's engine compiled to WebAssembly ` +
  `(browseML), running ${v.model} on the visitor's CPU with ${v.threads} thread${v.threads > 1 ? "s" : ""}. bankML ${v.version} ` +
  `verified the model against its sha256 pin before this conversation, and every answer you give carries a receipt that ` +
  `the page checks. No server is involved. Answer briefly and plainly; when you do not know, say so.`;

async function ask() {
  const field = $("message");
  const message = field.value.trim();
  if (!message || busy) return;
  busy = true; $("send").disabled = true;
  field.value = "";
  userBubble(message);
  const w = answerWindow();
  w.show();
  try {
    if (!browseml.isReady()) {
      if (!browseml.isLoading() && !(await browseml.isCached())) {
        if (!(await offer(w))) {
          log("mode", "the one-time download was not started");
          w.fail("nodelivery", `No delivery from the model: ${WHO} has no model yet (the ${MB(browseml.MODEL.bytes)} MB download was not started), so there is no answer to show. Nothing was added to this conversation.`);
          return;
        }
      }
      const mirror = (t) => { w.progress.textContent = t; };
      listeners.add(mirror);
      try { verified = await load(); } finally { listeners.delete(mirror); }
    }
    const v = verified || (verified = await load());
    const thr = `${v.threads} thread${v.threads > 1 ? "s" : ""}`;
    const msgs = [{ role: "system", content: SYSTEM(v) }, ...history.slice(-2 * HISTORY_TURNS), { role: "user", content: message }];
    const maxTokens = Number($("maxtok").value) || 128;
    const t0 = performance.now();
    let first = null, text = "", n = 0, lastShown = 0;
    const reading = setInterval(() => { if (first === null) w.progress.textContent = `reading the prompt in this browser · ${secs((performance.now() - t0) / 1000)} · ${thr}`; }, 500);
    w.progress.textContent = `reading the prompt in this browser · ${thr}`;
    let result;
    try {
      result = await browseml.chat({ messages: msgs, max_tokens: maxTokens }, (piece) => {
        n++;
        const now = performance.now();
        if (first === null) { first = now; log("answer", `first token after ${secs((first - t0) / 1000)}`); }
        text += piece;
        w.answer.textContent = answerOnly(text);
        live(`streaming · ${plural(n, "piece")}`);
        if (now - lastShown > 400) {
          lastShown = now;
          // the pieces as they arrive, not the engine's token count (that comes with the delivery line)
          const pps = n > 1 ? (n - 1) / ((now - first) / 1000) : 0;
          w.progress.textContent = `writing · ${plural(n, "piece")} received${pps ? ` · ${pps.toFixed(2)} pieces/s` : ""} · ${maxTokens} tokens at most`;
          w.show();
        }
      }, engineLog);
    } finally { clearInterval(reading); live(""); }
    w.progress.textContent = "";

    const receipt = result.bankml_receipt, tm = result.timings || {};
    const ok = !!receipt && (await sha256hex(text)) === receipt.response_sha256;
    // the whole prompt is usage.prompt_tokens; timings.prompt_n is only what the engine read anew (it reuses the rest)
    const promptN = result.usage?.prompt_tokens ?? tm.prompt_n, readNew = tm.prompt_n;
    const completion = result.usage?.completion_tokens ?? tm.predicted_n ?? 0;
    const total = performance.now() - t0, ttft = first === null ? null : first - t0;
    const shown = answerOnly(text).trim();
    if (!completion || !shown) {
      log("warn", "no delivery: the engine counted no completion tokens" + (completion ? " outside a <think> section" : ""));
      w.fail("nodelivery", `No delivery from the model: ${WHO} counted no completion tokens${completion ? " outside its <think> section" : ""}, so there is no answer to show. Nothing was added to this conversation.`);
      return;
    }
    w.answer.textContent = shown;
    // the page's own notes, apart from the answer and labelled as the page's
    const notes = [];
    if (shown.length < text.trim().length) notes.push("the model's <think> section is hidden here; the receipt covers the full text");
    if ((result.choices?.[0]?.finish_reason) === "length") notes.push(`the answer reached its length (${maxTokens} tokens) and was cut there`);
    if (notes.length) { w.note = document.createElement("div"); w.note.className = "pagenote"; w.note.textContent = "added by this page, not the model: " + notes.join(" · "); w.card.insertBefore(w.note, w.actions); }
    // the DELIVERY line
    w.delivery = document.createElement("div"); w.delivery.className = "delivery";
    const facts = `delivered by the model — ${WHO} (${String(v.model).replace(/\.gguf$/, "")}, browseML): ${promptN ?? "?"} prompt` +
      (readNew !== undefined && promptN !== undefined && readNew < promptN ? ` (${readNew} read anew, the rest reused)` : "") +
      ` → ${completion} completion tokens` +
      ` · first token ${secs(ttft === null ? null : ttft / 1000)}` + (tm.predicted_per_second ? ` · ${Number(tm.predicted_per_second).toFixed(2)} tok/s` : "") +
      ` · ${secs(total / 1000)} · in this browser on your CPU (WebAssembly, ${thr}${v.build ? ", " + v.build : ""}) · `;
    const verdict = document.createElement("span"); verdict.className = ok ? "ok" : "bad"; verdict.textContent = verdictText(receipt, ok);
    w.delivery.append(facts, verdict);
    w.card.insertBefore(w.delivery, w.actions);
    const turn = tally.turns + 1;  // this answer's number, fixed now: counted() below makes it the session's
    const pop = document.createElement("button"); pop.textContent = "pop out";
    pop.onclick = () => spawn(`${WHO} · ${turn} · ${message}`, shown, w.delivery);
    w.actions.replaceChildren(pop);
    log(receipt ? (ok ? "ok" : "error") : "warn", receipt ? `receipt ${ok ? "✓" : "✗"} · ${promptN} + ${completion} tokens · ${secs(total / 1000)}` : "no receipt came with this answer");
    counted(promptN, completion, `last: ${completion} tokens · ${WHO}`);
    history.push({ role: "user", content: message }, { role: "assistant", content: text });
    w.show();
  } catch (e) {
    const m = String(e && e.message || e);
    log("error", m);
    w.fail("failed", `${WHO} did not answer: ${m.slice(0, 300)}${hint(m)}`);
  } finally {
    busy = false; $("send").disabled = false;
  }
}

// ── pop-outs: an answer in its own window, glass over the page, moved by its title bar ────────────────────────
let spawned = 0;
function spawn(title, text, delivery) {
  const win = document.createElement("section"); win.className = "spawn"; win.setAttribute("role", "dialog"); win.setAttribute("aria-label", title);
  // each one a step down (and, where there is room, right) of the last; on a phone it fills the width between the gutters
  const off = (spawned++ % 6) * 28, phone = innerWidth <= 640, gutter = 16;
  win.style.left = `${phone ? gutter : Math.max(gutter, Math.min(innerWidth - 580, 80 + off))}px`;
  win.style.top = `${phone ? gutter + off : 80 + off}px`;
  const bar = document.createElement("header");
  const b = document.createElement("b"); b.textContent = title;
  const close = document.createElement("button"); close.textContent = "×"; close.setAttribute("aria-label", "close"); close.onclick = () => win.remove();
  bar.append(b, close);
  const body = document.createElement("div"); body.className = "answer"; body.textContent = text;
  const d = delivery.cloneNode(true);  // the delivery line as shown under the answer, its receipt verdict still coloured
  win.append(bar, body, d);
  bar.addEventListener("pointerdown", (e) => {
    if (e.target === close) return;
    const sx = e.clientX - win.offsetLeft, sy = e.clientY - win.offsetTop;
    bar.setPointerCapture(e.pointerId);
    const move = (ev) => { win.style.left = `${Math.max(0, Math.min(innerWidth - 80, ev.clientX - sx))}px`; win.style.top = `${Math.max(0, Math.min(innerHeight - 40, ev.clientY - sy))}px`; };
    bar.addEventListener("pointermove", move);
    bar.addEventListener("pointerup", () => bar.removeEventListener("pointermove", move), { once: true });
  });
  $("spawns").append(win);
  close.focus();
}
addEventListener("keydown", (e) => { if (e.key === "Escape") { const s = $("spawns").lastElementChild; if (s) s.remove(); } });

// ── the field: Enter sends, Shift+Enter a new line; the examples go into the field, still the visitor's to send ─
$("send").addEventListener("click", ask);
$("message").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(); }
});
document.querySelectorAll("[data-example]").forEach((b) => b.addEventListener("click", () => {
  $("message").value = b.dataset.example; $("message").focus();
}));
$("load").addEventListener("click", () => load().catch(() => {}));
$("tmode").addEventListener("click", () => {
  const on = $("screen").classList.toggle("t");
  $("tmode").setAttribute("aria-pressed", String(on));
  try { localStorage.setItem("browseml-t", on ? "1" : ""); } catch {}
});
try { if (localStorage.getItem("browseml-t")) { $("screen").classList.add("t"); $("tmode").setAttribute("aria-pressed", "true"); } } catch {}
try { const m = localStorage.getItem("browseml-maxtok"); if (m) $("maxtok").value = m; } catch {}
$("maxtok").addEventListener("change", () => { try { localStorage.setItem("browseml-maxtok", $("maxtok").value); } catch {} });

clientEnv();
browseml.isCached().then((yes) => {
  if (yes) { $("load").textContent = "Load bankML (the model is in this browser's cache)"; say("The model is in this browser's cache: it loads with your first question, in seconds.", ""); }
});
