// SPDX-License-Identifier: MIT OR Apache-2.0
// The page behind the input field: this computer's controls and diagnostics. What the visitor's own machine gives
// bankML (threads, memory for the context, the length of an answer, a lean or full prompt), what it measured
// (each answer's prompt and generation speed, time to first token, the engine's memory, the browser's storage),
// and what to change, worked out from those measurements. The same report is T mode's `diag` (window.bankmlDiag).
// Reads window.bankmlStats and window.bankmlSettings (bankml-chat.js); everything stays in this browser.
// The disk and cache controls (browseml-fetch.js): what browseML keeps here and its size, persistent storage and the
// request to keep it, "remove the model after this session", and "clear browseML's cache" (confirmed in the page).
import * as browseml from "./browseml.js";
import * as fetcher from "./browseml-fetch.js";

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids.filter((k) => k !== null && k !== undefined)); return e; };
const cores = navigator.hardwareConcurrency || 1;
const isolated = !!(globalThis.crossOriginIsolated && typeof SharedArrayBuffer !== "undefined");
const MB = (b) => (b ? `${(b / 1e6).toFixed(0)} MB` : "—");
const fmt = (x, d = 1, unit = "") => (x === null || x === undefined || Number.isNaN(x) ? "—" : `${Number(x).toFixed(d)}${unit}`);
// Bonsai-1.7B's KV cache in f16: 28 layers × (8 heads × 128) × K and V × 2 bytes, per token of context
const KV_BYTES_PER_TOKEN = 28 * 1024 * 2 * 2;
const CTX = [1024, 2048, 4096];
// a token is about 3.6 characters of English for this tokenizer: an estimate, said as one
const tokensOf = (chars) => Math.round(chars / 3.6);

let storage = null, kept = [], persistedNow = null, confirmClear = false;
async function refreshStorage() {
  try { storage = navigator.storage && navigator.storage.estimate ? await navigator.storage.estimate() : null; } catch { storage = null; }
  try { storage = { ...(storage || {}), model: await browseml.cachedBytes() }; } catch {}
  try { kept = await fetcher.keptFiles(); } catch { kept = []; }
  persistedNow = await fetcher.persisted();
}
// the prompt of whoever speaks now (bankML or Savante), lean and full, as bankml-chat.js built it: an estimate
const promptSize = () => { const p = window.bankmlStats && window.bankmlStats.prompt; return p ? { who: p.speaker, lean: tokensOf(p.lean), full: tokensOf(p.full) } : { who: "the persona", lean: 0, full: 0 }; };
// the threads the engine can take here: every core up to browseML's cap when isolated, else one
const maxThreads = () => browseml.threads();

/** What the measurements say to change: each line is a finding and what to do about it. */
function advice(S, st) {
  const out = [], last = st.answers.filter((a) => a.mode === "browse").slice(-1)[0], persona = promptSize();
  const mode = st.mode || "browse";
  if (mode === "browse") {
    if (!isolated) out.push(["warn", "One thread: this page is not cross-origin isolated (or this browser has no SharedArrayBuffer). Open the Space directly — pythai-bankml.static.hf.space — in Chrome, Edge or Firefox to use every core."]);
    else if (st.browse && st.browse.threads < maxThreads()) out.push(["ok", `${st.browse.threads} of ${maxThreads()} threads in use${cores > maxThreads() ? ` (bankML in this browser takes at most ${browseml.MAX_THREADS} of the ${cores} cores)` : ""}. More threads answer faster; fewer leave the computer free for other work.`]);
    if (!st.browse) out.push(["warn", `browseML is not loaded: Load downloads ${MB(browseml.MODEL.bytes)} once, then it starts in seconds from this browser's cache.`]);
    if (last && last.prompt_tps) {
      const next = (S.lean ? persona.lean : persona.full) + 40;
      out.push(["ok", `Reading speed ${fmt(last.prompt_tps, 1)} tokens/s: a new conversation's first answer reads about ${next} tokens of ${persona.who}'s persona first — about ${fmt(next / last.prompt_tps, 0)} s. Later answers in the same window read only what is new (${last.cache_n ?? 0} tokens were reused last time).`]);
    }
    if (!S.lean && persona.full) out.push(["warn", `The full prompt adds ${persona.who}'s context summary: about ${persona.full - persona.lean} more tokens to read before every new conversation. “Lean” keeps only the persona and still adds the passages that match each question.`]);
    const more = isolated && st.browse && st.browse.threads < maxThreads();
    if (last && last.gen_tps && last.gen_tps < 1) out.push(["warn", `Writing at ${fmt(last.gen_tps, 2)} tokens/s: close other busy tabs, ${more ? "raise the threads, " : ""}or shorten answers (now ${S.maxTokens} tokens at most).`]);
    const kv = S.ctx * KV_BYTES_PER_TOKEN, dev = navigator.deviceMemory ? navigator.deviceMemory * 1e9 : null;
    if (dev && browseml.MODEL.bytes + kv > dev * 0.5) out.push(["warn", `Context ${S.ctx} reserves ${MB(kv)} for its cache, beside the ${MB(browseml.MODEL.bytes)} model, on a device that reports about ${navigator.deviceMemory} GB: a smaller context leaves more for the rest.`]);
    if (storage && storage.quota && storage.usage / storage.quota > 0.8) out.push(["warn", `This site's storage is ${fmt(100 * storage.usage / storage.quota, 0)} % full: the browser may evict the cached model.`]);
  }
  if (mode === "local" && !(st.local && st.local.verified && st.local.verified.guard === "play")) out.push(["warn", "Your own bankML is not connected: start it with ./install.sh start --space, then connect (who answers: your own bankML)."]);
  if (mode === "hf") out.push(["warn", "A provider answers: not bankML, no receipt, and free Hugging Face accounts have no included inference credit since 2026-10-07. browseML is free and verified."]);
  if (mode === "space") out.push(["warn", "The free CPU answers: Bonsai on Hugging Face's free CPU, llama.cpp, not bankML's engine, no receipt; what you type goes to a public Space. browseML (bankML in this browser) gives verified answers with a receipt."]);
  if (!out.length) out.push(["ok", "Nothing to change: the measurements look healthy."]);
  return out;
}

function sparkline(values) {
  const w = 220, h = 40, max = Math.max(1e-9, ...values);
  const pts = values.map((v, i) => `${values.length === 1 ? w / 2 : (i * w) / (values.length - 1)},${h - 2 - (v / max) * (h - 6)}`).join(" ");
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`); svg.setAttribute("class", "spark"); svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `generation speed of the last ${values.length} answers, at most ${max.toFixed(2)} tokens per second`);
  if (values.length) { const pl = document.createElementNS(svg.namespaceURI, "polyline"); pl.setAttribute("points", pts); svg.append(pl); }
  return svg;
}

function card(title, ...body) { return el("div", { className: "dcard" }, el("h3", { textContent: title }), ...body); }
function kv(rows) {
  const dl = el("dl", { className: "dkv" });
  for (const [k, v, cls] of rows) dl.append(el("dt", { textContent: k }), el("dd", { textContent: v, className: cls || "" }));
  return dl;
}

function render() {
  const root = $("dash"); if (!root || !window.bankmlSettings) return;
  // never under the visitor's hand: a slider being dragged or a menu open keeps its element
  if (root.contains(document.activeElement) && /^(INPUT|SELECT)$/.test(document.activeElement.tagName)) return;
  const S = window.bankmlSettings.get(), st = window.bankmlStats, set = window.bankmlSettings.set;
  const b = st.browse, mode = st.mode || "browse";
  const answers = st.answers, last = answers[answers.length - 1];
  // the threads a reload would give: capped as browseml.js caps them, so above 8 cores "reload to apply" does not stay
  const loaded = !!b, pendingThreads = Math.max(1, Math.min(maxThreads(), S.threads || cores)), stale = loaded && (b.threads !== pendingThreads || b.ctx !== S.ctx);

  // who answers, and the engine's state
  const engineRows = mode === "browse"
    ? (b ? [["engine", `bankML ${b.version} in this browser (WebAssembly)`], ["model", `${b.model} · verified`, "ok"], ["sha256", `${b.sha256.slice(0, 16)}…`],
            ["threads", `${b.threads} of ${cores} cores${isolated ? (cores > maxThreads() ? ` (at most ${browseml.MAX_THREADS} here)` : "") : " (not isolated: one)"}`],
            ["build", b.build === "browseml-mt-relaxed.wasm" ? "relaxed SIMD (fused multiply-add) — the fastest" : b.build === "browseml-mt.wasm" ? "SIMD, threads (no relaxed SIMD here)" : "SIMD, one thread"], ["context", `${b.ctx} tokens`], ["engine memory", MB(browseml.memoryBytes)], ["loaded in", fmt(b.load_ms / 1000, 0, " s")]]
         : [["engine", "browseML not loaded"], ["model", `${browseml.MODEL.title}, ${MB(browseml.MODEL.bytes)}`], ["in this browser's cache", storage && storage.model ? "yes" : "no"]])
    : mode === "local"
    ? (st.local ? [["engine", `your bankml serve at ${st.local.endpoint}`], ["model", st.local.verified?.name || "—", st.local.verified?.guard === "play" ? "ok" : "warn"], ["round trip", fmt(st.local.rtt_ms, 0, " ms")]] : [["engine", "your own bankML: not connected"]])
    : mode === "space"
    ? [["engine", "Bonsai on Hugging Face's free CPU — llama.cpp, not bankML's engine"], ["where", "the public Space Gregory-L/mindXhfgradio"], ["receipt", "none", "warn"]]
    : [["engine", "a Hugging Face provider — not bankML"], ["receipt", "none", "warn"]];
  const engineBtns = el("div", { className: "drow" });
  if (mode === "browse") {
    engineBtns.append(
      // never a download from here: a model not yet kept in this browser goes through browseML's consent card
      el("button", { type: "button", textContent: loaded ? (stale ? "Apply: reload the engine" : "Reload the engine") : storage && storage.model ? "Start bankML in this browser" : "Open browseML's card (download and verify)", className: stale || !loaded ? "primary" : "",
        onclick: async (e) => { e.target.disabled = true; try { if (loaded) await window.bankmlReloadBrowse(); else if (storage && storage.model) await window.bankmlLoadBrowse(); else window.bankmlBrowseCard && window.bankmlBrowseCard.show(); } finally { render(); } } }),
      ...(loaded ? [el("button", { type: "button", textContent: "Unload (free its memory)", onclick: () => { try { browseml.unload(); st.browse = null; st.emit(); } catch (err) { alertLine(err.message); } } })] : []));  // append() would write "null"
  }

  // the controls
  const threadOut = el("output", { textContent: `${pendingThreads}` });
  const threads = el("input", { type: "range", min: 1, max: maxThreads(), value: pendingThreads, disabled: !isolated,
    oninput: (e) => { threadOut.textContent = e.target.value; }, onchange: (e) => set({ threads: +e.target.value }) });
  const ctx = el("select", { onchange: (e) => set({ ctx: +e.target.value }) },
    ...CTX.map((c) => el("option", { value: c, selected: S.ctx === c, textContent: `${c} tokens · ${MB(c * KV_BYTES_PER_TOKEN)} of cache · about ${MB(browseml.MODEL.bytes + c * KV_BYTES_PER_TOKEN + 96e6)} in all (estimate)` })));
  const maxOut = el("output", { textContent: `${S.maxTokens}` });
  const maxT = el("input", { type: "range", min: 32, max: 512, step: 32, value: S.maxTokens,
    oninput: (e) => { maxOut.textContent = e.target.value; }, onchange: (e) => set({ maxTokens: +e.target.value }) });
  const persona = promptSize();
  const prompt = el("div", { className: "drow", role: "radiogroup" },
    el("label", {}, el("input", { type: "radio", name: "dprompt", checked: S.lean, onchange: () => set({ lean: true }) }), ` lean — ${persona.who}'s persona (~${persona.lean} tokens)`),
    el("label", {}, el("input", { type: "radio", name: "dprompt", checked: !S.lean, onchange: () => set({ lean: false }) }), ` full — and its context summary (~${persona.full})`));
  const controls = card("Controls",
    el("label", { className: "dctl" }, el("span", { textContent: "threads" }), threads, threadOut,
      el("small", { textContent: isolated ? `of ${cores} cores${cores > maxThreads() ? ` (at most ${browseml.MAX_THREADS} here)` : ""} · applies on reload` : "one: this page is not cross-origin isolated" })),
    el("label", { className: "dctl" }, el("span", { textContent: "context" }), ctx, el("small", { textContent: "how much of a conversation it holds · applies on reload" })),
    el("label", { className: "dctl" }, el("span", { textContent: "answer length" }), maxT, maxOut, el("small", { textContent: "tokens at most, from the next answer" })),
    el("div", { className: "dctl" }, el("span", { textContent: "prompt" }), prompt),
    stale ? el("p", { className: "dnote warn", textContent: `The engine runs ${b.threads} thread${b.threads > 1 ? "s" : ""} and context ${b.ctx}: reload it to apply.` }) : null);

  // usage, measured
  const gen = answers.filter((a) => a.gen_tps).map((a) => a.gen_tps).slice(-30);
  const totals = answers.reduce((t, a) => ({ n: t.n + 1, tok: t.tok + (a.predicted_n || 0), ms: t.ms + (a.total_ms || 0) }), { n: 0, tok: 0, ms: 0 });
  const usage = card("Usage, measured",
    kv(last ? [["last answer", `${last.speaker} via ${{ space: "the free CPU (not bankML)", browse: "browseML", local: "your bankML", hf: "a provider" }[last.mode]} · ${last.ok ? "✓ receipt" : last.mode === "hf" || last.mode === "space" ? "no receipt" : "✗ receipt"}`, last.ok ? "ok" : "warn"],
               ["first token", fmt(last.ttft_ms / 1000, 1, " s")], ["reading", `${fmt(last.prompt_tps, 1, " tok/s")} · ${last.prompt_n ?? "—"} tokens (${last.cache_n ?? 0} reused)`],
               ["writing", `${fmt(last.gen_tps, 2, " tok/s")} · ${last.predicted_n ?? "—"} tokens`], ["whole answer", fmt(last.total_ms / 1000, 1, " s")]]
             : [["last answer", "none yet — ask in the field below"]]),
    el("div", { className: "dspark" }, sparkline(gen), el("small", { textContent: gen.length ? `writing speed, last ${gen.length} answers` : "writing speed appears after an answer" })),
    kv([["this visit", `${totals.n} answer${totals.n === 1 ? "" : "s"} · ${totals.tok} tokens written · ${fmt(totals.ms / 1000, 0, " s")} computing`]]));

  // memory and storage
  const heap = performance.memory ? `${MB(performance.memory.usedJSHeapSize)} of ${MB(performance.memory.jsHeapSizeLimit)}` : "not reported by this browser";
  const mem = card("Memory",
    kv([["engine (WebAssembly)", loaded ? MB(browseml.memoryBytes) : "—"], ["page (JavaScript heap)", heap],
        ["this device", navigator.deviceMemory ? `about ${navigator.deviceMemory} GB (rounded by the browser)` : "not reported"],
        ["context cache", `${MB(S.ctx * KV_BYTES_PER_TOKEN)} at ${S.ctx} tokens (estimate)`],
        ["browseML needs", `about ${MB(browseml.MODEL.bytes + S.ctx * KV_BYTES_PER_TOKEN + 96e6)}: the model, the context cache and the engine's scratch (estimate)`]]),
    navigator.deviceMemory && navigator.deviceMemory <= 4 ? el("p", { className: "dnote warn", textContent: `This device reports about ${navigator.deviceMemory} GB: the browser may close the tab under memory pressure. A smaller context helps, or use the free CPU tab.` }) : null);
  // disk and cache: what browseML keeps here, persistent storage, remove after this session, clear (confirmed here)
  const cacheBytes = kept.filter((f) => f.where === "Cache Storage").reduce((n, f) => n + f.bytes, 0);
  const files = kept.length
    ? el("ul", { className: "dadvice" }, ...kept.map((f) => el("li", { className: "ok", textContent: `${f.name} · ${MB(f.bytes)} · ${f.where}${f.where === "HTTP cache" ? " (the browser manages it)" : ""}` })))
    : el("p", { className: "dnote", textContent: "browseML keeps nothing in this browser yet." });
  const after = el("label", { className: "drow" }, el("input", { type: "checkbox", checked: !!S.removeAfterSession, onchange: (e) => set({ removeAfterSession: e.target.checked }) }),
    " remove the model after this session (the next visit starts without it)");
  const clearRow = el("div", { className: "drow" });
  if (!confirmClear) clearRow.append(el("button", { type: "button", textContent: `Clear browseML's cache${cacheBytes ? ` (frees ${MB(cacheBytes)})` : ""}`, disabled: !cacheBytes, onclick: () => { confirmClear = true; render(); } }));
  else clearRow.append(el("span", { className: "dnote warn", textContent: `Remove ${MB(cacheBytes)} from this browser? The next browseML answer downloads the model again.` }),
    el("button", { type: "button", className: "primary", textContent: "Yes, clear it", onclick: async () => { const freed = await fetcher.clearCache(); confirmClear = false; await refreshStorage(); alertLine(`cleared: ${MB(freed)} freed`); st.emit(); window.bankmlBrowseCard && window.bankmlBrowseCard.refresh(); } }),
    el("button", { type: "button", textContent: "Keep it", onclick: () => { confirmClear = false; render(); } }));
  const disk = card("Disk and cache",
    kv([["this site's storage", storage && storage.quota ? `${MB(storage.usage)} used of ${(storage.quota / 1e9).toFixed(1)} GB allowed` : "not reported"],
        ["browseML's cache", cacheBytes ? MB(cacheBytes) : "empty"],
        ["kept under pressure", persistedNow === null ? "this browser does not say" : persistedNow ? "yes — persistent storage granted" : "no — the browser may evict it when space runs low", persistedNow ? "ok" : "warn"]]),
    files,
    el("div", { className: "drow" }, el("button", { type: "button", textContent: "Keep it (ask for persistent storage)", disabled: persistedNow !== false,
      onclick: async () => { const p = await fetcher.askPersist(); await refreshStorage(); alertLine(p ? "persistent storage granted" : "the browser did not grant persistent storage (it decides; a bookmarked or installed site is more likely to get it)"); render(); } })),
    after, clearRow);

  const adv = card("What to change", el("ul", { className: "dadvice" }, ...advice(S, st).map(([lvl, t]) => el("li", { className: lvl, textContent: t }))));
  const who = card("Who answers", kv(engineRows), engineBtns);

  root.replaceChildren(el("h2", { textContent: "This computer — controls and diagnostics" }),
    el("p", { className: "dlede", textContent: `Everything here is measured in this browser and stays in it. ${cores} logical cores · ${isolated ? "cross-origin isolated: threads on" : "not isolated: one thread"}.` }),
    el("p", { className: "dnow", id: "dash-now", textContent: st.activity ? "now: " + st.activity : "now: idle" }),
    el("div", { className: "dgrid" }, who, controls, usage, mem, disk, adv));
}
let alertTimer = null;
function alertLine(t) { const p = $("dash")?.querySelector(".dlede"); if (!p) return; const old = p.textContent; p.textContent = t; clearTimeout(alertTimer); alertTimer = setTimeout(() => (p.textContent = old), 4000); }

/** T mode's diag: the same measurements, as an accordion. */
window.bankmlDiag = async () => {
  await refreshStorage();
  const S = window.bankmlSettings.get(), st = window.bankmlStats, b = st.browse, last = st.answers.slice(-1)[0];
  const sec = (title, level, lines) => ({ title, level, lines });
  const adv = advice(S, st);
  return {
    title: `diag · this computer · ${new Date().toLocaleTimeString()} · ${adv.some(([l]) => l === "warn") ? "warn" : "ok"}`,
    sections: [
      sec("processor", isolated ? "ok" : "warn", [`${cores} logical cores`, isolated ? "cross-origin isolated: SharedArrayBuffer, threads available" : "not cross-origin isolated: one thread", `user agent: ${navigator.userAgent}`]),
      sec("engine", b ? "ok" : "warn", b ? [`${b.model} verified · sha256 ${b.sha256}`, `bankML ${b.version} · ${b.threads} thread${b.threads > 1 ? "s" : ""} · context ${b.ctx}`, `engine memory ${MB(browseml.memoryBytes)}`] : ["browseML not loaded"]),
      sec("memory", "ok", [`device: ${navigator.deviceMemory ? `about ${navigator.deviceMemory} GB` : "not reported"}`, `page heap: ${performance.memory ? MB(performance.memory.usedJSHeapSize) : "not reported"}`, `context cache: ${MB(S.ctx * KV_BYTES_PER_TOKEN)}`]),
      sec("storage", storage && storage.quota && storage.usage / storage.quota > 0.8 ? "warn" : "ok", [storage && storage.quota ? `${MB(storage.usage)} of ${(storage.quota / 1e9).toFixed(1)} GB` : "not reported", `model cached: ${storage && storage.model ? MB(storage.model) : "no"}`, `persistent storage: ${persistedNow === null ? "not reported" : persistedNow ? "granted" : "not granted"}`, ...kept.map((f) => `${f.name}: ${MB(f.bytes)} (${f.where})`), `remove after this session: ${S.removeAfterSession ? "on" : "off"}`]),
      sec("speed", last ? "ok" : "warn", last ? [`first token ${fmt(last.ttft_ms / 1000, 1, " s")}`, `reading ${fmt(last.prompt_tps, 1, " tok/s")} (${last.prompt_n ?? "—"} tokens, ${last.cache_n ?? 0} reused)`, `writing ${fmt(last.gen_tps, 2, " tok/s")} (${last.predicted_n ?? "—"} tokens)`] : ["no answer yet"]),
      sec("what to change", adv.some(([l]) => l === "warn") ? "warn" : "ok", adv.map(([, t]) => t)),
    ],
  };
};

(async () => {
  await refreshStorage();
  render();
  window.addEventListener("bankml:stats", () => { refreshStorage().then(render); });
  // the live line, without redrawing the cards: it changes every second while something runs
  window.addEventListener("bankml:activity", () => { const n = $("dash-now"); if (n) n.textContent = window.bankmlStats.activity ? "now: " + window.bankmlStats.activity : "now: idle"; });
  setInterval(() => { if (!document.hidden) render(); }, 5000);  // the engine's memory and the page's heap move on their own
})();
