// SPDX-License-Identifier: MIT OR Apache-2.0
// bankML — the page over sAGI/console.py: Ask (the landing), Admin, Receipts, Logs and Diagnostics. Every value is rendered with textContent;
// nothing bankML did not measure is drawn or filled in ("not measured"); every receipt is checked in the browser.
"use strict";
const $ = (id) => document.getElementById(id);
// a popped-out output window of the input field shows only that output
if (window.BankmlUIF && BankmlUIF.isOutputWindow()) {
  document.body.textContent = "";
  const w = document.createElement("div");
  document.body.append(w);
  BankmlUIF.mount(w);
  throw new Error("bankML: an output window, nothing else to run here");
}
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const fmt = (v, d = 1, unit = "") => (v === null || v === undefined || Number.isNaN(+v)) ? "not measured" : `${(+v).toFixed(d)}${unit}`;
const GB = 1e9, MB = 1e6, KEEP = 150;
const short = (h, n = 12) => (h ? String(h).slice(0, n) + "…" : "—");
const clock = (s) => new Date(s * 1000).toTimeString().slice(0, 5);
let history = [], publicMode = false, slidersSet = false;
const series = { t: [], cpu: [], rss: [], avail: [], gbusy: [], watts: [] };

async function sha256hex(text) {
  const h = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// ── tabs ────────────────────────────────────────────────────────────────────────────────────────────────────────
function show(tab) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === tab)));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === tab));
  if (tab === "receipts") loadReceipts();
  if (tab === "logs") loadLogs();
  if (tab === "diagnostics") loadDiagnostics();
  if (tab === "engine") loadEngine();
  if (tab === "thesis") loadThesis();
  if (tab === "ask") $("q").focus();
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));

// ── Ask ─────────────────────────────────────────────────────────────────────────────────────────────────────────
const grow = () => { const q = $("q"); q.style.height = "auto"; q.style.height = Math.min(q.scrollHeight, innerHeight * 0.4) + "px"; };
$("q").addEventListener("input", grow);
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("form").requestSubmit(); } });

$("form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const q = $("q").value.trim();
  if (!q) return;
  $("empty")?.remove();
  $("q").value = ""; grow(); $("send").disabled = true;
  $("chat").append(el("div", "turn-q", q));
  const a = el("div", "turn-a waiting", "…"); const rc = el("div", "receipt");
  const turn = el("div"); turn.append(a, rc); $("chat").append(turn);
  const t0 = Date.now();
  const tick = setInterval(() => { if (a.classList.contains("waiting")) a.textContent = `… reading the prompt · ${Math.round((Date.now() - t0) / 1000)} s`; }, 1000);
  turn.scrollIntoView({ block: "end" });
  let text = "";
  try {
    const r = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: q, history }) });
    if (!r.ok) throw new Error(await r.text());
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 1);
        if (!line.trim()) continue;
        const d = JSON.parse(line);
        if (d.piece) { a.classList.remove("waiting"); text += d.piece; a.textContent = text; }
        if (d.done) {
          a.classList.remove("waiting");
          if (d.error) { a.textContent = text || "—"; rc.append(el("span", "bad", "refused: " + d.error)); }
          else if (d.receipt) {
            const ok = (await sha256hex(text)) === d.receipt.response_sha256;
            const m = d.metrics || {};
            rc.append(el("span", ok ? "ok" : "bad", ok ? "✓ receipt" : "≠ answer and receipt differ"),
              el("span", null, `model ${short(d.receipt.model_sha256)}`), el("span", null, `answer ${short(d.receipt.response_sha256)}`),
              el("span", null, `${d.receipt.prompt_tokens} + ${d.receipt.completion_tokens} tokens`),
              el("span", null, `first token ${fmt(m.ttft_ms, 0, " ms")}`), el("span", null, `${fmt(m.eval_tps, 1, " tok/s")}`));
          }
          history.push({ role: "user", content: q }, { role: "assistant", content: text });
          history = history.slice(-12);
        }
      }
    }
  } catch (e) {
    a.classList.remove("waiting"); a.textContent = text || "—";
    rc.append(el("span", "bad", "no answer: " + String(e.message || e).slice(0, 200)));
  }
  clearInterval(tick);
  $("send").disabled = false; $("q").focus();
});

// ── Admin ───────────────────────────────────────────────────────────────────────────────────────────────────────
const sync = () => { $("threads-v").textContent = $("threads").value; $("ram-v").textContent = (+$("ram").value).toFixed(1) + " GB";
  $("gpu-v").textContent = +$("gpu").value === 0 ? "off" : $("gpu").value + " %"; };
["threads", "ram", "gpu"].forEach((id) => $(id).addEventListener("input", sync));
$("apply").addEventListener("click", async () => {
  $("apply").disabled = true;
  const r = await fetch("/api/resources", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ threads: +$("threads").value, ram_gb: +$("ram").value, gpu_limit: +$("gpu").value / 100 }) });
  const j = r.headers.get("Content-Type")?.includes("json") ? await r.json() : { ok: false, error: await r.text() };
  $("job").textContent = j.ok ? "restarting…" : "refused: " + j.error; $("apply").disabled = publicMode;
});

function chart(id, lines, opts = {}) {
  const svg = d3.select("#" + id), node = svg.node(); const W = node.clientWidth || 300, H = 130, m = { t: 6, r: 6, b: 18, l: 34 };
  svg.attr("viewBox", `0 0 ${W} ${H}`); svg.selectAll("*").remove();
  const all = lines.flatMap((l) => l.data.filter((p) => p.y !== null && p.y !== undefined));
  if (!all.length) { svg.append("text").attr("class", "none").attr("x", W / 2).attr("y", H / 2).attr("text-anchor", "middle").text(opts.empty || "not measured yet"); return; }
  const x = d3.scaleLinear().domain(d3.extent(lines.flatMap((l) => l.data.map((p) => p.x)))).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, (d3.max([d3.max(all, (p) => p.y), opts.limit || 0]) || 1) * 1.1]).nice().range([H - m.b, m.t]);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(3).tickFormat((s) => clock(s)).tickSizeOuter(0));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(3).tickSizeOuter(0));
  if (opts.limit) svg.append("line").attr("class", "limit").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(opts.limit)).attr("y2", y(opts.limit));
  const ln = d3.line().defined((p) => p.y !== null && p.y !== undefined).x((p) => x(p.x)).y((p) => y(p.y));
  lines.forEach((l) => svg.append("path").attr("class", l.cls).attr("d", ln(l.data)));
}
const fact = (dl, k, v) => dl.append(el("dt", null, k), el("dd", null, v));

async function poll() {
  let s;
  try { s = await (await fetch("/api/state")).json(); } catch (e) { $("engine").textContent = "the console is not answering: " + e; return; }
  publicMode = !!s.public;
  $("mantra").textContent = s.persona.mantra;
  // the switch: Savante beside this console on this machine, or Savante's public Space from a hosted console
  $("to-savante").href = publicMode ? "https://huggingface.co/spaces/PYTHAI/savante" : `${location.protocol}//${location.hostname}:7873/`;
  const v = (s.serve || {}).verified || {};
  $("engine").textContent = "";
  $("engine").append(v.verdict === "play" ? el("span", "ok", "✓ verified ") : el("span", "bad", "no verified engine yet "),
    document.createTextNode(v.verdict === "play" ? `${v.name || "model"} · sha256 ${short(v.model_sha256, 16)} · bankML ${v.bankml}` : "— is bankml serve running?"));
  const u = s.usage || {}, m = s.metrics || {}, lim = u.gpu_limiter || null, g = (u.gpus || [])[0] || {};
  if (!slidersSet) {
    const r = s.resources || {};
    $("threads").max = u.cores || 4; $("threads").value = r.threads || 1;
    $("ram").max = Math.max(1, (u.mem_total_bytes || 6 * GB) / GB).toFixed(1); $("ram").value = r.ram_gb || 2;
    $("gpu").value = Math.round(100 * (r.gpu_limit ?? 0.8)); sync(); slidersSet = true;
    if (publicMode) { ["threads", "ram", "gpu", "apply"].forEach((id) => { $(id).disabled = true; }); $("apply").textContent = "read-only: a public console"; }
  }
  $("job").textContent = s.job.busy ? s.job.what : (s.job.error ? "failed: " + s.job.error : (s.job.done ? "done — the engine restarted, verified" : ""));
  const now = $("now"); now.textContent = "";
  const last = (m.records || []).slice(-1)[0] || {};
  fact(now, "tokens in · out", `${m.prompt_tokens ?? 0} · ${m.completion_tokens ?? 0}`);
  fact(now, "first token, last answer", fmt(last.ttft_ms, 0, " ms"));
  fact(now, "prompt · generation", `${fmt(last.prompt_tps, 1)} · ${fmt(last.eval_tps, 1)} tok/s`);
  fact(now, "CPU", fmt(u.cpu_percent, 0, " %"));
  fact(now, "memory held", fmt(u.rss_bytes / GB, 2, " GB"));
  fact(now, "memory available", fmt(u.mem_available_bytes / GB, 2, " GB"));
  fact(now, "GPU", lim ? `${(100 * lim.limit).toFixed(0)} % limit · ${(lim.allocated_bytes / MB).toFixed(0)} MB` : "no card in use");
  fact(now, "power · per token", `${fmt(u.package_watts, 1, " W")} · ${fmt(m.joules_per_token, 3, " J")}`);
  if (!$("admin").classList.contains("active")) return;
  const t = Date.now() / 1000, push = (k, val) => { series[k].push(val); if (series[k].length > KEEP) series[k].shift(); };
  push("t", t); push("cpu", u.cpu_percent ?? null); push("rss", u.rss_bytes != null ? u.rss_bytes / GB : null);
  push("avail", u.mem_available_bytes != null ? u.mem_available_bytes / GB : null); push("gbusy", g.busy_percent ?? null); push("watts", u.package_watts ?? null);
  const pts = (k) => series.t.map((x, i) => ({ x, y: series[k][i] }));
  const recs = (m.records || []).slice(-60);
  chart("c-tps", [{ cls: "l1", data: recs.map((r) => ({ x: r.at, y: r.eval_tps })) }, { cls: "l2", data: recs.map((r) => ({ x: r.at, y: r.prompt_tps })) }], { empty: "no answers yet" });
  chart("c-ttft", [{ cls: "l1", data: recs.map((r) => ({ x: r.at, y: r.ttft_ms })) }], { empty: "no answers yet" });
  chart("c-cpu", [{ cls: "l1", data: pts("cpu") }]);
  chart("c-mem", [{ cls: "l1", data: pts("rss") }, { cls: "l2", data: pts("avail") }]);
  chart("c-gpu", [{ cls: "l1", data: pts("gbusy") }], { limit: lim ? 100 * lim.limit : null, empty: "no GPU reading" });
  chart("c-pow", [{ cls: "l1", data: pts("watts") }], { empty: "power not measured (./install.sh power)" });
}

// ── Receipts ────────────────────────────────────────────────────────────────────────────────────────────────────
let info = null;
async function loadReceipts() {
  const j = await (await fetch("/api/log")).json();
  const list = $("receipt-list"); list.textContent = "";
  const rows = j.exchanges.slice().reverse();
  let verified = 0;
  for (const x of rows) {
    const r = x.receipt || null;
    const ok = r ? (await sha256hex(x.answer || "")) === r.response_sha256 : false;
    verified += ok;
    const li = el("li"), d = el("details"), s = el("summary");
    s.append(el("span", "t", clock(x.at)), el("span", "qq", x.question),
      el("span", "v " + (x.error ? "bad" : ok ? "ok" : "bad"), x.error ? "refused" : ok ? "✓ " + short(r.response_sha256, 8) : "≠ check"));
    const body = el("div", "body");
    body.append(el("p", null, x.error ? "refused: " + x.error : x.answer));
    body.append(el("pre", null, JSON.stringify(r || { error: x.error }, null, 1)));
    d.append(s, body); li.append(d); list.append(li);
  }
  if (!rows.length) list.append(el("li", "none-yet", publicMode ? "A public console keeps no record of its visitors' questions." : "No answers yet — ask on the Ask tab."));
  $("logsum").textContent = rows.length ? `${rows.length} answers · ${verified} receipts check out in this browser` : "";
  info = await (await fetch("/api/infotags")).json();
  const dl = $("attrs"); dl.textContent = "";
  info.attributes.forEach((a) => fact(dl, a.trait_type, String(a.value)));
}
// ── Logs ────────────────────────────────────────────────────────────────────────────────────────────────────────
async function loadLogs() {
  const j = await (await fetch("/api/log")).json();
  $("englog").textContent = (j.engine_log || []).join("\n") || "the engine has written nothing yet";
}
// ── Thesis: docs/TECHNICAL.md's thesis section, as DOM nodes (never innerHTML) ──────────────────────────────────
// Inline: **bold**, *italic*, `code`, [text](url). Blocks: ### headings, numbered or dashed lists, paragraphs.
function inline(parent, text, base) {
  const re = /(\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`|\[([^\]]+)\]\(([^)\s]+)\))/g;
  let at = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > at) parent.append(text.slice(at, m.index));
    if (m[2] !== undefined) { const b = el("strong"); inline(b, m[2], base); parent.append(b); }
    else if (m[3] !== undefined) { const i = el("em"); inline(i, m[3], base); parent.append(i); }
    else if (m[4] !== undefined) parent.append(el("code", null, m[4]));
    else {
      const href = /^(https?:)?\/\//.test(m[6]) ? m[6] : m[6].startsWith("#") ? base + "TECHNICAL.md" + m[6] : base + m[6];
      const a = el("a"); a.href = href; a.target = "_blank"; a.rel = "noreferrer"; inline(a, m[5], base); parent.append(a);
    }
    at = re.lastIndex;
  }
  if (at < text.length) parent.append(text.slice(at));
}
function markdown(root, md, base) {
  root.textContent = "";
  const blocks = md.split(/\n\s*\n/);
  for (const raw of blocks) {
    const lines = raw.split("\n");
    const first = lines[0];
    let h;
    if ((h = /^(#{2,4})\s+(.*)$/.exec(first))) { const e = el(h[1].length === 2 ? "h2" : "h3", "prose-h"); inline(e, h[2], base); root.append(e); continue; }
    if (/^(\d+\.|-)\s/.test(first)) {
      const list = el(/^\d/.test(first) ? "ol" : "ul");
      let cur = null;
      for (const l of lines) {
        const item = /^(\d+\.|-)\s+(.*)$/.exec(l);
        if (item) { cur = el("li"); list.append(cur); inline(cur, item[2], base); }
        else if (cur) { cur.append(" "); inline(cur, l.trim(), base); }
      }
      root.append(list);
      continue;
    }
    const p = el("p");
    inline(p, lines.map((l) => l.trim()).join(" "), base);
    root.append(p);
  }
}
let thesisLoaded = false;
async function loadThesis() {
  if (thesisLoaded) return;
  const j = await (await fetch("/api/thesis")).json();
  if (j.error) { $("thesis-body").textContent = j.error; return; }
  markdown($("thesis-body"), j.markdown, j.base);
  $("thesis-src").href = j.url;
  thesisLoaded = true;
}
// ── Engine: bankml serve's status, every 3 s while the tab is open ─────────────────────────────────────────────────
async function loadEngine() {
  const root = $("engine-status");
  try {
    const r = await fetch("/api/engine");
    const j = await r.json();
    if (!r.ok) { root.textContent = j.error || "bankml serve did not answer"; return; }
    BankmlStatus.render(root, j);
  } catch (e) { root.textContent = "bankml serve did not answer: " + e; }
}
setInterval(() => { if ($("engine").classList.contains("active")) loadEngine(); }, 3000);
// ── Diagnostics ─────────────────────────────────────────────────────────────────────────────────────────────────
function spanNode(n, total) {
  const li = el("li", "span" + (n.error ? " bad" : "") + (n.open ? " open" : ""));
  const row = el("div", "row");
  const ms = n.open ? "open" : n.duration_ms === null ? "—" : `${n.duration_ms} ms`;
  row.append(el("span", "name", n.name), el("span", "ms", ms));
  const track = el("span", "track"), bar = el("span", "fill");
  if (n.duration_ms && total) bar.style.width = Math.max(0.5, Math.min(100, (100 * n.duration_ms) / total)) + "%";
  track.append(bar);
  row.append(track);
  li.append(row);
  const tags = Object.entries(n.tags || {}).map(([k, v]) => `${k} ${v}`).join(" · ");
  const evs = (n.events || []).map((e) => `${e.name} @ ${e.at_ms} ms`).join(" · ");
  if (tags || evs || n.error) li.append(el("div", "meta", [n.error && "✗ " + n.error, tags, evs].filter(Boolean).join("  ·  ")));
  if (n.children && n.children.length) {
    const ul = el("ul");
    n.children.forEach((c) => ul.append(spanNode(c, total)));
    li.append(ul);
  }
  return li;
}
function componentCard(c) {
  const card = el("article", "comp " + c.level);
  const head = el("header");
  head.append(el("span", "chip " + c.level, c.level === "ok" ? "ready" : c.level === "info" ? "idle" : c.level === "warn" ? "check" : "failing"),
    el("h3", null, c.component), el("span", "ms", c.ms + " ms"));
  card.append(head, el("p", "role", c.role), el("code", "file", c.file));
  if (c.seen) card.append(el("p", "seen", c.seen));
  return card;
}
async function loadDiagnostics() {
  const j = await (await fetch("/api/diagnostics")).json();
  const comps = j.components || [], grid = $("components");
  grid.textContent = "";
  comps.forEach((c) => grid.append(componentCard(c)));
  const n = (l) => comps.filter((c) => c.level === l).length;
  $("diag-sum").textContent = comps.length ? `${n("ok")} ready · ${n("info")} idle · ${n("warn")} to check · ${n("bad")} failing` : "";
  const ul = $("checks"); ul.textContent = "";
  for (const c of j.checks) {
    const li = el("li", c.level);
    li.append(el("span", "mark", c.level === "ok" ? "✓" : c.level === "warn" ? "!" : "✗"), el("span", "name", c.check), el("span", "seen", c.seen));
    ul.append(li);
  }
  const tr = $("traces"); tr.textContent = "";
  if (!j.traces.length) tr.append(el("p", "none-yet", "No trace yet — ask on the Ask tab; each answer is traced span by span."));
  for (const t of j.traces) {
    const box = el("div", "trace");
    box.append(el("div", "when", new Date(t.start * 1000).toLocaleTimeString()));
    const ul2 = el("ul", "tree");
    ul2.append(spanNode(t, t.duration_ms));
    box.append(ul2);
    tr.append(box);
  }
  $("trace-text").textContent = j.text || "";
  const sc = $("scientific"), x = j.scientific;
  if (x) {
    sc.textContent = "";
    const s = x.summary || {};
    const dl = el("dl", "facts");
    const add = (k, v) => dl.append(el("dt", null, k), el("dd", null, v));
    add("verdict", `${x.verdict} — ${s.bits_equal} of ${s.tokens} tokens bit-equal (top-5 included)`);
    add("model", `${(x.model || {}).name} · ${String((x.model || {}).sha256 || "").slice(0, 16)}… · bankML ${x.bankml} · ${x.threads} threads`);
    add("max |Δ| logprob", s.max_abs_delta);
    add("log-likelihood", `${(s.log_likelihood || {}).bankml} (bankML) · ${(s.log_likelihood || {}).reference} (llama.cpp)`);
    add("perplexity", `${(s.perplexity || {}).bankml} · ${(s.perplexity || {}).reference}`);
    const t = x.timing || {};
    add("generation", `${(t.bankml || {}).predicted_tokens_per_s} tok/s (bankML) · ${(t.reference || {}).predicted_tokens_per_s} (llama.cpp)`);
    add("resolution", `values ${(x.precision || {}).unit}; time ${(x.precision || {}).time_resolution_s} s (${(x.precision || {}).time_note})`);
    add("measured", x.at);
    sc.append(dl);
  }
  $("diag-source").textContent = j.source || "";
}
$("diag-refresh").addEventListener("click", loadDiagnostics);
$("download").addEventListener("click", () => {
  if (!info) return;
  const a = el("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(info, null, 1)], { type: "application/json" }));
  a.download = "bankml-infotags.json"; a.click(); URL.revokeObjectURL(a.href);
});

poll(); setInterval(poll, 2000);
// the Ask landing: the ultimate input field when it is built (uif/), the plain box otherwise
if (window.BankmlUIF) {
  $("form").hidden = true;
  $("ask").classList.add("uif");
  BankmlUIF.mount($("uif-landing"), { placeholder: "Ask bankML — T for terminal mode" });
} else {
  $("q").focus();
}
