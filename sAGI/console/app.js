// SPDX-License-Identifier: MIT OR Apache-2.0
// bankML console: four tabs over sAGI/console.py. Every value is rendered with textContent; the charts draw only what
// bankml measured (a null is drawn as a gap and labelled "not measured", never filled in).
"use strict";
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const fmt = (v, d = 1, unit = "") => (v === null || v === undefined || Number.isNaN(v)) ? "not measured" : `${(+v).toFixed(d)}${unit}`;
const GB = 1e9, MB = 1e6;
let history = [];
const series = { t: [], cpu: [], rss: [], avail: [], gbusy: [], galloc: [], watts: [] };
const KEEP = 150;

// ── tabs ────────────────────────────────────────────────────────────────────────────────────────────────────────
document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("nav button").forEach((x) => x.setAttribute("aria-selected", x === b ? "true" : "false"));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === b.dataset.tab));
  if (b.dataset.tab === "logging") loadLog();
  if (b.dataset.tab === "infotags") loadInfo();
}));

// ── interaction ─────────────────────────────────────────────────────────────────────────────────────────────────
$("ask").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const q = $("q").value.trim();
  if (!q) return;
  $("q").value = ""; $("send").disabled = true;
  $("chat").append(el("div", "msg user", q));
  const bot = el("div", "msg bot", ""); const body = el("div"); const rc = el("div", "receipt", "…");
  bot.append(body, rc); $("chat").append(bot); bot.scrollIntoView({ block: "end" });
  try {
    const r = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: q, history }) });
    const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = "", text = "";
    for (;;) {
      const { done, value } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 1);
        if (!line.trim()) continue;
        const d = JSON.parse(line);
        if (d.piece) { text += d.piece; body.textContent = text; bot.scrollIntoView({ block: "end" }); }
        if (d.done) {
          rc.textContent = "";
          if (d.error) { rc.append(el("span", "bad", "refused: " + d.error)); }
          else if (d.receipt) {
            const m = d.metrics || {};
            rc.append(el("span", d.answer_sha256_ok ? "ok" : "bad", d.answer_sha256_ok ? "✓ answer = receipt" : "≠ received!"),
              document.createTextNode(`  model ${String(d.receipt.model_sha256).slice(0, 12)}…  ` +
                `${d.receipt.prompt_tokens}+${d.receipt.completion_tokens} tok · TTFT ${fmt(m.ttft_ms, 0, " ms")} · tg ${fmt(m.eval_tps, 1, " tok/s")} · ${fmt(m.joules_per_token, 3, " J/tok")}`));
          }
          history.push({ role: "user", content: q }, { role: "assistant", content: text });
          history = history.slice(-12);
        }
      }
    }
  } catch (e) { rc.textContent = "error: " + e; }
  $("send").disabled = false; $("q").focus();
});
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("ask").requestSubmit(); } });

// ── admin: sliders ──────────────────────────────────────────────────────────────────────────────────────────────
const sync = () => { $("threads-v").textContent = $("threads").value; $("ram-v").textContent = (+$("ram").value).toFixed(1) + " GB";
  $("gpu-v").textContent = +$("gpu").value === 0 ? "off" : $("gpu").value + " %"; };
["threads", "ram", "gpu"].forEach((id) => $(id).addEventListener("input", sync));
let slidersSet = false;
$("apply").addEventListener("click", async () => {
  $("apply").disabled = true;
  const r = await fetch("/api/resources", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ threads: +$("threads").value, ram_gb: +$("ram").value, gpu_limit: +$("gpu").value / 100 }) });
  const j = await r.json(); $("job").textContent = j.ok ? "restarting…" : "refused: " + j.error; $("apply").disabled = false;
});

// ── admin: charts (D3) ──────────────────────────────────────────────────────────────────────────────────────────
function chart(id, lines, opts = {}) {
  const svg = d3.select("#" + id); const node = svg.node(); const W = node.clientWidth || 340, H = 150, m = { t: 8, r: 10, b: 20, l: 38 };
  svg.attr("viewBox", `0 0 ${W} ${H}`); svg.selectAll("*").remove();
  const all = lines.flatMap((l) => l.data.filter((p) => p.y !== null && p.y !== undefined));
  if (!all.length) { svg.append("text").attr("class", "none").attr("x", W / 2).attr("y", H / 2).attr("text-anchor", "middle").text(opts.empty || "not measured yet"); return; }
  const xs = lines.flatMap((l) => l.data.map((p) => p.x));
  const x = d3.scaleLinear().domain(d3.extent(xs)).range([m.l, W - m.r]);
  const ymax = d3.max([d3.max(all, (p) => p.y), opts.limit || 0]) || 1;
  const y = d3.scaleLinear().domain([0, ymax * 1.1]).nice().range([H - m.b, m.t]);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(4).tickFormat(opts.xfmt || d3.format("d")));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4));
  if (opts.limit) svg.append("line").attr("class", "limit").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(opts.limit)).attr("y2", y(opts.limit));
  const ln = d3.line().defined((p) => p.y !== null && p.y !== undefined).x((p) => x(p.x)).y((p) => y(p.y));
  lines.forEach((l) => svg.append("path").attr("class", l.cls).attr("d", ln(l.data)));
}
const tfmt = (s) => { const d = new Date(s * 1000); return d.toTimeString().slice(0, 8); };

function nowRow(dl, k, v) { dl.append(el("dt", null, k), el("dd", null, v)); }

async function poll() {
  let s;
  try { s = await (await fetch("/api/state")).json(); } catch (e) { $("status").textContent = "console: " + e; return; }
  const v = (s.serve || {}).verified || {}; const u = s.usage || {}; const m = s.metrics || {}; const lim = u.gpu_limiter || null;
  $("mantra").textContent = s.persona.mantra;
  $("status").textContent = "";
  $("status").append(v.verdict === "play" ? el("span", "ok", "✓ verified ") : el("span", "bad", "no verified engine "),
    document.createTextNode(`${v.name || ""} · bankML ${v.bankml || "?"} · ${String(v.model_sha256 || "").slice(0, 12)}…`));
  if (!slidersSet) {
    const r = s.resources || {}; const cores = u.cores || 4;
    $("threads").max = cores; $("threads").value = r.threads || 1;
    $("ram").max = Math.max(1, ((u.mem_total_bytes || 6 * GB) / GB)).toFixed(1); $("ram").value = r.ram_gb || 2;
    $("gpu").value = Math.round(100 * (r.gpu_limit ?? 0.8)); sync(); slidersSet = true;
    if (s.public) {  // a public console is read-only: the controls show the settings, they do not change them
      ["threads", "ram", "gpu", "apply"].forEach((id) => { $(id).disabled = true; });
      $("apply").textContent = "read-only: this is a public console";
    }
  }
  $("job").textContent = s.job.busy ? s.job.what : (s.job.error ? "failed: " + s.job.error : (s.job.done ? "done: the engine restarted, verified" : ""));
  const t = Date.now() / 1000; const g = (u.gpus || [])[0] || {};
  const push = (k, val) => { series[k].push(val); if (series[k].length > KEEP) series[k].shift(); };
  push("t", t); push("cpu", u.cpu_percent ?? null); push("rss", u.rss_bytes != null ? u.rss_bytes / GB : null);
  push("avail", u.mem_available_bytes != null ? u.mem_available_bytes / GB : null); push("gbusy", g.busy_percent ?? null);
  push("galloc", lim ? lim.allocated_bytes / MB : null); push("watts", u.package_watts ?? null);
  const pts = (k) => series.t.map((x, i) => ({ x, y: series[k][i] }));
  const recs = (m.records || []).slice(-60);
  chart("c-tps", [{ cls: "l1", data: recs.map((r) => ({ x: r.at, y: r.eval_tps })) }, { cls: "l2", data: recs.map((r) => ({ x: r.at, y: r.prompt_tps })) }], { xfmt: tfmt, empty: "no answers yet" });
  chart("c-ttft", [{ cls: "l1", data: recs.map((r) => ({ x: r.at, y: r.ttft_ms })) }], { xfmt: tfmt, empty: "no answers yet" });
  chart("c-cpu", [{ cls: "l1", data: pts("cpu") }], { xfmt: tfmt });
  chart("c-mem", [{ cls: "l1", data: pts("rss") }, { cls: "l2", data: pts("avail") }], { xfmt: tfmt });
  chart("c-gpu", [{ cls: "l1", data: pts("gbusy") }], { xfmt: tfmt, limit: lim ? 100 * lim.limit : null, empty: "no GPU reading" });
  chart("c-pow", [{ cls: "l1", data: pts("watts") }], { xfmt: tfmt, empty: "power not measured (./install.sh power)" });
  $("pow-note").textContent = `energy per token over the kept answers: ${fmt(m.joules_per_token, 3, " J")}`;
  const now = $("now"); now.textContent = "";
  nowRow(now, "tokens in / out", `${m.prompt_tokens ?? 0} / ${m.completion_tokens ?? 0}`);
  const last = (m.records || []).slice(-1)[0] || {};
  nowRow(now, "last TTFT", fmt(last.ttft_ms, 0, " ms")); nowRow(now, "last pp · tg", `${fmt(last.prompt_tps, 1)} · ${fmt(last.eval_tps, 1)} tok/s`);
  nowRow(now, "CPU", fmt(u.cpu_percent, 0, " %")); nowRow(now, "RSS", fmt(u.rss_bytes / GB, 2, " GB"));
  nowRow(now, "RAM available", fmt(u.mem_available_bytes / GB, 2, " GB"));
  nowRow(now, "GPU busy", fmt(g.busy_percent, 0, " %"));
  nowRow(now, "GPU limiter", lim ? `${(100 * lim.limit).toFixed(0)} % · ${(lim.allocated_bytes / MB).toFixed(0)} MB of ${(lim.heap_bytes / GB).toFixed(1)} GB · busy ${(100 * lim.busy).toFixed(0)} %` : "no card in use");
  nowRow(now, "power", fmt(u.package_watts, 1, " W"));
}

// ── logging ─────────────────────────────────────────────────────────────────────────────────────────────────────
async function loadLog() {
  const j = await (await fetch("/api/log")).json(); const tb = document.querySelector("#log tbody"); tb.textContent = "";
  j.exchanges.slice().reverse().forEach((x) => {
    const tr = el("tr"); const m = x.metrics || {}; const r = x.receipt || {};
    [tfmt(x.at), x.question.slice(0, 80), r.prompt_tokens != null ? `${r.prompt_tokens}+${r.completion_tokens}` : "—",
      fmt(m.ttft_ms, 0), fmt(m.eval_tps, 1), x.error ? "refused: " + x.error.slice(0, 60) : String(r.response_sha256 || "").slice(0, 16)]
      .forEach((c) => tr.append(el("td", null, c)));
    tb.append(tr);
  });
  $("englog").textContent = j.engine_log.join("\n");
}

// ── infotags ────────────────────────────────────────────────────────────────────────────────────────────────────
let info = null;
async function loadInfo() {
  info = await (await fetch("/api/infotags")).json();
  const tb = document.querySelector("#attrs tbody"); tb.textContent = "";
  info.attributes.forEach((a) => { const tr = el("tr"); tr.append(el("td", null, a.trait_type), el("td", null, String(a.value))); tb.append(tr); });
  $("infojson").textContent = JSON.stringify(info, null, 1);
}
$("download").addEventListener("click", () => {
  if (!info) return;
  const a = el("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(info, null, 1)], { type: "application/json" }));
  a.download = "bankml-infotags.json"; a.click(); URL.revokeObjectURL(a.href);
});

poll(); setInterval(poll, 2000);
