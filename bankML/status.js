// SPDX-License-Identifier: MIT OR Apache-2.0
// The engine's status, drawn from `GET /bankml/status`: checks, CPU, memory, disk, GPU, the answers measured and the
// engine's own log. One renderer for two pages: bankml serve's own page (GET / from a browser, this file inlined)
// and the bankML console's Engine tab (served as /engine.js). Text only: nothing the engine returns becomes markup.
(function (global) {
  "use strict";
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined && text !== null) e.textContent = text; return e; };
  const gb = (b) => (b === null || b === undefined ? "not measured" : (b / 1e9).toFixed(2) + " GB");
  const num = (v, d = 1, unit = "") => (v === null || v === undefined || Number.isNaN(+v) ? "not measured" : (+v).toFixed(d) + unit);
  const short = (h, n = 16) => (h ? String(h).slice(0, n) + "…" : "—");
  const LEVEL = ["error", "warn", "info", "debug"];

  function card(title, rows, extra) {
    const c = el("section", "bs-card");
    c.append(el("h3", null, title));
    const dl = el("dl", "bs-facts");
    for (const [k, v] of rows) { dl.append(el("dt", null, k), el("dd", null, v)); }
    c.append(dl);
    if (extra) c.append(extra);
    return c;
  }
  function meter(used, total, label) {
    const w = el("div", "bs-meter");
    const f = el("span", "bs-fill");
    if (used !== null && used !== undefined && total) f.style.width = Math.min(100, (100 * used) / total).toFixed(1) + "%";
    w.append(f);
    const box = el("div", "bs-meterbox");
    box.append(w, el("span", "bs-meterlabel", label));
    return box;
  }

  function render(root, s) {
    root.textContent = "";
    const u = s.usage || {}, m = s.metrics || {}, v = s.verified || {};
    // the checks first: what to act on
    const checks = el("ul", "bs-checks");
    for (const c of s.checks || []) {
      const li = el("li", "bs-" + c.level);
      li.append(el("span", "bs-mark", c.level === "ok" ? "✓" : c.level === "warn" ? "!" : c.level === "bad" ? "✗" : "·"), el("span", "bs-name", c.check), el("span", "bs-seen", c.seen));
      checks.append(li);
    }
    root.append(checks);
    const grid = el("div", "bs-grid");
    grid.append(card("Engine", [
      ["bankML", s.bankml], ["model", (v.name || "—") + " · " + (v.arch || "")], ["sha256", short(v.model_sha256)],
      ["guard", v.guard || "none"], ["native", s.serve && s.serve.native ? "yes: bankML's own forward pass" : "no: llama-server behind the gate"],
      ["listening", (s.serve && s.serve.listen) || "—"], ["up", num(s.serve && s.serve.uptime_s, 0, " s")],
      ["threads", (s.serve && s.serve.threads) || "default"], ["KV cache", (s.serve && s.serve.cache_type) || "f16"],
    ]));
    const cpu = s.cpu || {};
    const mhz = cpu.mhz || [];
    grid.append(card("CPU", [
      ["processor", cpu.model || "not measured"], ["logical CPUs", String(cpu.logical || "—")],
      ["clock now", mhz.length ? mhz.map((x) => Math.round(x)).join(" · ") + " MHz" : "not measured"],
      ["bankML uses", num(u.cpu_percent, 0, " % of one core")], ["package power", num(u.package_watts, 1, " W")],
    ]));
    grid.append(card("Memory", [
      ["total", gb(u.mem_total_bytes)], ["available", gb(u.mem_available_bytes)], ["bankML holds (RSS)", gb(u.rss_bytes)],
      ["swap used", u.swap_total_bytes ? gb(u.swap_total_bytes - u.swap_free_bytes) + " of " + gb(u.swap_total_bytes) : "no swap"],
    ], meter(u.mem_total_bytes - u.mem_available_bytes, u.mem_total_bytes, "in use")));
    const d = s.disk || {};
    grid.append(card("Disk", [
      ["model file", gb(d.model_bytes)], ["on", d.path || "—"], ["total", gb(d.total_bytes)], ["available", gb(d.available_bytes)],
      ["bankML read", gb(d.read_bytes)], ["bankML wrote", gb(d.write_bytes)],
    ], meter(d.total_bytes - d.available_bytes, d.total_bytes, "in use")));
    const gpus = u.gpus || [];
    const lim = u.gpu_limiter || {};
    const gpuRows = gpus.length ? gpus.flatMap((g) => [
      [g.card + " (" + (g.driver || "?") + ")", num(g.busy_percent, 0, " % busy")],
      ["VRAM", gb(g.vram_used_bytes) + " of " + gb(g.vram_total_bytes)],
      ["GTT", gb(g.gtt_used_bytes) + " of " + gb(g.gtt_total_bytes)],
    ]) : [["cards", "none found"]];
    gpuRows.push(["bankML's limit", lim.limit === undefined || lim.limit === null ? "not set" : num(lim.limit * 100, 0, " % of memory and time")]);
    gpuRows.push(["bankML holds", gb(lim.allocated_bytes)]);
    grid.append(card("GPU", gpuRows));
    const recs = m.records || [];
    const last = recs[recs.length - 1] || {};
    grid.append(card("Answers measured", [
      ["answers", String(recs.length)], ["tokens read", num(m.prompt_tokens, 0)], ["tokens written", num(m.completion_tokens, 0)],
      ["last: time to first token", num(last.ttft_ms, 0, " ms")], ["last: prompt", num(last.prompt_tps, 2, " tok/s")],
      ["last: generation", num(last.eval_tps, 2, " tok/s")], ["energy per token", num(m.joules_per_token, 3, " J")],
    ]));
    root.append(grid);
    const lh = el("h3", "bs-loghead", "Engine log — newest last");
    const pre = el("pre", "bs-log");
    pre.textContent = (s.log || []).map((l) => new Date(l.at * 1000).toTimeString().slice(0, 8) + " " + (LEVEL[l.level] || l.level).padEnd(5) + " " + l.msg).join("\n") || "nothing logged yet";
    root.append(lh, pre);
    root.append(el("p", "bs-foot", "GET /bankml/status · every reading measured now, \"not measured\" when it was not · " + new Date((s.at || 0) * 1000).toLocaleTimeString()));
  }

  global.BankmlStatus = { render };
})(typeof window !== "undefined" ? window : this);
