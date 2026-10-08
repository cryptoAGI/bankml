// SPDX-License-Identifier: MIT OR Apache-2.0
// The Space's landing and tabs: the highlights until the first question (then the answers take their place), the
// tabs under the conversation (FAQ, Logs, Release, Thesis, Advantages, About; arrow keys move between them, the
// address keeps the open one), and this page's own log, which the chat writes to through window.bankmlLog.
"use strict";
(function () {
  const $ = (id) => document.getElementById(id);

  // ── the log: what this page did, newest last, kept in this tab only ──────────────────────────────────────────
  const MAX = 300;
  const lines = [];
  window.bankmlLog = (kind, text) => {
    const at = new Date();
    lines.push({ at, kind, text: String(text) });
    if (lines.length > MAX) lines.shift();
    const ol = $("pagelog"); if (!ol) return;
    const empty = ol.querySelector(".empty"); if (empty) empty.remove();
    const li = document.createElement("li");
    li.className = "is-" + kind;
    const t = document.createElement("time"); t.textContent = at.toLocaleTimeString();
    const k = document.createElement("span"); k.className = "k"; k.textContent = kind;
    const x = document.createElement("span"); x.className = "x"; x.textContent = String(text);
    li.append(t, k, x);
    ol.append(li);
    while (ol.children.length > MAX) ol.firstChild.remove();
    const tab = $("tab-logs"); if (tab && kind === "error") tab.classList.add("has-news");
  };
  document.addEventListener("click", async (e) => {
    if (e.target.id === "logclear") { lines.length = 0; const ol = $("pagelog"); ol.textContent = ""; const li = document.createElement("li"); li.className = "empty"; li.textContent = "cleared"; ol.append(li); }
    if (e.target.id === "logcopy") {
      const txt = lines.map((l) => `${l.at.toISOString()} ${l.kind} ${l.text}`).join("\n");
      try { await navigator.clipboard.writeText(txt); e.target.textContent = "copied"; } catch (err) { e.target.textContent = "select and copy"; }
      setTimeout(() => (e.target.textContent = "copy"), 1500);
    }
  });

  // ── the highlights: shown until the first question, then they give way to the answers ───────────────────────
  window.bankmlDismissHighlights = () => {
    const h = $("highlights"); if (!h || h.classList.contains("leaving")) return;
    h.classList.add("leaving");
    setTimeout(() => h.remove(), matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 380);
  };

  // ── the tabs ─────────────────────────────────────────────────────────────────────────────────────────────────
  const tabs = () => [...document.querySelectorAll('.tabs [role="tab"]')];
  function open(name, focus) {
    for (const t of tabs()) {
      const on = t.id === "tab-" + name;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      const p = $(t.getAttribute("aria-controls")); if (p) p.hidden = !on;
      if (on) { t.classList.remove("has-news"); if (focus) t.focus(); }
    }
    try { history.replaceState(null, "", "#" + name); } catch (err) { /* a sandboxed frame */ }
  }
  document.addEventListener("click", (e) => {
    const t = e.target.closest('.tabs [role="tab"]'); if (t) { open(t.id.slice(4)); return; }
    const a = e.target.closest('a[href^="#"]'); if (!a) return;
    const name = a.getAttribute("href").slice(1);
    if ($("tab-" + name)) { e.preventDefault(); open(name); $("tab-" + name).scrollIntoView({ block: "start", behavior: "smooth" }); }
  });
  document.addEventListener("keydown", (e) => {
    const t = e.target.closest && e.target.closest('.tabs [role="tab"]'); if (!t) return;
    const all = tabs(), i = all.indexOf(t);
    const j = e.key === "ArrowRight" ? (i + 1) % all.length : e.key === "ArrowLeft" ? (i - 1 + all.length) % all.length
      : e.key === "Home" ? 0 : e.key === "End" ? all.length - 1 : -1;
    if (j >= 0) { e.preventDefault(); open(all[j].id.slice(4), true); }
  });
  document.addEventListener("DOMContentLoaded", () => {
    const want = location.hash.slice(1);
    if (want && $("tab-" + want)) open(want);
  });
})();
