// SPDX-License-Identifier: MIT OR Apache-2.0
// The console's windows as a desk: drag a window by its bar to reorder it, resize it by its corner (its height, and its
// width as one column or both), and use the three lights — red collapses, amber widens, green focuses (Esc returns).
// Alt+↑/↓ moves the focused window. The arrangement is kept per browser (localStorage) and Reset restores it.
// A container opts in with data-desk="<name>"; each .window child is one window, named by data-win.
"use strict";
(function () {
  const load = (k) => { try { return JSON.parse(localStorage.getItem(k) || "null"); } catch (e) { return null; } };
  const save = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* a private window: this page only */ } };

  function desk(box) {
    const KEY = "bankml.desk." + box.dataset.desk;
    const wins = () => [...box.querySelectorAll(":scope > .window")];
    const initial = wins().map((w) => w.dataset.win);
    let st = load(KEY) || {};
    const persist = () => {
      st = { order: wins().map((w) => w.dataset.win), span: {}, height: {}, collapsed: {} };
      for (const w of wins()) {
        const id = w.dataset.win;
        if (w.classList.contains("span-2")) st.span[id] = 2;
        if (w.style.height) st.height[id] = parseInt(w.style.height, 10);
        if (w.classList.contains("collapsed")) st.collapsed[id] = true;
      }
      save(KEY, st);
    };
    const apply = () => {
      const byId = Object.fromEntries(wins().map((w) => [w.dataset.win, w]));
      for (const id of [...(st.order || []), ...initial]) if (byId[id]) { box.append(byId[id]); delete byId[id]; }
      for (const w of wins()) {
        const id = w.dataset.win;
        w.classList.toggle("span-2", (st.span || {})[id] === 2);
        w.classList.toggle("collapsed", !!(st.collapsed || {})[id]);
        w.style.height = (st.height || {})[id] && !(st.collapsed || {})[id] ? st.height[id] + "px" : "";
      }
    };

    // ── the lights, the grip ──────────────────────────────────────────────────────────────────────────────────
    for (const w of wins()) {
      const bar = w.querySelector(".window-bar");
      const title = (bar.querySelector("b") || {}).textContent || w.dataset.win;
      const lights = bar.querySelector(".lights");
      lights.removeAttribute("aria-hidden");
      lights.textContent = "";
      for (const [cls, label, act] of [["red", "collapse", "collapse"], ["amber", "half or full width", "span"], ["green", "focus", "focus"]]) {
        const b = document.createElement("button");
        b.type = "button"; b.className = "light " + cls; b.dataset.act = act;
        b.title = label; b.setAttribute("aria-label", `${label}: ${title}`);
        lights.append(b);
      }
      bar.tabIndex = 0;
      bar.setAttribute("aria-label", `${title} — drag to move, Alt+↑/↓ to move, double-click to widen`);
      const grip = document.createElement("div");
      grip.className = "win-grip"; grip.title = "drag to resize";
      w.append(grip);
    }

    // ── focus (green), and its veil ──────────────────────────────────────────────────────────────────────────
    const veil = document.createElement("div");
    veil.className = "desk-veil";
    document.body.append(veil);
    const unfocus = () => { for (const w of wins()) w.classList.remove("focused"); veil.classList.remove("on"); };
    veil.addEventListener("click", unfocus);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") unfocus(); });

    box.addEventListener("click", (e) => {
      const b = e.target.closest(".light"); if (!b) return;
      const w = b.closest(".window");
      if (b.dataset.act === "collapse") { w.classList.toggle("collapsed"); if (w.classList.contains("collapsed")) w.style.height = ""; }
      if (b.dataset.act === "span") w.classList.toggle("span-2");
      if (b.dataset.act === "focus") { const on = !w.classList.contains("focused"); unfocus(); if (on) { w.classList.add("focused"); veil.classList.add("on"); } }
      persist();
    });
    box.addEventListener("dblclick", (e) => {
      const bar = e.target.closest(".window-bar"); if (!bar || e.target.closest("button")) return;
      bar.parentElement.classList.toggle("span-2"); persist();
    });
    box.addEventListener("keydown", (e) => {
      const bar = e.target.closest(".window-bar"); if (!bar || !e.altKey || (e.key !== "ArrowUp" && e.key !== "ArrowDown")) return;
      e.preventDefault();
      const w = bar.parentElement, sib = e.key === "ArrowUp" ? w.previousElementSibling : w.nextElementSibling;
      if (sib && sib.classList.contains("window")) { e.key === "ArrowUp" ? box.insertBefore(w, sib) : box.insertBefore(sib, w); bar.focus(); persist(); }
    });

    // ── drag a window by its bar: it lifts, a slot shows where it lands ──────────────────────────────────────────
    box.addEventListener("pointerdown", (e) => {
      const bar = e.target.closest(".window-bar");
      if (!bar || e.button !== 0 || e.target.closest("button") || bar.parentElement.classList.contains("focused")) return;
      const w = bar.parentElement, r = w.getBoundingClientRect(), dx = e.clientX - r.left, dy = e.clientY - r.top, h0 = w.style.height;
      let slot = null;
      const start = { x: e.clientX, y: e.clientY };
      const move = (ev) => {
        if (!slot) {
          if (Math.hypot(ev.clientX - start.x, ev.clientY - start.y) < 5) return;
          slot = document.createElement("div");
          slot.className = "win-slot" + (w.classList.contains("span-2") ? " span-2" : "");
          slot.style.height = r.height + "px";
          box.insertBefore(slot, w);
          Object.assign(w.style, { position: "fixed", left: r.left + "px", top: r.top + "px", width: r.width + "px", height: r.height + "px", zIndex: 60 });
          w.classList.add("dragging");
          document.body.classList.add("desk-dragging");
        }
        w.style.left = ev.clientX - dx + "px";
        w.style.top = ev.clientY - dy + "px";
        for (const o of wins()) {
          if (o === w) continue;
          const b = o.getBoundingClientRect();
          if (ev.clientX < b.left || ev.clientX > b.right || ev.clientY < b.top || ev.clientY > b.bottom) continue;
          const after = box.querySelector(":scope > .win-slot") && (ev.clientY > b.top + b.height / 2 || (ev.clientX > b.left + b.width / 2 && ev.clientY > b.top + b.height / 4));
          box.insertBefore(slot, after ? o.nextSibling : o);
          break;
        }
      };
      const up = () => {
        document.removeEventListener("pointermove", move);
        document.removeEventListener("pointerup", up);
        if (!slot) return;
        box.insertBefore(w, slot); slot.remove();
        for (const k of ["position", "left", "top", "width", "zIndex"]) w.style[k] = "";
        w.style.height = h0;  // the height it had: set by its grip, or none
        w.classList.remove("dragging");
        document.body.classList.remove("desk-dragging");
        persist();
      };
      document.addEventListener("pointermove", move);
      document.addEventListener("pointerup", up);
    });

    // ── resize by the corner: the height in pixels, the width as one column or both ──────────────────────────────
    box.addEventListener("pointerdown", (e) => {
      const g = e.target.closest(".win-grip"); if (!g || e.button !== 0) return;
      e.preventDefault();
      const w = g.parentElement, r = w.getBoundingClientRect(), x0 = e.clientX, y0 = e.clientY;
      const col = (box.getBoundingClientRect().width - 18) / 2;
      const wide = getComputedStyle(box).gridTemplateColumns.split(" ").length > 1;
      w.classList.remove("collapsed");
      w.classList.add("resizing");
      const move = (ev) => {
        w.style.height = Math.max(120, Math.min(1600, r.height + ev.clientY - y0)) + "px";
        if (wide) w.classList.toggle("span-2", r.width + ev.clientX - x0 > col * 1.4);
      };
      const up = () => {
        document.removeEventListener("pointermove", move); document.removeEventListener("pointerup", up);
        w.classList.remove("resizing"); persist();
      };
      document.addEventListener("pointermove", move);
      document.addEventListener("pointerup", up);
    });

    const reset = document.querySelector(`[data-desk-reset="${box.dataset.desk}"]`);
    if (reset) reset.addEventListener("click", () => { st = {}; save(KEY, null); apply(); });
    apply();
  }

  document.addEventListener("DOMContentLoaded", () => document.querySelectorAll("[data-desk]").forEach(desk));
})();
