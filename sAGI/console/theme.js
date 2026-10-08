// SPDX-License-Identifier: MIT OR Apache-2.0
// bankML console theme: loaded in <head> so the page paints in the right theme. data-theme on <html> drives theme.css,
// and the class "dark" drives the input field's own tokens. The choice is kept per browser; the system's is the default.
"use strict";
(function () {
  const KEY = "bankml.theme";
  const root = document.documentElement;
  const stored = () => { try { return localStorage.getItem(KEY); } catch (e) { return null; } };
  const system = () => (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  function apply(theme) {
    root.dataset.theme = theme;
    root.classList.toggle("dark", theme === "dark");
    const b = document.getElementById("theme-toggle");
    if (b) {
      const dark = theme === "dark";
      b.setAttribute("aria-pressed", String(dark));
      b.title = dark ? "Light mode" : "Dark mode";
      b.setAttribute("aria-label", b.title);
      b.textContent = dark ? "☀" : "☾";
    }
  }
  apply(stored() || system());
  if (window.matchMedia) {
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (!stored()) apply(system()); });
  }
  document.addEventListener("DOMContentLoaded", () => {
    const b = document.getElementById("theme-toggle");
    if (!b) return;
    apply(root.dataset.theme);
    b.addEventListener("click", () => {
      const next = root.dataset.theme === "dark" ? "light" : "dark";
      try { localStorage.setItem(KEY, next); } catch (e) { /* private window: this page only */ }
      apply(next);
    });
  });
})();
