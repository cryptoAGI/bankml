/*! bankml-creds.js — SPDX-License-Identifier: GPL-3.0-only
 * The visitor's Hugging Face credential, held by the visitor's browser, for the visitor: sign-in (OAuth, PKCE),
 * the token's storage (this tab by default; remembered on this device only when chosen), the only function that
 * sends it (to huggingface.co and router.huggingface.co, nowhere else), and the "your credentials" panel.
 * Copyright (C) 2026 cryptoAGI — Professor Codephreak and Gregory L. Magnusson. GNU GPL version 3 only; the full text:
 * https://www.gnu.org/licenses/gpl-3.0.txt and src/creds/LICENSE. Source: src/creds/ of bankML UIF. */
const SESSION_KEY = "bankml-creds:session";
const DEVICE_KEY = "bankml-creds:device";
const LEGACY_SESSION_KEY = "bankml.oauth";
const PASTED_REMEMBER_DAYS = 30;
const read = (s, k) => {
  try {
    const v = JSON.parse(s?.getItem(k) || "null");
    return v && typeof v.token === "string" && v.token ? v : null;
  } catch {
    return null;
  }
};
const write = (s, k, v) => {
  try {
    v ? s?.setItem(k, JSON.stringify(v)) : s?.removeItem(k);
  } catch {
  }
};
const expired = (k, now = Date.now()) => !!k.expiresAt && new Date(k.expiresAt).getTime() <= now || !!k.rememberUntil && new Date(k.rememberUntil).getTime() <= now;
class CredStore {
  constructor(st, now = () => Date.now()) {
    this.st = st;
    this.now = now;
    this.mem = null;
    this.migrate();
  }
  /** 0.2.4's session copy (the OAuth result as @huggingface/hub returned it), moved into this store once. */
  migrate() {
    try {
      const raw = this.st.session?.getItem(LEGACY_SESSION_KEY);
      if (!raw) return;
      this.st.session?.removeItem(LEGACY_SESSION_KEY);
      const o = JSON.parse(raw);
      if (o?.accessToken) {
        this.put({
          token: o.accessToken,
          via: "oauth",
          user: o.userInfo?.preferred_username || o.userInfo?.name || null,
          expiresAt: o.accessTokenExpiresAt ? new Date(o.accessTokenExpiresAt).toISOString() : null,
          scopes: String(o.scope || "").split(/\s+/).filter(Boolean)
        }, false);
      }
    } catch {
    }
  }
  /** The token kept now (memory, then this tab, then this device), or null; an expired copy is removed. */
  current() {
    const k = this.mem ?? read(this.st.session, SESSION_KEY) ?? read(this.st.device, DEVICE_KEY);
    if (!k) return null;
    if (expired(k, this.now())) {
      this.clear();
      return null;
    }
    this.mem = k;
    return k;
  }
  /** Where it is kept: on this device (remembered), or in this tab only. */
  where() {
    if (!this.current()) return null;
    return read(this.st.device, DEVICE_KEY) ? "device" : "session";
  }
  /** Keep a token: in this tab; on this device too only when `remember`. */
  put(k, remember) {
    const kept = { ...k };
    delete kept.rememberUntil;
    this.mem = kept;
    write(this.st.session, SESSION_KEY, kept);
    this.remember(remember);
  }
  /** Remember on this device (until the token's expiry, or 30 days for a pasted one), or stop remembering. */
  remember(on) {
    const k = this.mem ?? this.current();
    if (!on || !k) {
      write(this.st.device, DEVICE_KEY, null);
      return;
    }
    const cap = this.now() + PASTED_REMEMBER_DAYS * 864e5;
    const until = k.expiresAt ? Math.min(new Date(k.expiresAt).getTime(), cap) : cap;
    write(this.st.device, DEVICE_KEY, { ...k, rememberUntil: new Date(until).toISOString() });
  }
  /** The device copy's expiry, when remembered. */
  rememberedUntil() {
    return read(this.st.device, DEVICE_KEY)?.rememberUntil ?? null;
  }
  /** Forget: remove the copy on this device; this tab keeps it until it closes. */
  forget() {
    write(this.st.device, DEVICE_KEY, null);
  }
  /** Sign out: every copy, memory included. */
  clear() {
    this.mem = null;
    write(this.st.session, SESSION_KEY, null);
    write(this.st.device, DEVICE_KEY, null);
  }
}
const TOKEN_HOSTS = ["huggingface.co", "router.huggingface.co"];
function tokenMayGo(url) {
  let u;
  try {
    u = new URL(url);
  } catch {
    return false;
  }
  return u.protocol === "https:" && !u.port && !u.username && !u.password && TOKEN_HOSTS.includes(u.hostname);
}
const STYLE_ID = "bankml-creds-style";
const CSS = `
.bkc { display: grid; gap: 8px; font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--bkc-text, #e2e8f0); }
.bkc h3 { margin: 0; font-size: 13px; text-transform: uppercase; letter-spacing: .08em; color: var(--bkc-dim, #94a3b8); }
.bkc p { margin: 0; overflow-wrap: anywhere; }
.bkc .bkc-state.is-in { color: var(--candle-green, #0ECB81); }
.bkc .bkc-fine { font-size: 12px; color: var(--bkc-dim, #94a3b8); }
.bkc .bkc-bad { color: var(--stop-red, #F6465D); }
.bkc .bkc-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.bkc button, .bkc a.bkc-btn { min-height: 44px; padding: 0 14px; border-radius: 10px; font: 600 14px system-ui, sans-serif; cursor: pointer;
  border: 1px solid var(--btc-orange, #F7931A); background: transparent; color: var(--btc-orange, #F7931A); text-decoration: none; display: inline-flex; align-items: center; }
.bkc button.bkc-primary { background: var(--btc-orange, #F7931A); color: #0b0f14; }
.bkc button.bkc-danger { border-color: var(--stop-red, #F6465D); color: var(--stop-red, #F6465D); }
.bkc button:focus-visible, .bkc a.bkc-btn:focus-visible, .bkc input:focus-visible { outline: 2px solid var(--btc-orange, #F7931A); outline-offset: 2px; }
.bkc label.bkc-check { display: inline-flex; gap: 8px; align-items: center; min-height: 44px; cursor: pointer; }
.bkc label.bkc-check input { width: 20px; height: 20px; accent-color: var(--btc-orange, #F7931A); }
.bkc input.bkc-token { flex: 1 1 14rem; min-width: 0; min-height: 44px; padding: 0 12px; border-radius: 10px; font: 16px ui-monospace, monospace;
  color: var(--uif-field-text, #f1f5f9); background: var(--uif-field-bg, #0f151c); border: 1px solid #334155; }
.bkc dl { margin: 0; display: grid; gap: 4px; }
.bkc dl div { font-size: 13px; overflow-wrap: anywhere; }
`;
function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  Object.assign(e, props);
  for (const k of kids) if (k !== null) e.append(k);
  return e;
}
function mountPanel(host) {
  if (!document.getElementById(STYLE_ID)) document.head.append(el("style", { id: STYLE_ID, textContent: CSS }));
  let note = redirectNote ?? "";
  let busy = false;
  const root = el("section", { className: "bkc" });
  root.setAttribute("aria-label", "Your Hugging Face credentials");
  host.append(root);
  const say = (t) => {
    note = t;
    draw();
  };
  function draw() {
    const s = status();
    root.replaceChildren();
    root.append(el("h3", { textContent: "Your credentials" }));
    if (!s.signedIn) {
      root.append(el("p", { className: "bkc-state", textContent: "Not signed in to Hugging Face. Nothing is kept." }));
      const remember = el("input", { type: "checkbox", checked: false });
      const rememberRow = el("label", { className: "bkc-check" }, remember, "remember on this device (off: gone when this tab closes)");
      if (s.oauthAvailable) {
        const go = el("button", { type: "button", className: "bkc-primary", textContent: "Sign in with Hugging Face", disabled: busy });
        go.onclick = async () => {
          busy = true;
          draw();
          const n = await signIn({ remember: remember.checked });
          busy = false;
          if (n) say(n);
        };
        root.append(el("div", { className: "bkc-row" }, go, rememberRow));
        root.append(el("p", { className: "bkc-fine", textContent: "Asks Hugging Face for your name and for inference calls only (scopes openid, profile, inference-api), with PKCE. Nothing else of your account." }));
      } else {
        root.append(el("p", { textContent: "Hugging Face sign-in needs a Hugging Face Space; this page is not one. You may paste a token instead — optional; everything else here works without it." }));
        const tok = el("input", { type: "password", className: "bkc-token", placeholder: "hf_…", autocomplete: "off", spellcheck: false });
        tok.setAttribute("aria-label", "Hugging Face access token (optional)");
        const use = el("button", { type: "submit", className: "bkc-primary", textContent: "Use this token", disabled: busy });
        const form = el("form", { className: "bkc-row" }, tok, use);
        form.onsubmit = async (e) => {
          e.preventDefault();
          busy = true;
          draw();
          const n = await usePastedToken(tok.value, remember.checked);
          tok.value = "";
          busy = false;
          say(n || "");
        };
        root.append(form, el("div", { className: "bkc-row" }, rememberRow));
        const link = el("a", { href: "https://huggingface.co/settings/tokens", target: "_blank", rel: "noreferrer", textContent: "huggingface.co/settings/tokens" });
        root.append(el("p", { className: "bkc-fine" }, 'Make a fine-grained token with only "Make calls to Inference Providers" at ', link, ". It is checked once with huggingface.co, then sent only to huggingface.co and router.huggingface.co."));
      }
      const join = el("a", { className: "bkc-btn", href: "https://huggingface.co/join", target: "_blank", rel: "noreferrer", textContent: "Create a Hugging Face account ↗" });
      root.append(el("div", { className: "bkc-row" }, join));
    } else {
      root.append(el("p", { className: "bkc-state is-in", textContent: `Signed in as ${s.user ?? "you"} (${s.via === "oauth" ? "Hugging Face sign-in" : "a pasted token"}).` }));
      const dl = el("dl");
      for (const line of describe()) dl.append(el("div", { textContent: line }));
      root.append(dl);
      const until = rememberedUntil();
      const rem = el("input", { type: "checkbox", checked: !!until });
      rem.onchange = () => setRemember(rem.checked);
      root.append(el("label", { className: "bkc-check" }, rem, until ? `remembered on this device until ${new Date(until).toLocaleString()}` : "remember on this device (opt-in)"));
      const fgt = el("button", { type: "button", textContent: "Forget (this device)", disabled: !until });
      fgt.onclick = () => {
        forget();
        say("Forgotten on this device: this tab keeps it until it closes.");
      };
      const out = el("button", { type: "button", className: "bkc-danger", textContent: "Sign out" });
      out.onclick = () => {
        signOut();
        say("Signed out: nothing is kept.");
      };
      root.append(el("div", { className: "bkc-row" }, fgt, out));
    }
    if (note) root.append(el("p", { className: /not|did not|could not|refused/i.test(note) && !/^Forgotten|^Signed out/.test(note) ? "bkc-bad" : "bkc-fine", textContent: note, role: "status" }));
    root.append(el("p", { className: "bkc-fine", textContent: "This panel and the token handling are bankml-creds.js, GPL-3.0-only (its source is published with every build); the rest of the page is MIT." }));
  }
  draw();
  const off = onChange(draw);
  return () => {
    off();
    root.remove();
  };
}
const VERSION = "1.0.0";
const HUB_ESM = "https://cdn.jsdelivr.net/npm/@huggingface/hub@2.17.1/+esm";
const SCOPES = "openid profile inference-api";
const REMEMBER_PENDING = "bankml-creds:remember-pending";
const safe = (f) => {
  try {
    return f();
  } catch {
    return null;
  }
};
const store = new CredStore({ session: safe(() => sessionStorage), device: safe(() => localStorage) });
const fns = /* @__PURE__ */ new Set();
const changed = () => {
  fns.forEach((f) => {
    try {
      f();
    } catch {
    }
  });
  window.dispatchEvent(new Event("bankmlcreds:change"));
};
const esm = (url) => import(
  /* @vite-ignore */
  url
);
const oauthAvailable = () => !!window.huggingface?.variables?.OAUTH_CLIENT_ID;
function status() {
  const k = store.current();
  return {
    signedIn: !!k,
    user: k?.user ?? null,
    via: k?.via ?? null,
    where: store.where(),
    expiresAt: k?.expiresAt ?? null,
    scopes: k?.scopes ?? [],
    oauthAvailable: oauthAvailable()
  };
}
const when = (iso) => iso ? new Date(iso).toLocaleString() : "not set";
function describe() {
  const k = store.current();
  if (!k) return ["Nothing is kept: you are not signed in to Hugging Face on this page."];
  const until = store.rememberedUntil();
  return [
    `what: a Hugging Face access token for ${k.user ?? "your account"} — ${k.via === "oauth" ? "from Hugging Face sign-in (OAuth, PKCE)" : "pasted by you"}`,
    `where: ${until ? `this tab, and remembered on this device (localStorage) until ${when(until)}` : "this tab only (memory and sessionStorage): gone when the tab closes"}`,
    `scopes: ${k.scopes.length ? k.scopes.join(", ") : k.via === "pasted" ? 'as you made the token on huggingface.co (fine-grained, "Make calls to Inference Providers" is all it needs)' : "not reported"}`,
    `expires: ${k.expiresAt ? when(k.expiresAt) : k.via === "pasted" ? "as you set it on huggingface.co" : "not reported"}`,
    `sent only to: ${TOKEN_HOSTS.join(", ")} — never logged, never written to .history, .memory, a .profile or a layout`
  ];
}
async function signIn(opts = {}) {
  const v = window.huggingface?.variables;
  if (!v?.OAUTH_CLIENT_ID) return "Hugging Face sign-in works only on a Hugging Face Space with OAuth. Here you may paste a token instead (optional).";
  const { oauthLoginUrl } = await esm(HUB_ESM);
  try {
    sessionStorage.setItem(REMEMBER_PENDING, opts.remember ? "1" : "0");
  } catch {
  }
  location.href = await oauthLoginUrl({ clientId: v.OAUTH_CLIENT_ID, scopes: SCOPES });
}
async function handleRedirect() {
  if (!/[?&]code=/.test(location.search) || !/[?&]state=/.test(location.search)) return null;
  try {
    const { oauthHandleRedirectIfPresent } = await esm(HUB_ESM);
    const res = await oauthHandleRedirectIfPresent();
    if (res?.accessToken) {
      let remember = false;
      try {
        remember = sessionStorage.getItem(REMEMBER_PENDING) === "1";
        sessionStorage.removeItem(REMEMBER_PENDING);
      } catch {
      }
      store.put({
        token: res.accessToken,
        via: "oauth",
        user: res.userInfo?.preferred_username || res.userInfo?.name || null,
        expiresAt: res.accessTokenExpiresAt ? new Date(res.accessTokenExpiresAt).toISOString() : null,
        scopes: String(res.scope || SCOPES).split(/\s+/).filter(Boolean)
      }, remember);
    }
    return null;
  } catch (e) {
    return `Hugging Face sign-in did not finish: ${String(e?.message || e).slice(0, 160)}`;
  } finally {
    history.replaceState(null, "", location.pathname + location.hash);
  }
}
let redirectNote = null;
async function usePastedToken(token, remember) {
  const t = token.trim();
  if (!/^hf_[A-Za-z0-9]{20,}$/.test(t)) return "That is not a Hugging Face token (they start with hf_).";
  let r;
  try {
    r = await fetch("https://huggingface.co/api/whoami-v2", { headers: { Authorization: `Bearer ${t}` }, credentials: "omit", referrerPolicy: "no-referrer", redirect: "error" });
  } catch {
    return "huggingface.co could not be reached to check the token.";
  }
  if (r.status === 401) return "Hugging Face did not accept that token.";
  if (!r.ok) return `Hugging Face answered HTTP ${r.status} when the token was checked.`;
  const who = await r.json().catch(() => ({}));
  const a = who?.auth?.accessToken ?? {};
  const scopes = a.role === "fineGrained" ? (a.fineGrained?.global ?? []).map((s) => `fine-grained: ${s}`) : a.role ? [`${a.role} (a classic token: every ${a.role} permission — a fine-grained token limited to inference is better)`] : [];
  const kept = { token: t, via: "pasted", user: who?.name ?? null, expiresAt: null, scopes };
  store.put(kept, remember);
  changed();
}
function signOut() {
  store.clear();
  changed();
}
function forget() {
  store.forget();
  changed();
}
function setRemember(on) {
  store.remember(on);
  changed();
}
const rememberedUntil = () => store.rememberedUntil();
async function fetchWithToken(url, init = {}) {
  if (!tokenMayGo(url)) throw new Error(`the token is sent only to ${TOKEN_HOSTS.join(" and ")} over https: refused ${(() => {
    try {
      return new URL(url).host;
    } catch {
      return "an invalid address";
    }
  })()}`);
  const k = store.current();
  if (!k) {
    changed();
    throw new Error("not signed in to Hugging Face (or the sign-in expired)");
  }
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${k.token}`);
  return fetch(url, { ...init, headers, credentials: "omit", referrerPolicy: "no-referrer", redirect: "error" });
}
function onChange(f) {
  fns.add(f);
  return () => {
    fns.delete(f);
  };
}
const api = {
  version: VERSION,
  licence: "GPL-3.0-only",
  status,
  signIn,
  signOut,
  describe,
  fetchWithToken,
  onChange,
  mountPanel: (el2) => mountPanel(el2)
};
void handleRedirect().then((note) => {
  redirectNote = note;
  Object.defineProperty(window, "bankmlCreds", { value: Object.freeze(api), configurable: false, writable: false });
  window.dispatchEvent(new Event("bankmlcreds:ready"));
  changed();
});
export {
  SCOPES,
  VERSION,
  describe,
  fetchWithToken,
  forget,
  onChange,
  redirectNote,
  rememberedUntil,
  setRemember,
  signIn,
  signOut,
  status,
  store,
  usePastedToken
};
