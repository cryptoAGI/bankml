// SPDX-License-Identifier: MIT OR Apache-2.0
// Ask bankML, from this static page: two ways, never confused.
//  · Your own bankML — this browser talks to `bankml serve` on the visitor's machine (started with
//    --allow-origin for this page): bankML's verified arithmetic, the visitor's CPU and RAM, a receipt on every
//    answer whose sha256 this page checks. Free.
//  · A Hugging Face provider — sign in with Hugging Face; a provider-hosted model answers under bankML's persona,
//    billed to the visitor's inference quota. NOT bankML's arithmetic, no receipt; said on every answer.
// Nothing an engine or a provider returns becomes markup: text only.
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

// ── the persona: the same file the bankML console speaks from ─────────────────────────────────────────────────
async function loadPersona() {
  try {
    const r = await fetch("sAGI/personas/bankml.persona", { cache: "no-store" });
    persona = await r.json();
  } catch (e) {
    persona = null;
    $("askstatus").textContent = "the persona could not be read: " + e;
  }
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
function waiting(out, what) {
  const t0 = Date.now();
  const tick = () => { if (out.dataset.started !== "1") out.textContent = `… ${what} · ${Math.round((Date.now() - t0) / 1000)} s`; };
  tick();
  const id = setInterval(tick, 1000);
  return () => { out.dataset.started = "1"; clearInterval(id); };
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
  const b = await get(ep, "/bankml");
  if (!b) {
    $("localstatus").textContent = "not reachable — is bankml serve running with --allow-origin " + location.origin + " ? (see below)";
    return false;
  }
  const v = b.verified || {};
  $("localstatus").textContent = v.guard === "play"
    ? `✓ connected: ${v.name || b.resident || "model"} · bankML ${v.bankml} · sha256 ${String(v.model_sha256 || "").slice(0, 12)}…`
    : "connected, but no verified model is loaded yet";
  return v.guard === "play";
}
async function askLocal(message) {
  const ep = $("endpoint").value.trim().replace(/\/+$/, "") || DEFAULT_ENDPOINT;
  const system = persona.system_prompt + "\n\nSELF (measured by bankML just now):\n" + await selfText(ep);
  const out = bubble("assistant", "…");
  const started = waiting(out, "your bankML is reading the prompt (the first answer of a session reads the whole persona; it can take a few minutes on a busy CPU)");
  let text = "", receipt = null;
  const r = await fetch(ep + "/v1/chat/completions", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "system", content: system }, ...history.slice(-12), { role: "user", content: message }],
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
      if (piece) { started(); text += piece; out.textContent = text; }
    }
  }
  started();
  const ok = receipt && (await sha256hex(text)) === receipt.response_sha256;
  note(out, receipt
    ? `${ok ? "✓" : "✗"} receipt — bankML ${receipt.bankml} · model sha256 ${String(receipt.model_sha256 || "").slice(0, 12)}… · answer sha256 ${ok ? "matches the text received" : "does NOT match the text received"}`
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
  const system = persona.system_prompt + `\n\nWHERE THIS REPLY COMES FROM: you speak for bankML — its design, its rule, its voice — but this particular reply is generated by ${model} through a Hugging Face inference provider, not by bankML's engine: there is no bankML arithmetic, no receipt and no SELF block behind it. Answer as bankML would about what bankML is and does; when asked about your speed, use, receipt, verification or what is running right now, say plainly that this reply comes from ${model}, not from bankML, and that the visitor's own bankML gives verified answers.`;
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
  $("localrow").hidden = m !== "local";
  $("hfrow").hidden = m !== "hf";
  $("modenote").textContent = m === "local"
    ? "bankML's own arithmetic on your machine: verified model, receipt checked here, nothing sent anywhere else."
    : "Not bankML: a provider-hosted model speaks with bankML's persona, without its arithmetic or a receipt.";
}
document.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", renderMode));
$("connect").addEventListener("click", connect);
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
  bubble("user", message);
  try {
    await (mode() === "local" ? askLocal(message) : askProvider(message));
  } catch (e) {
    const local = mode() === "local";
    $("askstatus").textContent = (local ? "your bankML did not answer: " : "the provider did not answer: ") + String(e.message || e).slice(0, 300)
      + (local ? " — is bankml serve running with --allow-origin " + location.origin + " ?" : "");
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
  await loadPersona();
  if (persona) $("mantra").textContent = "“" + persona.mantra + "”";
  try { const res = await oauthHandleRedirectIfPresent(); if (res) writeSession(res); } catch (e) { $("who").textContent = "sign-in failed: " + String(e).slice(0, 120); }
  oauth = readSession();
  renderAuth();
})();
