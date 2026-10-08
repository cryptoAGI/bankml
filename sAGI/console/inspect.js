// SPDX-License-Identifier: MIT OR Apache-2.0
// The Diagnostics Inspector: choose a component, an engine check or a trace span, and it says what that is, what its
// level means, what was measured (and the raw record), and — when there is something to do — how to deal with it:
// the commands (copyable), the tab or document to look at, a refresh. Everything rendered with textContent.
"use strict";
(function () {
  const REPO = "https://github.com/cryptoAGI/bankml/blob/main/";
  const LEVEL = {
    ok: ["ready", "Measured just now and as expected. Nothing to do."],
    info: ["idle", "Not running, or not configured. It is optional: start it only if you use it."],
    warn: ["to check", "It answered, but something is off or slow. Read what was seen; it may be passing (a busy machine)."],
    bad: ["failing", "It did not answer or did not pass. The steps below bring it back."],
  };
  const START = { cmd: "./install.sh restart", note: "a fresh bankml serve on the model it served last, and the UIs that were running" };
  const MODEL = { cmd: "./install.sh model", note: "the verified model, imported if absent, and bankml serve on it" };
  // per component (as /api/diagnostics names them): what it is, and what to do when it is not ready
  const COMPONENTS = {
    "bankml serve": { what: "The verified engine: it hashes the model against its pin before it listens, answers every page, and puts a receipt on every answer.",
      fix: [MODEL, START, { cmd: "curl -s 127.0.0.1:18093/bankml", note: "is it answering, and with which model" },
            { cmd: "tail -n 40 ~/.local/share/bankml/savante/carrier.log", note: "why it stopped, in its own words" }, { tab: "engine" }, { tab: "logs" }],
      doc: "docs/modules/serve.md" },
    console: { what: "This page: bankML as itself, its receipts, logs, diagnostics and the Advanced memory.", fix: [START], doc: "docs/modules/console.md" },
    Savante: { what: "The chat interface, Savante's persona over bankML (port 7873).", fix: [{ cmd: "./install.sh start", note: "starts Savante and this console if they are not running" }], doc: "docs/usage.md" },
    view: { what: "The read-only page for the local network (port 7874). Optional.", fix: [{ cmd: "./install.sh start --view", note: "start it beside the others" }], doc: "docs/usage.md" },
    models: { what: "The importer: pinned models, the carrier, the resources it starts with.", fix: [{ cmd: "python3 sAGI/models.py list", note: "what is pinned here" }, MODEL], doc: "docs/install.md" },
    personas: { what: "Who speaks: each .persona file, checked against its doctrine root.", fix: [{ cmd: "ls sAGI/personas/", note: "the persona files" }], doc: "docs/modules/console.md" },
    agents: { what: "Custom agents derived from the Savante template (Savante's Agents tab).", fix: [], doc: "docs/usage.md" },
    thot: { what: "THOT manifests: the dataset bundle an iNFT points to. Idle until an agent is bound.", fix: [], doc: "docs/usage.md" },
    chain: { what: "iNFT minting and loading, on a local devnet only. Idle without anvil.", fix: [], doc: "docs/usage.md" },
    connectors: { what: "PostgreSQL for .history and THOT. Idle when no database is configured.", fix: [], doc: "docs/usage.md" },
    embed: { what: "bge-m3 meaning search through the local Ollama; it loads only with 1.3 GB free.",
      fix: [{ cmd: "ollama pull bge-m3", note: "fetch the embedding model" }, { cmd: "free -m", note: "is there 1.3 GB free" }], doc: "docs/embedding.md" },
    voice: { what: "Savante's voice: Piper, one render at a time on this machine.", fix: [{ cmd: "./install.sh voice", note: "install Piper and the voice" }], doc: "docs/playback.md" },
    diagnostics: { what: "This instrumentation: every answer traced span by span.", fix: [], doc: "docs/modules/console.md" },
  };
  const CHECKS = {
    "bankml serve answers": { what: "A GET of /bankml, timed. Under 500 ms is ready.", fix: [MODEL, START, { tab: "engine" }],
      warn: "Slow, not down: the machine is busy (a long answer, a gate, a render). The diag button in T mode shows CPU and memory." },
    "the model is verified": { what: "The guard's verdict on the model file: play means its sha256 matches its pin.",
      fix: [{ cmd: "python3 sAGI/models.py list", note: "the pins and the files" }, { cmd: "python3 sAGI/models.py use FILE", note: "serve a pinned file" }], doc: "docs/oracles.md" },
    "the engine": { what: "Which bankML, which engine (native or llama-server) and which architecture is answering.", fix: [] },
    metrics: { what: "/bankml/metrics: time to first token and tokens per second of each answer.", fix: [START],
      warn: "The endpoint did not answer: an older serve. Restart it on this build." },
    usage: { what: "/bankml/usage: the engine's CPU, memory and GPU.", fix: [START] },
    "memory available": { what: "Memory the machine still has. Under 1 GB is tight, under 0.5 GB is failing.",
      fix: [{ cmd: "free -m", note: "what holds the memory" }, { tab: "admin", note: "lower the RAM budget, or choose a smaller model on Savante's Models tab" }] },
    CPU: { what: "The engine's share of one core, now.", fix: [] },
    "the engine log": { what: "The newest error line in the engine's log, if any.", fix: [{ tab: "logs" }, { cmd: "tail -n 80 ~/.local/share/bankml/savante/carrier.log" }] },
    "console spans": { what: "Whether any of this console's own traced steps failed.", fix: [{ note: "Choose the failing span in Traces to see its error." }] },
  };
  const SPANS = {
    ask: "One question, from the press of Ask to the receipt check.",
    self_block: "Measuring SELF: the engine's usage and metrics, read just before the question.",
    memory: "Adding the collection and this window's .memory to the persona.",
    recall: "Searching .history for earlier exchanges that match the question.",
    "engine.stream": "The engine reading the prompt and writing the answer; its 'first piece' event is the time to first token.",
    metrics: "Reading the engine's record of this answer.",
    "log.write": "Writing the exchange to the console's .history.",
    "receipt.verify": "Checking the answer's sha256 against the receipt the engine signed off.",
    sysdiag: "Reading CPU, memory, disk and GPU from /proc and /sys.",
    apply: "Restarting the engine with new resources (Admin → Apply).",
  };

  const $ = (id) => document.getElementById(id);
  const mk = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  const section = (h) => { const s = mk("section", "insp-s"); s.append(mk("h4", null, h)); return s; };

  function actions(list) {
    const ul = mk("ul", "insp-do");
    for (const a of list) {
      const li = mk("li");
      if (a.cmd) {
        const code = mk("code", null, a.cmd), b = mk("button", "ghost insp-copy", "copy");
        b.type = "button"; b.title = "copy the command";
        b.addEventListener("click", async () => { try { await navigator.clipboard.writeText(a.cmd); b.textContent = "copied"; } catch (e) { b.textContent = "select it"; } setTimeout(() => (b.textContent = "copy"), 1500); });
        li.append(code, b);
        if (a.note) li.append(mk("span", "insp-note", a.note));
      } else if (a.tab) {
        const b = mk("button", "ghost", `Open the ${a.tab[0].toUpperCase() + a.tab.slice(1)} tab`);
        b.type = "button"; b.addEventListener("click", () => window.show && window.show(a.tab));
        li.append(b);
        if (a.note) li.append(mk("span", "insp-note", a.note));
      } else if (a.note) li.append(mk("span", "insp-note", a.note));
      ul.append(li);
    }
    return ul;
  }

  function render(sel) {
    const body = $("inspector"); if (!body) return;
    body.textContent = "";
    const { kind, data } = sel;
    const level = data.level || (data.error ? "bad" : "ok");
    const name = kind === "component" ? data.component : kind === "check" ? data.check : data.name;
    const kb = kind === "component" ? COMPONENTS[name] || {} : kind === "check" ? CHECKS[name] || {} : {};
    const head = mk("header", "insp-head");
    head.append(mk("span", "chip " + level, (LEVEL[level] || [level])[0]), mk("h3", null, name), mk("span", "insp-kind", kind));
    body.append(head);
    const what = section("What it is");
    what.append(mk("p", null, kind === "span" ? SPANS[name] || "A traced step of the console." : kb.what || "—"));
    if (kind === "component" && data.file) {
      const a = mk("a", null, data.file + " ↗"); a.href = REPO + data.file; a.target = "_blank"; a.rel = "noreferrer";
      what.append(a);
    }
    body.append(what);
    const seen = section("What was measured");
    if (kind === "span") {
      const total = sel.total || data.duration_ms;
      seen.append(mk("p", null, data.open ? "still open" : `${data.duration_ms} ms` + (total && data.duration_ms !== total ? ` — ${Math.round((100 * data.duration_ms) / total)} % of the answer` : "")));
      for (const ev of data.events || []) seen.append(mk("p", "insp-mono", `${ev.name} at ${ev.at_ms} ms`));
      const tags = Object.entries(data.tags || {});
      if (tags.length) seen.append(mk("p", "insp-mono", tags.map(([k, v]) => `${k} ${v}`).join(" · ")));
      if (data.error) seen.append(mk("p", "insp-err", "✗ " + data.error));
    } else {
      seen.append(mk("p", "insp-mono", data.seen || "—"));
      if (data.ms !== undefined) seen.append(mk("p", "insp-note", `asked in ${data.ms} ms`));
    }
    const raw = mk("details", "insp-raw"); raw.append(mk("summary", null, "the record, as received"), mk("pre", null, JSON.stringify(data, (k, v) => (k === "children" ? (v || []).length + " children" : v), 2)));
    seen.append(raw);
    body.append(seen);
    const mean = section("What the level means");
    mean.append(mk("p", null, (LEVEL[level] || ["", "—"])[1]));
    if (level === "warn" && kb.warn) mean.append(mk("p", null, kb.warn));
    body.append(mean);
    const todo = level === "ok" ? [] : kind === "span" ? (data.error ? [{ note: "The error is the step's own, as it was raised. Ask again; if it repeats, the Logs tab has the engine's side." }, { tab: "logs" }] : []) : kb.fix || [];
    if (kind === "span" && name === "engine.stream" && data.duration_ms > 30000)
      todo.push({ note: "Long: most of it is the prompt being read before the first token. A shorter conversation, the persona kept warm, or a smaller model shortens it." });
    if (todo.length || level !== "ok") {
      const s = section(level === "ok" ? "Nothing to do" : "What to do");
      s.append(todo.length ? actions(todo) : mk("p", "insp-note", "Nothing to run: it is optional, or it needs no action at this level."));
      const r = mk("button", "ghost", "Refresh diagnostics"); r.type = "button";
      r.addEventListener("click", () => $("diag-refresh") && $("diag-refresh").click());
      s.append(r);
      body.append(s);
    }
    if (kb.doc) {
      const d = section("Read more"); const a = mk("a", null, kb.doc + " ↗"); a.href = REPO + kb.doc; a.target = "_blank"; a.rel = "noreferrer";
      d.append(a); body.append(d);
    }
  }

  function choose(node) {
    document.querySelectorAll("#diagnostics .is-chosen").forEach((x) => x.classList.remove("is-chosen"));
    node.classList.add("is-chosen");
    render(node.__diag);
    const w = $("inspector") && $("inspector").closest(".window");
    if (w) { w.classList.remove("collapsed"); w.classList.add("is-flash"); setTimeout(() => w.classList.remove("is-flash"), 700); }
  }
  document.addEventListener("click", (e) => {
    const n = e.target.closest("#diagnostics .comp, #diagnostics #checks li, #diagnostics .span > .row");
    if (n && n.__diag && !e.target.closest("button, a")) choose(n);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    const n = e.target.closest && e.target.closest("#diagnostics .comp, #diagnostics #checks li, #diagnostics .span > .row");
    if (n && n.__diag) { e.preventDefault(); choose(n); }
  });
})();
