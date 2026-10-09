// SPDX-License-Identifier: MIT OR Apache-2.0
// browseML's worker: bankML's engine (browseml.wasm) runs here, off the page's thread, so the page stays responsive
// while the visitor's CPU reads the prompt and writes the answer. Messages:
//   page → worker  {type: "open", name, bytes: ArrayBuffer (transferred), fork}  |  {type: "chat", request}
//   worker → page  {type: "log", level, text} · {type: "opened", rc, out} · {type: "piece", text} · {type: "done", rc, out}
import { wasi } from "./browseml-wasi.js";

let x = null, w = null;
const dec = new TextDecoder(), enc = new TextEncoder();
const text = (p, n) => dec.decode(new Uint8Array(x.memory.buffer, p, n));
const put = (s) => { const b = enc.encode(s); const p = x.browseml_alloc(b.length); new Uint8Array(x.memory.buffer, p, b.length).set(b); return [p, b.length]; };
const out = () => text(x.browseml_out_ptr(), x.browseml_out_len());

async function open({ name, bytes, fork }) {
  w = wasi({ files: { [name]: new Uint8Array(bytes) }, env: { BANKML_CACHE_RAM: "0" },
             write: (fd, s) => postMessage({ type: "log", level: fd === 2 ? 1 : 2, text: s.trimEnd() }) });
  const { instance } = await WebAssembly.instantiateStreaming(fetch(new URL("./browseml.wasm", import.meta.url)), {
    ...w.imports,
    browseml: {
      piece: (p, n) => { postMessage({ type: "piece", text: text(p, n) }); return 1; },
      log: (level, p, n) => postMessage({ type: "log", level, text: text(p, n) }),
    },
  });
  x = instance.exports;
  w.bind(x.memory);
  const rc = x.browseml_open(...put(`/models/${name}`), ...put(fork), 4096);
  w.release(name);  // the engine holds the model in its own memory now; the downloaded copy can go
  postMessage({ type: "opened", rc, out: out() });
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "open") await open(m);
    else if (m.type === "chat") {
      if (!x) throw new Error("no model is open");
      const rc = x.browseml_chat(...put(JSON.stringify(m.request)));
      postMessage({ type: "done", rc, out: out() });
    }
  } catch (err) {
    // a trap (out of memory, a panic) ends this engine; the page says so and can load it again
    postMessage({ type: m.type === "open" ? "opened" : "done", rc: -99, out: String(err && err.message || err) });
    x = null;
  }
};
