// SPDX-License-Identifier: MIT OR Apache-2.0
// One thread of bankML's engine in the browser (browseml-mt.wasm). The engine worker creates these before the engine
// starts — a thread that is waiting cannot start a worker — and hands each one a thread when the engine's pool
// asks (wasi.thread-spawn). Each instantiates the same module on the same shared memory, then runs
// wasi_thread_start(tid, start_arg), which runs the pool's worker loop until the engine is closed.
//   engine → thread  {type: "init", module, memory}  then  {type: "start", tid, arg}
//   thread → engine  {type: "ready"} · {type: "log", level, text} · {type: "exit", tid, error?}
import { wasi } from "./browseml-wasi.js";

let instance = null;
onmessage = async (e) => {
  const m = e.data;
  if (m.type === "init") {
    // no files here: the threads only compute; standard error goes to the page's log
    const w = wasi({ files: {}, write: (fd, s) => postMessage({ type: "log", level: 1, text: s.trimEnd() }) });
    instance = await WebAssembly.instantiate(m.module, {
      ...w.imports,
      env: { memory: m.memory },
      wasi: { "thread-spawn": () => -1 },  // the engine's threads do not start threads of their own
      browseml: { piece: () => 1, log: (level, p, n) => postMessage({ type: "log", level, text: "(thread) " + new TextDecoder().decode(new Uint8Array(m.memory.buffer).slice(p, p + n)) }) },
    });
    w.bind(m.memory);
    postMessage({ type: "ready" });
  } else if (m.type === "start") {
    try {
      instance.exports.wasi_thread_start(m.tid, m.arg);
      postMessage({ type: "exit", tid: m.tid });
    } catch (err) {
      postMessage({ type: "exit", tid: m.tid, error: String(err && err.message || err) });
    }
  }
};
