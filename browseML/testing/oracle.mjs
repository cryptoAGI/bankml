// SPDX-License-Identifier: MIT OR Apache-2.0
// browseML's oracle: the WebAssembly engine answers the requests llama-server b11192 answered, and every text,
// finish reason and token count must be identical to its record (the same records `serve --native` is held to).
//   node browseML/testing/oracle.mjs browseml.wasm MODELS_DIR STEM FORK_JSON [first N turns]
import fs from "node:fs";
import path from "node:path";
import { wasi as makeWasi } from "../../hf/space/browseml-wasi.js";

const [wasmPath, models, stem, forkPath, limit] = process.argv.slice(2);
// the page's own WASI, given the model file as the page gives it the download
const w = makeWasi({ files: { [`${stem}.gguf`]: new Uint8Array(fs.readFileSync(path.join(models, `${stem}.gguf`))) },
                     env: { BANKML_CACHE_RAM: "0" }, write: (fd, s) => process.stderr.write(s) });
let mem, onPiece = () => 1, log = [];
const text = (p, n) => Buffer.from(mem().buffer, p, n).toString("utf8");
const { instance } = await WebAssembly.instantiate(fs.readFileSync(wasmPath), {
  ...w.imports,
  browseml: { piece: (p, n) => onPiece(text(p, n)), log: (lv, p, n) => log.push(text(p, n)) },
});
const x = instance.exports; mem = () => x.memory;
w.bind(x.memory);
const put = (s) => { const b = Buffer.from(s, "utf8"); const p = x.browseml_alloc(b.length); new Uint8Array(x.memory.buffer, p, b.length).set(b); return [p, b.length]; };
const out = () => text(x.browseml_out_ptr(), x.browseml_out_len());

let t0 = performance.now();
console.log(`instantiated; opening (the guard, then the sha256 of the whole file, then the forward pass)…`);
const rc = x.browseml_open(...put(`/models/${stem}.gguf`), ...put(fs.readFileSync(forkPath, "utf8")), 4096);
console.log(`open ${rc} in ${((performance.now() - t0) / 1000).toFixed(1)} s: ${out()}`);
if (rc !== 0) process.exit(1);
w.release(`${stem}.gguf`);
console.log(`memory after open: ${(x.memory.buffer.byteLength / 1e6).toFixed(0)} MB in the module`);
const recs = fs.readFileSync(path.join(models, `oracle-forward/serve-${stem}.jsonl`), "utf8").trim().split("\n").map(JSON.parse).slice(0, +limit || undefined);
let same = 0;
for (const r of recs) {
  let streamed = "";
  let n = 0, tFirst = null;
  onPiece = (s) => { streamed += s; n++; if (tFirst === null) { tFirst = performance.now(); console.log(`  first piece after ${((tFirst - t0) / 1000).toFixed(1)} s (prompt read)`); }
                     else if (n % 16 === 0) console.log(`  ${n} pieces, ${(((n - 1) * 1000) / (performance.now() - tFirst)).toFixed(2)} pieces/s`); return 1; };
  t0 = performance.now();
  const c = x.browseml_chat(...put(JSON.stringify(r.request)));
  const ms = performance.now() - t0, j = JSON.parse(out());
  if (c !== 0) { console.log(`turn ${r.conversation}.${r.turn}: error ${c} ${out()}`); continue; }
  const ch = j.choices[0], got = ch.message.content;
  const ok = got === r.text && streamed === got && ch.finish_reason === r.finish_reason
    && (r.usage ? j.usage.completion_tokens === r.usage.completion_tokens && j.usage.prompt_tokens === r.usage.prompt_tokens : true);
  same += ok;
  console.log(`turn ${r.conversation}.${r.turn}: ${ok ? "identical" : "DIFFERENT"} — ${j.usage.completion_tokens} tokens in ${(ms / 1000).toFixed(1)} s ` +
    `(${(j.usage.completion_tokens / (ms / 1000)).toFixed(2)} tok/s); receipt ${j.bankml_receipt ? "response_sha256 " + j.bankml_receipt.response_sha256.slice(0, 16) : "MISSING"}`);
  if (!ok) console.log(`  want: ${JSON.stringify(r.text).slice(0, 300)}\n  got:  ${JSON.stringify(got).slice(0, 300)}\n  rec keys: ${Object.keys(r)}`);
}
console.log(`browseML oracle (${stem}): ${same} of ${recs.length} answers identical to llama-server b11192's record`);
console.log(log.join("\n"));
process.exit(same === recs.length ? 0 : 1);
