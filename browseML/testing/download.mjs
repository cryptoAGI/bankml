// SPDX-License-Identifier: MIT OR Apache-2.0
// browseML's download in pieces (hf/space/browseml.js download()): against a local server that cuts responses
// mid-body, as Brave on Linux saw huggingface.co's do, the file must arrive whole and byte-identical, each cut piece
// asked for again from the byte where it was cut; a server that ignores Range is read as one stream; a 4xx stops at once.
//   node browseML/testing/download.mjs
import http from "node:http";
import crypto from "node:crypto";
import { download } from "../../hf/space/browseml.js";

const SIZE = 5 * 1024 * 1024 + 12345, PIECE = 1024 * 1024;
const file = crypto.randomBytes(SIZE);
let cuts = 0, requests = [];
const server = http.createServer((req, res) => {
  const m = /bytes=(\d+)-(\d+)/.exec(req.headers.range || "");
  requests.push(req.url + " " + (req.headers.range || "-"));
  if (req.url === "/missing") { res.writeHead(404); return res.end(); }
  if (!m || req.url === "/norange") { res.writeHead(200, { "Content-Length": SIZE }); return res.end(file); }
  const a = +m[1], b = Math.min(+m[2], SIZE - 1);
  res.writeHead(206, { "Content-Length": b - a + 1, "Content-Range": `bytes ${a}-${b}/${SIZE}` });
  // every third request is cut halfway through its body
  if (req.url === "/cut" && requests.length % 3 === 0) {
    cuts++;
    res.write(file.subarray(a, a + Math.floor((b - a + 1) / 2)));
    return setTimeout(() => res.socket.destroy(), 20);
  }
  res.end(file.subarray(a, b + 1));
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
const base = `http://127.0.0.1:${server.address().port}`;
let failed = 0;
const check = (name, ok, more = "") => { console.log(`${ok ? "ok  " : "FAIL"} ${name}${more ? " — " + more : ""}`); failed += !ok; };

let retries = 0, resumedAt = [];
const got = await download(base + "/cut", SIZE, (n, _t, phase) => { if (phase === "retry") { retries++; resumedAt.push(n); } }, { piece: PIECE, pauseMs: 1 });
check("cut responses: the file arrives whole", Buffer.compare(Buffer.from(got), file) === 0, `${cuts} cuts, ${retries} retries`);
check("each cut is asked for again from the byte where it was cut", resumedAt.every((n) => n % PIECE !== 0) && retries === cuts, resumedAt.join(", "));

requests = [];
const whole = await download(base + "/norange", SIZE, () => {}, { piece: PIECE, pauseMs: 1 });
check("a server that ignores Range: one stream", Buffer.compare(Buffer.from(whole), file) === 0 && requests.length === 1);

requests = [];
let err = "";
try { await download(base + "/missing", SIZE, () => {}, { piece: PIECE, pauseMs: 1 }); } catch (e) { err = String(e.message); }
check("a 404 stops at once (no retries)", /HTTP 404/.test(err) && requests.length === 1, err);

server.close();
process.exit(failed ? 1 : 0);
