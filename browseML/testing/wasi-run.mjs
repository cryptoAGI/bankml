// SPDX-License-Identifier: MIT OR Apache-2.0
// cargo's runner for the engine's unit tests on wasm32-wasip1 (`tools/browseml.sh test`): Node's WASI, the test
// binary's own arguments. Any `env` import (the GPU loader's dlopen/dlsym, linked by the tests but never reached by
// the kernels) traps if it is ever called.
import { WASI } from "node:wasi";
import fs from "node:fs";

const [file, ...args] = process.argv.slice(2);
const wasi = new WASI({ version: "preview1", args: [file, ...args], env: process.env, preopens: { "/": "/" }, returnOnExit: true });
const mod = await WebAssembly.compile(fs.readFileSync(file));
const env = {};
for (const i of WebAssembly.Module.imports(mod)) if (i.module === "env") env[i.name] = () => { throw new Error(`env.${i.name} called`); };
process.exitCode = wasi.start(await WebAssembly.instantiate(mod, { ...wasi.getImportObject(), env }));
