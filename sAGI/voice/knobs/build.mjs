// SPDX-License-Identifier: MIT OR Apache-2.0
// Build Savante's DreamKnob voice controls → ui/voice/knobs/savante_knobs.js (an IIFE, global SavanteKnobs).
// The recipe of DeltaVerse's playdocs rack: React, react-dom and dreamknob from the local DreamKnob workspace
// (DREAMKNOB_DEPS, default /home/hacker/dreamknob/node_modules), esbuild from its pnpm store. The artifact is
// committed so the UI needs no Node at run time.
import { pathToFileURL, fileURLToPath } from 'node:url'
import { existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
const HERE = dirname(fileURLToPath(import.meta.url))
const DEPS = [process.env.DREAMKNOB_DEPS, join(HERE, 'node_modules'), '/home/hacker/dreamknob/node_modules']
  .filter(Boolean).find((d) => existsSync(join(d, 'dreamknob')) && existsSync(join(d, 'react')))
if (!DEPS) { console.error('no node_modules with react and dreamknob; set DREAMKNOB_DEPS'); process.exit(1) }
const ES = [process.env.ESBUILD_DIR, join(DEPS, '.pnpm', 'esbuild@0.21.5', 'node_modules', 'esbuild'), join(DEPS, 'esbuild')]
  .filter(Boolean).find((p) => existsSync(join(p, 'lib', 'main.js')))
if (!ES) { console.error('no esbuild; set ESBUILD_DIR'); process.exit(1) }
const { build } = await import(pathToFileURL(join(ES, 'lib', 'main.js')).href)
await build({
  entryPoints: [join(HERE, 'src', 'knobs.jsx')], bundle: true, format: 'iife', globalName: 'SavanteKnobs',
  outfile: join(HERE, 'savante_knobs.js'), minify: true, target: ['es2019'], jsx: 'automatic',
  nodePaths: [DEPS], define: { 'process.env.NODE_ENV': '"production"' }, legalComments: 'none',
  banner: { js: '/* Savante voice controls — React 18, react-dom and dreamknob 1.1.4, bundled by ui/voice/knobs/build.mjs. Do not edit. */' },
})
console.log('built', join(HERE, 'savante_knobs.js'), 'from', DEPS)
