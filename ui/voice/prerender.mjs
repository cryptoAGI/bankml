// SPDX-License-Identifier: MIT OR Apache-2.0
// Pre-render statements in a house eSpeak NG voice, on this computer, with the same WASM build the DeltaVerse
// voice worker runs in browsers (vendor/espeak-ng/0.3.5-en) and the cast's own voice files installed BEFORE the
// first synthesis (eSpeak caches its voice list at first use; a variant installed later is silently ignored).
//
//   stdin:  {"espeak": <dir of espeak-ng.js>, "voices": <mindx-voices dir>, "voice": "en-gb-x-rp+jaimla",
//            "wpm": 168, "out": <dir>, "items": [{"key": "...", "say": "..."}]}
//   stdout: {"engine": "espeak-ng-wasm 0.3.5", "rate": 22050, "items": [{"key", "samples", "seconds", "file"}]}
import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'

const job = JSON.parse(readFileSync(0, 'utf8'))
const { default: Factory } = await import(join(job.espeak, 'espeak-ng.js'))
const M = await Factory({ locateFile: (f) => join(job.espeak, f) })

// install every house voice file (bases and variants) under /usr/share/espeak-ng-data/voices/
const ROOT = '/usr/share/espeak-ng-data/'
const walk = (dir, rel = '') => {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name), r = rel ? rel + '/' + name : name
    if (statSync(p).isDirectory()) walk(p, r)
    else if (name !== 'index.html' && name !== 'index.json' && name !== 'SOURCE.md') {
      const full = ROOT + r
      const parts = full.split('/').filter(Boolean).slice(0, -1)
      let at = ''
      for (const d of parts) { at += '/' + d; try { M.FS.mkdir(at) } catch (e) { /* exists */ } }
      M.FS.writeFile(full, readFileSync(p, 'utf8'))
    }
  }
}
walk(join(job.voices, 'voices'), 'voices')

const t = new M.eSpeakNGWorker()
const rate = t.get_samplerate()
t.set_voice(job.voice)
t.set_rate(Math.max(80, Math.min(450, Math.round(job.wpm))))

function wav(int16) {
  const b = Buffer.alloc(44 + int16.length * 2)
  b.write('RIFF', 0); b.writeUInt32LE(36 + int16.length * 2, 4); b.write('WAVE', 8); b.write('fmt ', 12)
  b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22); b.writeUInt32LE(rate, 24)
  b.writeUInt32LE(rate * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34); b.write('data', 36)
  b.writeUInt32LE(int16.length * 2, 40)
  for (let i = 0; i < int16.length; i++) b.writeInt16LE(int16[i], 44 + 2 * i)
  return b
}

const out = []
for (const it of job.items) {
  const chunks = []
  let n = 0
  t.synthesize(String(it.say), (audio) => { if (audio && audio.length) { chunks.push(Int16Array.from(audio)); n += audio.length } })
  const all = new Int16Array(n)
  let o = 0
  for (const c of chunks) { all.set(c, o); o += c.length }
  const file = join(job.out, it.key + '.wav')
  writeFileSync(file, wav(all))
  out.push({ key: it.key, samples: n, seconds: n / rate, file })
}
process.stdout.write(JSON.stringify({ engine: 'espeak-ng-wasm 0.3.5 (echogarden)', rate, voice: job.voice, wpm: job.wpm, items: out }))
