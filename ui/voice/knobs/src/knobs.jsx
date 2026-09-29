// SPDX-License-Identifier: MIT OR Apache-2.0
// Savante's voice controls — DreamKnob in the Savante UI's settings panel. One global, SavanteKnobs.mount(el).
// SPEED: playback rate with the pitch preserved (the browser's own time-stretch), snap points as the playdocs rack.
// FM: frequency modulation of the voice — an LFO sweeping a short delay line (vibrato), RATE in Hz, DEPTH in %.
// State lives in window.bkVoice (and this browser's localStorage); a 'bk-voice' event tells the player.
import React from 'react'
import { createRoot } from 'react-dom/client'
import { DreamknobProvider, VintageKnob, NeonKnob, SegmentDisplay } from 'dreamknob'

const KEY = 'bankml-savante-voice-v1'
const DEFAULTS = { speed: 1, fmRate: 5, fmDepth: 0, gain: 0, volume: 80 }
const SPEEDS = [0.666, 0.8, 1, 1.25, 1.5, 2]

function load() {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) || '{}') } } catch (e) { return { ...DEFAULTS } }
}

const same = (a, b) => a && b && ['speed', 'fmRate', 'fmDepth', 'gain', 'volume'].every((k) => a[k] === b[k])

function Panel({ size = 64, layout = 'row' }) {
  const [v, setV] = React.useState(() => ({ ...DEFAULTS, ...(window.bkVoice || load()) }))
  React.useEffect(() => {  // every panel (settings, card) shares one state: follow the others
    const on = (e) => { if (!same(e.detail, v)) setV(e.detail) }
    window.addEventListener('bk-voice', on)
    return () => window.removeEventListener('bk-voice', on)
  }, [v])
  React.useEffect(() => {
    if (same(window.bkVoice, v)) return
    window.bkVoice = v
    try { localStorage.setItem(KEY, JSON.stringify(v)) } catch (e) {}
    window.dispatchEvent(new CustomEvent('bk-voice', { detail: v }))
  }, [v])
  const set = (k) => (x) => setV((o) => ({ ...o, [k]: x }))
  const cell = { display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, minWidth: 0 }
  return (
    <DreamknobProvider base="dark">
      <div style={{ display: 'grid', gridTemplateColumns: layout === 'column' ? '1fr' : 'repeat(5, 1fr)', gap: layout === 'column' ? 10 : 6, alignItems: 'end', justifyItems: 'center' }}>
        <div style={cell} title={'SPEED ' + v.speed.toFixed(3) + '× — pitch kept; snap points at ' + SPEEDS.join(', ')}>
          <VintageKnob value={v.speed} min={0.5} max={2.5} step={0.001} detents={SPEEDS} detentSize={0.012}
            size={size} label="SPEED" color="#2dd4bf" aria-label="speaking speed" onChange={set('speed')} />
          <SegmentDisplay value={v.speed.toFixed(2)} height={14} color="#2dd4bf" />
        </div>
        <div style={cell} title="FM RATE — how fast the voice's frequency is modulated">
          <NeonKnob value={v.fmRate} min={0} max={12} step={0.1} size={size} label="FM RATE" unit="Hz" color="#d9a23a"
            aria-label="frequency modulation rate" onChange={set('fmRate')} />
        </div>
        <div style={cell} title="FM DEPTH — how far the frequency swings; 0 is the voice as rendered">
          <NeonKnob value={v.fmDepth} min={0} max={100} step={1} size={size} label="FM DEPTH" unit="%" color="#d9a23a"
            aria-label="frequency modulation depth" onChange={set('fmDepth')} />
        </div>
        <div style={cell} title="GAIN — her level into the chain, in dB (0 = as rendered)">
          <NeonKnob value={v.gain} min={-12} max={12} step={0.5} size={size} label="GAIN" unit="dB" color="#5eead4" arcFrom="center"
            aria-label="gain" onChange={set('gain')} />
        </div>
        <div style={cell} title="VOLUME — the master output">
          <NeonKnob value={v.volume} min={0} max={100} step={1} size={size} label="VOLUME" unit="%" color="#5eead4"
            aria-label="volume" onChange={set('volume')} />
        </div>
      </div>
    </DreamknobProvider>
  )
}

// mount (or re-lay-out) the knobs in el: size in px, layout 'row' (five across) or 'column' (a vertical stack)
export function mount(el, size, layout) {
  if (!el) return
  if (!window.bkVoice) window.bkVoice = load()
  if (!el.__bkRoot) { el.__bkRoot = createRoot(el); el.dataset.bkKnobs = '1' }
  el.__bkRoot.render(<Panel size={size || 64} layout={layout || 'row'} />)
}
export const defaults = DEFAULTS
