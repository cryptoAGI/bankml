#!/usr/bin/env python3
"""bankml · view — the read-only page for anyone on the LAN who wants to watch the testing.

Why not Gradio: the Gradio installed here (3.37) has path-traversal bugs that let a client read files from the
host (e.g. CVE-2023-51449, fixed in 4.11). That is acceptable on loopback for the operator, not on a network.
This server is the Python standard library with a fixed set of GET routes, nothing else:

  /                  the page (HTML + CSS + JS inline; the JS renders every value with textContent)
  /api/state         JSON: the live testing log, its stage, the machine, the release records' names, CI,
                     bankml serve's verification, Savante's office and ledger check
  /api/result?name=  one release record — only a name that `testing/results/` actually lists
  /savante.png       the canon's named image (read-only)

No chat, no .history, no command execution, no file paths from the client. Refreshes every 2 s.

  python3 ui/view.py --host 0.0.0.0 --port 7874      # the LAN
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import savante as S  # noqa: E402  (stdlib-only at import; gradio is imported only by its build())

CANON = S.Canon(S.CANON)
KNOBS = Path(__file__).resolve().parent / "voice" / "knobs" / "savante_knobs.js"
import re  # noqa: E402
import speak  # noqa: E402


def listen() -> dict:
    """Savante's introduction and voice examples, as far as they are rendered (keys only; audio via /audio/)."""
    chs = speak.intro_chapters(S.CANON, CANON.persona, CANON.card)
    ex = [x for x in CANON.persona.get("voice_examples") or [] if isinstance(x, str)]
    out = []
    for title, sents in chs + [("Savante speaks — her voice examples", ex)]:
        items = speak.cached(sents)
        if all(items):
            out.append({"title": title, "items": [{"text": i["text"], "key": i["key"], "seconds": i["seconds"]} for i in items]})
        else:
            out.append({"title": title, "rendering": True, "count": len(sents)})
    return {"voice": speak.savante_voice(), "chapters": out}


def audio_file(key: str):
    """Only a 24-hex key the voice manifest lists; nothing else is reachable."""
    if not re.fullmatch(r"[0-9a-f]{24}", key or ""):
        return None
    try:
        man = json.loads((speak.VOICE_DIR / "savante" / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    f = speak.VOICE_DIR / "savante" / f"{key}.ogg"
    return f if key in man and f.is_file() else None


def state() -> dict:
    p, card = CANON.persona, CANON.card
    serve = S.serve_status()
    return {
        "live": {**S.live_state(), "tail": S.live_tail(80)},
        "machine": S.machine_state(),
        "results": S.results_list(),
        "ci": S.ci_runs(),
        "serve": {"ok": "error" not in serve, "error": serve.get("error"),
                  "model": Path(serve.get("model", "")).name, "sha256": (serve.get("verified") or {}).get("model_sha256"),
                  "bankml": (serve.get("verified") or {}).get("bankml"), "engine": serve.get("engine")},
        "private": {"history": S.commitment(S.HISTORY), "memory": S.commitment(S.MEMORY)},
        "listen": listen(),
        "savante": {"name": p.get("name"), "kind": p.get("kind"), "mantra": p.get("mantra"), "oath": p.get("oath"),
                    "status": (card.get("savante") or {}).get("status"), "type": card.get("type"),
                    "doctrine_root": (CANON.ledger.get("doctrine_root") or {}).get("value"),
                    "ledger": [{"entry": n, "path": r, "ok": ok, "detail": d} for n, r, ok, d in CANON.rows]},
    }


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>bankml · view</title>
<style>
:root{--bg:#0b0f14;--panel:rgba(148,163,184,.06);--panel2:rgba(148,163,184,.04);--line:#263241;--line2:#344457;--text:#e8eef5;--muted:#9fb0c3;
--accent:#39d3c7;--gold:#d9a23a;--ok:#56d364;--bad:#ff6b6b;--warn:#f5b73b;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
@media (prefers-color-scheme:light){:root{--bg:#f4f6f9;--panel:#ffffff;--panel2:#f8fafc;--line:#cfd8e3;--line2:#b6c3d1;--text:#0f1722;
--muted:#4a5a6c;--accent:#0f8f86;--gold:#9a6a0b;--ok:#1a7f37;--bad:#c62828;--warn:#9a6700}}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;gap:16px;align-items:center;padding:14px 20px;border-bottom:1px solid var(--line2);background:var(--panel)}
header img{width:44px;height:44px;border-radius:50%;border:2px solid var(--gold);object-fit:cover}
header h1{font-size:18px;margin:0;letter-spacing:.2px}header .sub{color:var(--muted);font-size:13px}
main{max-width:1320px;margin:0 auto;padding:18px 16px 40px;display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;box-shadow:0 1px 0 rgba(0,0,0,.25);
resize:both;overflow:auto;min-width:260px;min-height:120px;max-width:100%;display:flex;flex-direction:column}
.card>h2{margin:0;padding:10px 14px;font-size:13px;letter-spacing:.8px;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--line);
background:var(--panel2);border-radius:10px 10px 0 0;cursor:grab;user-select:none;display:flex;justify-content:space-between;position:sticky;top:0;z-index:1}
.card>h2::after{content:'⠿ drag · ◢ resize';letter-spacing:0;text-transform:none;font-size:11px;opacity:.7}
.card.dragging{opacity:.45}.card.over{outline:2px dashed var(--accent);outline-offset:3px}
.body{padding:12px 14px;flex:1;min-height:0}.body pre{max-height:60vh}
.w8{width:calc(66.6% - 8px)}.w4{width:calc(33.3% - 8px)}.w6{width:calc(50% - 8px)}
@media (max-width:900px){.w8,.w4,.w6{width:100%}}
.tool{background:var(--panel2);color:var(--text);border:1px solid var(--line2);border-radius:6px;padding:5px 10px;font:inherit;font-size:12px;cursor:pointer;margin-left:auto}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600;border:1px solid currentColor}
.run{color:var(--ok)}.idle{color:var(--muted)}.bad{color:var(--bad)}.warn{color:var(--warn)}
pre{margin:0;font:12.5px/1.45 var(--mono);white-space:pre-wrap;word-break:break-word;background:var(--panel2);border:1px solid var(--line);
border-radius:8px;padding:10px;max-height:62vh;overflow:auto;color:var(--text)}
dl{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;margin:0}dt{color:var(--muted)}dd{margin:0;font-family:var(--mono);font-size:13px;word-break:break-all}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;background:var(--panel2)}tr:last-child td{border-bottom:0}
select{background:var(--panel2);color:var(--text);border:1px solid var(--line2);border-radius:6px;padding:6px 8px;font:inherit}
.mantra{color:var(--gold);font-style:italic}
.kn{max-width:520px;border:1px solid var(--line);border-radius:12px;background:var(--panel2);max-height:0;opacity:0;overflow:hidden;
padding:0 10px;margin:0;transform:translateY(-6px) scale(.97);transition:max-height .45s ease,opacity .35s ease,transform .45s cubic-bezier(.2,.9,.3,1.2),padding .3s,margin .3s}
.kn.on{max-height:150px;opacity:1;padding:8px 10px 4px;margin:0 0 10px;transform:none;box-shadow:0 0 20px rgba(57,211,199,.18)}
@media (prefers-reduced-motion:reduce){.kn{transition:none}}
.lbar{display:flex;gap:8px;margin:6px 0 10px}.lbar button,.lch button{cursor:pointer;border-radius:8px;border:1px solid var(--accent);background:transparent;
color:var(--accent);font:600 12px var(--mono);padding:5px 12px}.lbar button:hover,.lch button:hover{background:rgba(57,211,199,.12)}
.lch{border:1px solid var(--line);border-radius:10px;margin:6px 0}.lch summary{cursor:pointer;padding:7px 10px;display:flex;gap:10px;align-items:center}
.lch summary b{flex:1}.lch ol{margin:0;padding:0 10px 8px 34px}.lch li{padding:3px 4px;border-radius:6px;cursor:pointer;font-size:13.5px;line-height:1.5}
.lch li:hover{background:rgba(57,211,199,.06)}.lch li.now{background:rgba(217,162,58,.16);box-shadow:inset 3px 0 0 var(--gold)}
.lwait{color:var(--muted);font-size:13px;padding:6px 10px}.foot{color:var(--muted);font-size:12px;padding:0 20px 24px;text-align:center}
</style></head><body>
<header><img id="av" alt="Savante" src="/savante.png"><div><h1>bankml · view — watching the testing live</h1>
<div class="sub">Read-only. A speed counts only if every oracle passed on the same code. Refreshes every 2 s · <span id="clock"></span></div></div>
<button class="tool" id="reset" title="put every panel back">reset layout</button></header>
<main>
<section class="card w8" id="p-live"><h2>Testing — live</h2><div class="body">
<p><span id="stage" class="pill idle">…</span> <strong id="title"></strong><br><span style="color:var(--muted)">step:</span> <span id="step"></span></p>
<pre id="log">loading…</pre></div></section>
<section class="card w4" id="p-mach"><h2>This laptop</h2><div class="body"><dl id="mach"></dl></div></section>
<section class="card w4" id="p-serve"><h2>bankml serve</h2><div class="body"><dl id="serve"></dl></div></section>
<section class="card w8" id="p-recs"><h2>Release records</h2><div class="body">
<select id="pick"></select><pre id="rec" style="margin-top:10px;max-height:48vh"></pre></div></section>
<section class="card w6" id="p-ci"><h2>CI — github.com/cryptoAGI/bankml</h2><div class="body"><table><thead><tr><th>run</th><th>result</th><th>when</th></tr></thead><tbody id="ci"></tbody></table></div></section>
<section class="card w8" id="p-listen"><h2>Listen to Savante</h2><div class="body">
<p style="margin-top:0;color:var(--muted)">New here? Savante reads herself to you — who she is, her oath, what she believes, how she
works, why she exists — verbatim from her canon, in her voice (<span id="lv"></span>).</p>
<div class="lbar"><button type="button" id="lall">▶ play the introduction</button><button type="button" id="lstop">■ stop</button></div>
<div id="lknobs" class="kn" aria-label="voice controls"></div>
<div id="lchaps"></div></div></section>
<section class="card w6" id="p-proof"><h2>Private data — commitments only</h2><div class="body">
<p style="margin-top:0;color:var(--muted)">.history and .memory stay on this laptop. What is shown is their commitment: anyone given one
exchange and its inclusion proof can check it against this root, without seeing the rest.</p><dl id="proof"></dl></div></section>
<section class="card w6" id="p-sav"><h2>Savante — office and iNFT ledger</h2><div class="body">
<p><strong id="sname"></strong> <span id="skind" style="color:var(--muted)"></span><br><span id="smantra" class="mantra"></span></p>
<dl id="scard"></dl><table style="margin-top:10px"><thead><tr><th>ledger entry</th><th>check</th></tr></thead><tbody id="ledger"></tbody></table></div></section>
</main>
<div class="foot">bankml · verified low-bit inference · cryptoAGI · this page serves no chat and no history</div>
<script src="/knobs.js"></script>
<script>
const $=id=>document.getElementById(id);
function dl(el,rows){el.replaceChildren();for(const [k,v,cls] of rows){const dt=document.createElement('dt');dt.textContent=k;
const dd=document.createElement('dd');dd.textContent=v==null?'—':String(v);if(cls)dd.className=cls;el.append(dt,dd)}}
function tr(tbody,rows){tbody.replaceChildren();for(const cells of rows){const r=document.createElement('tr');
for(const [v,cls] of cells){const td=document.createElement('td');td.textContent=v;if(cls)td.className=cls;r.append(td)}tbody.append(r)}}
let picked=null,known='';
async function rec(){if(!picked)return;const r=await fetch('/api/result?name='+encodeURIComponent(picked));$('rec').textContent=await r.text()}
async function tick(){$('clock').textContent=new Date().toLocaleTimeString();
 try{const s=await (await fetch('/api/state',{cache:'no-store'})).json();
 const L=s.live;$('stage').textContent=L.running?'running':'idle';$('stage').className='pill '+(L.running?'run':'idle');
 $('title').textContent=L.title;$('step').textContent=L.step;const lg=$('log'),atEnd=lg.scrollTop+lg.clientHeight>=lg.scrollHeight-8;
 lg.textContent=L.tail;if(atEnd)lg.scrollTop=lg.scrollHeight;
 const m=s.machine;dl($('mach'),m.error?[['error',m.error,'bad']]:[['cpu',m.cpu],['threads',m.threads],['load',m.load.join('  ')],
  ['RAM available',m.ram_available_gb+' of '+m.ram_total_gb+' GB'],['swap used',m.swap_used_mb+' MB'+(m.swap_full?' — full: measurements may be disturbed':''),m.swap_full?'warn':'']]);
 const v=s.serve;dl($('serve'),v.ok?[['status','verified — play','run'],['model',v.model],['sha256',v.sha256],['bankml',v.bankml],['engine',v.engine]]:[['status',v.error,'bad']]);
 const names=s.results.join(',');if(names!==known){known=names;const p=$('pick');p.replaceChildren();
  for(const n of s.results){const o=document.createElement('option');o.value=o.textContent=n;p.append(o)}if(!picked&&s.results.length){picked=s.results[0];rec()}}
 tr($('ci'),(s.ci.runs||[]).map(r=>[[r.title],[r.result,r.result==='success'?'run':(r.result==='failure'?'bad':'warn')],[r.when.replace('T',' ').replace('Z','')]]));
 if(s.listen)lrender(s.listen);
 const P=s.private;dl($('proof'),[['.history records',P.history.records],['.history Merkle root',P.history.merkle_root],['.history CIDv1',P.history.file_cid],
  ['.memory notes',P.memory.records],['.memory Merkle root',P.memory.merkle_root]]);
 const a=s.savante;$('sname').textContent=a.name||'';$('skind').textContent=a.kind?'— '+a.kind:'';$('smantra').textContent=a.mantra||'';
 dl($('scard'),[['card',a.type],['status',a.status+' (no mint here)'],['doctrine root',a.doctrine_root]]);
 tr($('ledger'),a.ledger.map(e=>[[e.entry+'  ('+e.path+')'],[(e.ok?'✓ ':'✗ ')+e.detail,e.ok?'run':'bad']]));
 }catch(e){$('stage').textContent='offline';$('stage').className='pill bad'}}
$('pick').addEventListener('change',e=>{picked=e.target.value;rec()});
// Listen to Savante: one audio element, a queue of clips, the line being read highlighted
const AU=new Audio();AU.preload='none';let LQ=[],lnow=null,lshown='';
// the card's voice controls: DreamKnob SPEED · FM RATE · FM DEPTH · GAIN · VOLUME, emerging when play is pressed
let AX=null;const VD={speed:1,fmRate:5,fmDepth:0,gain:0,volume:80};
function vset(){const x=Object.assign({},VD,window.bkVoice||{});AU.preservesPitch=true;AU.mozPreservesPitch=true;AU.playbackRate=+x.speed||1;
 if(!AX){AU.volume=Math.max(0,Math.min(1,x.volume/100));return}const t=AX.ctx.currentTime;
 AX.lfo.frequency.setTargetAtTime(Math.max(.01,+x.fmRate||0),t,.02);AX.depth.gain.setTargetAtTime(.0025*Math.max(0,Math.min(100,+x.fmDepth||0))/100,t,.02);
 AX.pre.gain.setTargetAtTime(Math.pow(10,Math.max(-12,Math.min(12,+x.gain||0))/20),t,.02);AX.vol.gain.setTargetAtTime(Math.max(0,Math.min(100,+x.volume))/100,t,.02)}
function vchain(){if(AX)return;try{const ctx=new(window.AudioContext||window.webkitAudioContext)(),src=ctx.createMediaElementSource(AU),
 pre=ctx.createGain(),delay=ctx.createDelay(.05),lfo=ctx.createOscillator(),depth=ctx.createGain(),vol=ctx.createGain();
 delay.delayTime.value=.006;lfo.connect(depth);depth.connect(delay.delayTime);lfo.start();src.connect(pre);pre.connect(delay);delay.connect(vol);vol.connect(ctx.destination);
 AX={ctx,lfo,depth,pre,vol}}catch(e){console.warn('voice chain',e)}}
function vemerge(){vchain();if(AX&&AX.ctx.state==='suspended')AX.ctx.resume();const el=$('lknobs');
 if(window.SavanteKnobs)SavanteKnobs.mount(el,58);el.classList.add('on');vset()}
window.addEventListener('bk-voice',vset);
function lmark(li){if(lnow)lnow.classList.remove('now');lnow=li;if(li){li.classList.add('now');const d=li.closest('details');if(d)d.open=true;li.scrollIntoView({block:'nearest',behavior:'smooth'})}}
function lnext(){const li=LQ.shift();if(!li){lmark(null);return}lmark(li);AU.src='/audio/'+li.dataset.k+'.ogg';AU.play().catch(()=>{})}
AU.addEventListener('ended',lnext);AU.addEventListener('play',vset);
function lplay(lis){AU.pause();vemerge();LQ=[...lis];lnext()}
$('lall').addEventListener('click',()=>lplay(document.querySelectorAll('#lchaps li')));
$('lstop').addEventListener('click',()=>{LQ=[];AU.pause();lmark(null)});
function lrender(L){const sig=JSON.stringify(L.chapters.map(c=>[c.title,!!c.rendering]));if(sig===lshown)return;lshown=sig;
 $('lv').textContent='house stand-in '+L.voice.voice+' at '+L.voice.wpm+' wpm, said sav-ont';const box=$('lchaps');box.replaceChildren();
 L.chapters.forEach((c,n)=>{if(c.rendering){const w=document.createElement('div');w.className='lwait';w.textContent=(n+1)+'. '+c.title+' — rendering ('+c.count+' sentences)';box.append(w);return}
  const d=document.createElement('details');d.className='lch';const s=document.createElement('summary');const b=document.createElement('b');b.textContent=(n+1)+'. '+c.title;
  const m=document.createElement('span');m.style.color='var(--muted)';m.textContent=(c.items.reduce((a,i)=>a+(i.seconds||0),0)/60).toFixed(1)+' min';
  const p=document.createElement('button');p.type='button';p.textContent='▶ chapter';p.addEventListener('click',e=>{e.preventDefault();lplay(d.querySelectorAll('li'))});
  s.append(b,m,p);const ol=document.createElement('ol');
  for(const i of c.items){const li=document.createElement('li');li.dataset.k=i.key;li.textContent=i.text;li.title='play from here';
   li.addEventListener('click',()=>{const all=[...document.querySelectorAll('#lchaps li')];lplay(all.slice(all.indexOf(li)))});ol.append(li)}
  d.append(s,ol);box.append(d)})}
// modular layout: drag a panel by its title to reorder, drag its corner to resize; kept in this browser only
const KEY='bankml-view-layout-v1',M=document.querySelector('main');
function save(){try{localStorage.setItem(KEY,JSON.stringify([...M.children].map(c=>({id:c.id,w:c.style.width,h:c.style.height}))))}catch(e){}}
function load(){try{const L=JSON.parse(localStorage.getItem(KEY)||'[]');for(const x of L){const c=document.getElementById(x.id);
 if(c){M.append(c);if(x.w)c.style.width=x.w;if(x.h)c.style.height=x.h}}}catch(e){}}
let drag=null;
for(const c of M.children){const h=c.querySelector('h2');h.draggable=true;
 h.addEventListener('dragstart',e=>{drag=c;c.classList.add('dragging');e.dataTransfer.effectAllowed='move';e.dataTransfer.setData('text/plain',c.id)});
 h.addEventListener('dragend',()=>{c.classList.remove('dragging');for(const x of M.children)x.classList.remove('over');drag=null;save()});
 c.addEventListener('dragover',e=>{if(!drag||drag===c)return;e.preventDefault();c.classList.add('over')});
 c.addEventListener('dragleave',()=>c.classList.remove('over'));
 c.addEventListener('drop',e=>{e.preventDefault();c.classList.remove('over');if(!drag||drag===c)return;
  const r=c.getBoundingClientRect(),after=(e.clientX-r.left)>r.width/2;M.insertBefore(drag,after?c.nextSibling:c);save()});
 new ResizeObserver(()=>{if(c.style.width||c.style.height)save()}).observe(c)}
$('reset').addEventListener('click',()=>{try{localStorage.removeItem(KEY)}catch(e){};location.reload()});
load();
tick();setInterval(tick,2000);setInterval(rec,15000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    server_version = "bankml-view"

    def send(self, code, ctype, body: bytes, cache="no-store"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; media-src 'self'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        if u.path == "/":
            return self.send(200, "text/html; charset=utf-8", PAGE.encode())
        if u.path == "/api/state":
            return self.send(200, "application/json", json.dumps(state()).encode())
        if u.path == "/api/result":
            name = (urllib.parse.parse_qs(u.query).get("name") or [""])[0]
            if name in S.results_list():  # only names the directory listing produced
                return self.send(200, "text/plain; charset=utf-8", S.results_read(name).encode())
            return self.send(404, "text/plain", b"no such record")
        if u.path == "/knobs.js":  # the DreamKnob voice controls, the same bundle the card uses
            try:
                return self.send(200, "text/javascript; charset=utf-8", KNOBS.read_bytes(), cache="max-age=3600")
            except OSError:
                return self.send(404, "text/plain", b"no knobs")
        if u.path.startswith("/audio/") and u.path.endswith(".ogg"):
            f = audio_file(u.path[len("/audio/"):-len(".ogg")])
            return self.send(200, "audio/ogg", f.read_bytes(), cache="max-age=86400") if f else self.send(404, "text/plain", b"no such clip")
        if u.path == "/savante.png":
            rel = (CANON.ledger.get("image_candidate") or {}).get("path") or "gfx/Savante3.png"
            b = CANON.file(rel)
            return self.send(200, "image/png", b, cache="max-age=3600") if b else self.send(404, "text/plain", b"no image")
        return self.send(404, "text/plain", b"not found")

    def do_POST(self):
        self.send(405, "text/plain", b"read-only")

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7874)
    a = ap.parse_args()
    print(f"bankml view on http://{a.host}:{a.port}/ (read-only) · canon ledger {sum(r[2] for r in CANON.rows)}/{len(CANON.rows)}", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
