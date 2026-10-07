# The console's landing: the ultimate input field

`uif.js` and `uif.css` are a build, not source. They hold the bankML console's Ask landing: the
[ultimate input field](https://github.com/Professor-Codephreak/ultimate-input-field) (MIT), with the console's own
engine (`POST /api/ask`, a receipt on every answer, checked in the browser).

- **Source:** `src/console/main.tsx`, `src/console/consoleEngine.ts` and `vite.console.config.ts` of ultimate-bankml-ui,
  branch `console-mount` (from release v0.1.1, cad8817; the input field vendored at 9c6b595). The Space
  [PYTHAI/ultimate-bankml-ui](https://huggingface.co/spaces/PYTHAI/ultimate-bankml-ui) carries the whole source.
- **Build:** `npx vite build -c vite.console.config.ts` → `dist-console/uif.js`, `uif.css`. React is bundled in, so
  the console still loads nothing from elsewhere (CSP `script-src 'self'`).
- Without these two files, the console falls back to its plain question box.

MIT. The input field is copyright Professor Codephreak; see the Space's LICENSE.
