// SPDX-License-Identifier: MIT OR Apache-2.0
// browseML's WASI: the few system calls bankML's engine makes, answered in memory, for the browser and for Node's
// oracle alike. One read-only directory (/models) holding the files the page gives it (the model it downloaded and
// checked), the clocks, random bytes, environment variables, and standard output/error to a callback. Everything
// else is refused with a WASI error, never faked: the engine reports it as it would on a machine without it.
// Zero dependencies.

const ERR = { SUCCESS: 0, ACCES: 2, BADF: 8, INVAL: 28, NOENT: 44, NOTSUP: 58 };
const FT = { DIR: 3, FILE: 4 };
const MOUNT = "/models";

/**
 * @param {{files: Object<string, Uint8Array>, env?: Object<string,string>, write?: (fd:number, text:string) => void}} o
 * @returns {{imports: object, bind: (memory: WebAssembly.Memory) => void, release: (name: string) => void}}
 */
export function wasi({ files, env = {}, write = () => {} }) {
  let memory = null;
  const dv = () => new DataView(memory.buffer);
  const u8 = () => new Uint8Array(memory.buffer);
  const dec = new TextDecoder(), enc = new TextEncoder();
  // copies, not views: with threads the memory is a SharedArrayBuffer, which TextDecoder and getRandomValues refuse
  const str = (p, n) => dec.decode(u8().slice(p, p + n));
  const fds = new Map([[3, { dir: true }]]); // 0–2 are stdio; 3 is the preopened /models
  let next = 4;

  const name = (p, n) => str(p, n).replace(/^\.?\/+/, "").replace(/^models\//, "");
  const stat = (buf, type, size) => {
    const v = dv();
    for (let i = 0; i < 64; i += 8) v.setBigUint64(buf + i, 0n, true);
    v.setUint8(buf + 16, type);
    v.setBigUint64(buf + 24, 1n, true);
    v.setBigUint64(buf + 32, BigInt(size), true);
  };
  const iovs = (p, n) => { const v = dv(), out = []; for (let i = 0; i < n; i++) out.push([v.getUint32(p + 8 * i, true), v.getUint32(p + 8 * i + 4, true)]); return out; };
  const envList = Object.entries(env).map(([k, v]) => enc.encode(`${k}=${v}\0`));

  const imports = {
    args_sizes_get: (argc, size) => { dv().setUint32(argc, 0, true); dv().setUint32(size, 0, true); return ERR.SUCCESS; },
    args_get: () => ERR.SUCCESS,
    environ_sizes_get: (count, size) => {
      dv().setUint32(count, envList.length, true); dv().setUint32(size, envList.reduce((a, e) => a + e.length, 0), true); return ERR.SUCCESS;
    },
    environ_get: (ptrs, buf) => {
      let at = buf; envList.forEach((e, i) => { dv().setUint32(ptrs + 4 * i, at, true); u8().set(e, at); at += e.length; }); return ERR.SUCCESS;
    },
    clock_time_get: (id, _precision, out) => {
      const ns = id === 0 ? BigInt(Date.now()) * 1000000n : BigInt(Math.round(performance.now() * 1e6));
      dv().setBigUint64(out, ns, true); return ERR.SUCCESS;
    },
    clock_res_get: (_id, out) => { dv().setBigUint64(out, 1000n, true); return ERR.SUCCESS; },
    random_get: (p, n) => {
      for (let i = 0; i < n; i += 65536) { const b = crypto.getRandomValues(new Uint8Array(Math.min(65536, n - i))); u8().set(b, p + i); }
      return ERR.SUCCESS;
    },
    fd_prestat_get: (fd, buf) => {
      if (fd !== 3) return ERR.BADF;
      dv().setUint8(buf, 0); dv().setUint32(buf + 4, MOUNT.length, true); return ERR.SUCCESS;
    },
    fd_prestat_dir_name: (fd, p, n) => { if (fd !== 3) return ERR.BADF; u8().set(enc.encode(MOUNT).subarray(0, n), p); return ERR.SUCCESS; },
    fd_fdstat_get: (fd, buf) => {
      const f = fds.get(fd); if (!f && fd > 2) return ERR.BADF;
      const v = dv(); v.setUint8(buf, f ? (f.dir ? FT.DIR : FT.FILE) : 2); v.setUint16(buf + 2, 0, true);
      v.setBigUint64(buf + 8, 0xffffffffn, true); v.setBigUint64(buf + 16, 0xffffffffn, true); return ERR.SUCCESS;
    },
    path_open: (dirfd, _dirflags, p, n, oflags, _rb, _ri, _fdflags, out) => {
      if (dirfd !== 3) return ERR.BADF;
      const f = name(p, n);
      if (f === "" || f === ".") { const fd = next++; fds.set(fd, { dir: true }); dv().setUint32(out, fd, true); return ERR.SUCCESS; }
      if (oflags & (1 | 4 | 8)) return ERR.ACCES; // create, exclusive, truncate: the mount is read-only
      if (!(f in files)) return ERR.NOENT;
      const fd = next++; fds.set(fd, { data: files[f], pos: 0 }); dv().setUint32(out, fd, true); return ERR.SUCCESS;
    },
    path_filestat_get: (dirfd, _flags, p, n, buf) => {
      if (dirfd !== 3) return ERR.BADF;
      const f = name(p, n);
      if (f === "" || f === ".") { stat(buf, FT.DIR, 0); return ERR.SUCCESS; }
      if (!(f in files)) return ERR.NOENT;
      stat(buf, FT.FILE, files[f].length); return ERR.SUCCESS;
    },
    path_readlink: () => ERR.INVAL, // nothing here is a link
    fd_filestat_get: (fd, buf) => {
      const f = fds.get(fd); if (!f) return ERR.BADF;
      stat(buf, f.dir ? FT.DIR : FT.FILE, f.dir ? 0 : f.data.length); return ERR.SUCCESS;
    },
    fd_read: (fd, iv, n, nread) => {
      const f = fds.get(fd); if (!f || f.dir) return ERR.BADF;
      let total = 0;
      for (const [p, len] of iovs(iv, n)) {
        const k = Math.min(len, f.data.length - f.pos); if (k <= 0) break;
        u8().set(f.data.subarray(f.pos, f.pos + k), p); f.pos += k; total += k;
      }
      dv().setUint32(nread, total, true); return ERR.SUCCESS;
    },
    fd_seek: (fd, offset, whence, out) => {
      const f = fds.get(fd); if (!f || f.dir) return ERR.BADF;
      const base = whence === 0 ? 0 : whence === 1 ? f.pos : f.data.length, to = base + Number(offset);
      if (to < 0) return ERR.INVAL;
      f.pos = to; dv().setBigUint64(out, BigInt(to), true); return ERR.SUCCESS;
    },
    fd_readdir: (fd, _buf, _len, _cookie, used) => { if (!fds.has(fd)) return ERR.BADF; dv().setUint32(used, 0, true); return ERR.SUCCESS; },
    fd_close: (fd) => (fds.delete(fd) ? ERR.SUCCESS : ERR.BADF),
    fd_write: (fd, iv, n, written) => {
      if (fd !== 1 && fd !== 2) return ERR.BADF;
      let total = 0, s = "";
      for (const [p, len] of iovs(iv, n)) { s += str(p, len); total += len; }
      write(fd, s); dv().setUint32(written, total, true); return ERR.SUCCESS;
    },
    sched_yield: () => ERR.SUCCESS,
    proc_exit: (code) => { throw new Error(`browseML exited (${code})`); },
  };
  // anything else the module imports is answered "not supported", never faked
  const proxy = new Proxy(imports, { get: (t, k) => (k in t ? t[k] : () => ERR.NOTSUP) });
  return {
    imports: { wasi_snapshot_preview1: proxy },
    bind: (m) => { memory = m; },
    // once the engine has read a file into its own memory, the page's copy can go
    release: (n) => { delete files[n]; },
  };
}
