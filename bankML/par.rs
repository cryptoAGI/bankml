// SPDX-License-Identifier: MIT OR Apache-2.0
//! Thread pool and row scheduler for the matmuls, using only `std`.
//!
//! Workers are spawned once and woken per matmul; rows are claimed in fixed-size chunks from an atomic
//! counter, as ggml's `mul_mat` does. Each row is computed by the same single-thread kernel, so the
//! output bits do not depend on the thread count.
//! Details: docs/modules/par.md.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Rows per scheduler claim (measured best for the ternary GEMV).
pub const CHUNK_ROWS: usize = 16;

type Job = *const (dyn Fn(usize) + Sync);

struct State {
    job: Option<Job>,
    generation: u64,
    running: usize,
    quit: bool,
}

// SAFETY: the raw job pointer is only dereferenced while `run` blocks, which keeps the closure alive.
unsafe impl Send for State {}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    done: Condvar,
}

/// A fixed set of worker threads.
///
/// The caller's thread is worker 0, so `Pool::new(1)` spawns nothing.
pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<std::thread::JoinHandle<()>>,
    n: usize,
}

impl Pool {
    pub fn new(threads: usize) -> Self {
        let n = threads.max(1);
        let shared = Arc::new(Shared {
            state: Mutex::new(State { job: None, generation: 0, running: 0, quit: false }),
            wake: Condvar::new(),
            done: Condvar::new(),
        });
        let workers = (1..n)
            .map(|id| {
                let s = shared.clone();
                std::thread::spawn(move || {
                    let mut seen = 0;
                    loop {
                        let job = {
                            let mut st = s.state.lock().unwrap();
                            while !st.quit && st.generation == seen {
                                st = s.wake.wait(st).unwrap();
                            }
                            if st.quit {
                                return;
                            }
                            seen = st.generation;
                            st.job.unwrap()
                        };
                        // SAFETY: `run` holds the closure alive until `running` reaches 0
                        unsafe { (*job)(id) };
                        let mut st = s.state.lock().unwrap();
                        st.running -= 1;
                        if st.running == 0 {
                            s.done.notify_one();
                        }
                    }
                })
            })
            .collect();
        Pool { shared, workers, n }
    }

    /// `BANKML_THREADS`, else the machine's available parallelism.
    pub fn from_env() -> Self {
        let n = std::env::var("BANKML_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
        Self::new(n)
    }

    pub fn threads(&self) -> usize {
        self.n
    }

    /// Runs `f(worker_id)` on every worker (ids 0..threads) and returns when all have finished.
    pub fn run(&self, f: &(dyn Fn(usize) + Sync)) {
        if self.n == 1 {
            return f(0);
        }
        // SAFETY: erase the borrow's lifetime; this function does not return until every worker is done with it
        let job: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f) };
        {
            let mut st = self.shared.state.lock().unwrap();
            st.job = Some(job);
            st.generation += 1;
            st.running = self.n - 1;
            self.shared.wake.notify_all();
        }
        f(0);
        let mut st = self.shared.state.lock().unwrap();
        while st.running > 0 {
            st = self.shared.done.wait(st).unwrap();
        }
        st.job = None;
    }

    /// Fills `out[..rows]`; `f(r0, chunk)` computes rows `r0..r0 + chunk.len()`.
    ///
    /// Chunks of `CHUNK_ROWS` are claimed dynamically, so a slower core takes fewer.
    pub fn rows(&self, rows: usize, out: &mut [f32], f: &(dyn Fn(usize, &mut [f32]) + Sync)) {
        assert!(out.len() >= rows);
        if self.n == 1 || rows <= CHUNK_ROWS {
            return f(0, &mut out[..rows]);
        }
        let next = AtomicUsize::new(0);
        let base = out.as_mut_ptr() as usize;
        self.run(&|_| loop {
            let r0 = next.fetch_add(CHUNK_ROWS, Ordering::Relaxed);
            if r0 >= rows {
                break;
            }
            // SAFETY: each chunk r0..r0+len is claimed by exactly one worker, inside out[..rows]
            let chunk = unsafe { std::slice::from_raw_parts_mut((base as *mut f32).add(r0), CHUNK_ROWS.min(rows - r0)) };
            f(r0, chunk);
        });
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().quit = true;
        self.shared.wake.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_cover_every_row_once_at_any_thread_count() {
        for th in [1, 2, 3, 5] {
            let pool = Pool::new(th);
            for rows in [0, 1, 15, 16, 17, 100, 1000] {
                let mut out = vec![-1f32; rows + 3];
                pool.rows(rows, &mut out, &|r0, o| o.iter_mut().enumerate().for_each(|(i, v)| *v = (r0 + i) as f32));
                assert!(out[..rows].iter().enumerate().all(|(i, v)| *v == i as f32), "th {th} rows {rows}");
                assert!(out[rows..].iter().all(|v| *v == -1.0));
            }
            // the pool is reusable
            let hits = AtomicUsize::new(0);
            for _ in 0..200 {
                pool.run(&|_| {
                    hits.fetch_add(1, Ordering::Relaxed);
                });
            }
            assert_eq!(hits.into_inner(), 200 * th);
        }
    }

    /// Streaming read bandwidth of a 768 MiB buffer at 1–4 threads: the decode kernels' memory floor.
    ///
    /// `cargo test --release -- --ignored bench_memory_floor --nocapture --test-threads=1`
    #[test]
    #[ignore = "benchmark; allocates 768 MiB"]
    fn bench_memory_floor() {
        let words = (768usize << 20) / 8;
        let buf: Vec<u64> = (0..words as u64).collect();
        for th in 1..=4 {
            let pool = Pool::new(th);
            let mut best = f64::MAX;
            for _ in 0..5 {
                let chunk = 1 << 16; // 512 KiB of u64 per claim
                let next = AtomicUsize::new(0);
                let total = AtomicUsize::new(0);
                let t = std::time::Instant::now();
                pool.run(&|_| {
                    let mut acc = 0u64;
                    loop {
                        let c = next.fetch_add(chunk, Ordering::Relaxed);
                        if c >= words {
                            break;
                        }
                        acc = buf[c..(c + chunk).min(words)].iter().fold(acc, |a, &v| a ^ v);
                    }
                    total.fetch_add(std::hint::black_box(acc) as usize & 1, Ordering::Relaxed);
                });
                best = best.min(t.elapsed().as_secs_f64());
            }
            let gbs = (words * 8) as f64 / best / 1e9;
            eprintln!("{th} thread(s): {gbs:5.2} GB/s read → floor for one ternary token (2.13 GB) {:.3} s, one 1-bit token (1.06 GB) {:.3} s", 2.13 / gbs, 1.06 / gbs);
        }
    }

    /// Wake-up cost per `run` against spawning scoped threads (`--ignored --nocapture`).
    #[test]
    #[ignore = "benchmark"]
    fn bench_pool_overhead() {
        let n = 3;
        let pool = Pool::new(n);
        let t = std::time::Instant::now();
        for _ in 0..2000 {
            pool.run(&|_| {});
        }
        let p = t.elapsed().as_secs_f64() / 2000.0 * 1e6;
        let t = std::time::Instant::now();
        for _ in 0..500 {
            std::thread::scope(|s| {
                for _ in 1..n {
                    s.spawn(|| {});
                }
            });
        }
        let sc = t.elapsed().as_secs_f64() / 500.0 * 1e6;
        eprintln!("{n} threads: pool.run {p:.1} µs/call, thread::scope spawn {sc:.1} µs/call (x253 matmuls/token: {:.1} vs {:.1} ms)", p * 0.253, sc * 0.253);
    }
}
