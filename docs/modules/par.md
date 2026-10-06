# `bankML/par.rs` — the persistent thread pool and row scheduler for the matmuls

## Summary

`par.rs` gives the kernels threads with no crate: a pool of workers spawned once and woken for each matmul, and a
row scheduler that hands out fixed-size row chunks from an atomic counter, as ggml's `mul_mat` does.

Rows are independent, and each row is computed by the same single-thread kernel, so the output bits do not depend on
the thread count. The oracle tests check that.

It exists because one token of the 8B models is 253 matmuls (`decode_budget_q1_0`). Spawning threads for each one
cost about 22 ms per token; waking the pool costs about 3 ms (CHANGELOG 0.0.3).
Callers: the `_par` functions of [q1_0.md](q1_0.md), [q2_0.md](q2_0.md) and [f16.md](f16.md); the native forward
pass (`forward.rs`, which builds its pool with `Pool::from_env()`); and the A/B harnesses, which run ggml's kernel on
the same pool so that the comparison is kernel against kernel.

## Technical usage

```rust
pub const CHUNK_ROWS: usize = 16;

pub struct Pool { /* private */ }
impl Pool {
    pub fn new(threads: usize) -> Self
    pub fn from_env() -> Self
    pub fn threads(&self) -> usize
    pub fn run(&self, f: &(dyn Fn(usize) + Sync))
    pub fn rows(&self, rows: usize, out: &mut [f32], f: &(dyn Fn(usize, &mut [f32]) + Sync))
}
```

- `new(threads)`: the caller's thread is worker 0, so `Pool::new(1)` spawns nothing. `threads` below 1 is taken as 1.
- `from_env()`: `BANKML_THREADS` if it parses, else the machine's available parallelism.
- `run(f)`: calls `f(worker_id)` on every worker (ids `0..threads`) and returns when all have finished.
- `rows(rows, out, f)`: fills `out[..rows]`; `f(r0, chunk)` computes rows `r0..r0 + chunk.len()`. Chunks of
  `CHUNK_ROWS` are handed out dynamically, so a slower core takes fewer. With one thread, or `rows <= CHUNK_ROWS`,
  it calls `f(0, &mut out[..rows])` directly.
- Dropping the pool sets `quit`, wakes the workers and joins them.

Environment:

| variable | effect |
|---|---|
| `BANKML_THREADS` | the pool's size in `from_env`, read when the forward pass opens a model (Savante's and the console's CPU-threads slider restart the engine with a new value); in the decode budget tests, a comma list of thread counts |

```rust
let pool = bankml::par::Pool::from_env();
let mut out = vec![0f32; rows];
pool.rows(rows, &mut out, &|r0, o| {
    for (i, v) in o.iter_mut().enumerate() { *v = compute_row(r0 + i); }
});
```

## How it is verified

- `rows_cover_every_row_once_at_any_thread_count`: 1, 2, 3 and 5 threads; 0 to 1,000 rows; every row written once
  and nothing past `rows`; the pool reused 200 times.
- The kernels' `par_bit_exact_with_single_thread` tests (in `q1_0.rs` and `q2_0.rs`) and the F16 tests run the
  `_par` paths against single-thread results, bit for bit.
- `decode_budget_q1_0` / `decode_budget_q2_0` compare every threaded output with ggml's.
- Benchmarks (`#[ignore]`): `bench_pool_overhead` (wake-up cost per `run` against `std::thread::scope`) and
  `bench_memory_floor` (added in 0.0.4: streaming read bandwidth of a 768 MiB buffer, far larger than cache, on
  the same scheduler, 1–4 threads; it prints the resulting floor for one ternary token, 2.13 GB, and one 1-bit token,
  1.06 GB). Run it with `cargo test --release -- --ignored bench_memory_floor --nocapture --test-threads=1`.

## Advantages and efficiency

- **Spawn once, wake per matmul.** Spawning threads with `std::thread::scope` for every matmul costs about 20 µs
  per spawned thread, tens of ms per token at 253 matmuls (`bench_pool_overhead`). Measured on the laptop at three
  threads (CHANGELOG 0.0.3): 12.6 µs per `run` against 88.8 µs for `thread::scope`, about 3 ms per token instead of
  22 ms. With it the three-thread ternary budget
  moved from 0.36 s to 0.23–0.25 s per token ([PERFORMANCE.md](../PERFORMANCE.md)).
- **Dynamic chunks.** 16 rows per claim measured best for the ternary GEMV on the dev box; a static split was 1.1×
  slower at three threads, because one core also runs the OS.
- **Deterministic bits.** The scheduler decides only which thread computes a row. Results are the same at any thread
  count, which keeps every oracle valid when threads change.
- **The memory floor it measures.** `bench_memory_floor` gave 14.9–15.2 GB/s at one thread and 16.6–17.2 GB/s at
  2–4 threads on the laptop, the floor both kernels are compared against.
- **Practice.** Only `std` (`Mutex`, `Condvar`, `AtomicUsize`, `Arc`). The one lifetime erasure in `run` is
  documented: `run` does not return until every worker is done with the closure. The raw job pointer's `Send` is
  justified the same way. Disjoint output chunks are claimed by exactly one worker each.

## Limitations

- Workers sleep on a condvar between matmuls. On F16 decode the condvar hand-off (about 15.6 µs per call) is the
  suspect for the gap to llama-server. A bounded spin was measured and rejected: on the 2-core SMT laptop it raised
  `pool.run` to 162.6 µs per call and did not move decode ([PERFORMANCE.md](../PERFORMANCE.md), [TODO.md](../TODO.md)).
- One pool runs one job at a time; `run` blocks the caller until the job ends.
- Speed-up from threads is bounded by the machine: on the laptop's two physical cores the ternary kernel scales 1.8×
  from one to three threads.

## See also

- [q1_0.md](q1_0.md), [q2_0.md](q2_0.md), [f16.md](f16.md)
- [../PERFORMANCE.md](../PERFORMANCE.md) (Threads, memory floor), [../usage.md](../usage.md) §13 (environment),
  [../TODO.md](../TODO.md)
