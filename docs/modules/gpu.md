# `bankML/gpu/` — the video-card component: find GPUs, verify them, and give a verified card a share of the 1-bit matmuls

## Summary

`bankML/gpu/` finds every GPU on the machine, describes it, and puts a card to work only after it has proved that it
gives the CPU kernels' bits. It is a registry of backends. Each backend is a module with one discovery function, and
nothing is linked at build time: the Vulkan backend opens its driver library at run time with `dlopen`, so bankML builds
and runs everywhere, and a machine without the library simply has no devices from that backend.

A card must be a real GPU (integrated or discrete) with a compute queue. Software renderers such as Mesa's llvmpipe run
on the CPU and are refused. Before a card computes anything it must pass the on-card oracle, `kernels::verify_q1_0`
(the oracle rule of [TECHNICAL.md §III.4](../TECHNICAL.md): the same bits first, then speed).
Inside the forward pass a verified card takes the first, calibrated share of a 1-bit (`Q1_0`) matrix's rows while
the CPU pool computes the rest — for each matrix shape where it was measured to pay, within the limit
`BANKML_GPU_LIMIT` sets (0.3.7). Each output element stays one device's exact dot product, so a GPU changes the speed,
never the tokens.

The component is reached three ways: the CLI (`bankml gpu`, `bankml gpu --remote`, `bankml gpu --verify`), the forward
pass (`Weights::open` opens a `gpu::worker::Worker` for `Q1_0` models), and the tests
(`gpu_q1_0_mat_vec_bit_exact`). Remote cards, the GPUs Hugging Face rents, are a separate kind: listed on request,
never selected or started.

## Technical usage

### CLI

```sh
bankml gpu              # every video card found (Vulkan, merged with /sys/class/drm) and which bankML will use, as JSON
bankml gpu --remote     # the same, plus the GPUs Hugging Face rents (listed only)
bankml gpu --verify     # the bit-exact kernel oracle on each selected card; exit code 2 if any card fails
```

`bankml gpu` prints `report_json(remote)`: a `devices` array, the `selected` cards, the `notes` (why a backend found
nothing), and a `kernels` field (Q1_0 on a verified card; Q2_0 and F16 on the CPU). With `--verify`, each selected card prints one line per kernel and shape, or
`NOT VERIFIED — <reason>; bankml will not use this card`. With no usable card it prints
`bankml gpu --verify: no usable GPU found; bankml runs on the CPU`.

### Environment

| variable | effect |
|---|---|
| `BANKML_GPU=off` (also `none`, `cpu`) | turns the component off: nothing is selected, the worker is not opened |
| `BANKML_GPU=0,2` | picks backend indices (the `index` that `bankml gpu` reports) |
| `BANKML_GPU_SHARE=0.3` | overrides the calibrated share of each 1-bit matrix's rows; clamped to 0–0.95 |
| `BANKML_GPU_LIMIT=0.8` | the share of the card's memory and time bankML may use (0.3.7; clamped 0.05–1, default 0.8); see below |

### `mod.rs` — the registry

```rust
pub type Discover = fn() -> Result<Vec<Device>, String>;
pub const BACKENDS: &[(&str, Discover)] = &[("vulkan", vulkan::devices)];
pub const REMOTE_BACKENDS: &[(&str, Discover)] = &[("huggingface", hf::devices)];

pub fn discover() -> (Vec<Device>, Vec<String>)
pub fn selected(devs: &[Device]) -> Vec<Device>
pub fn report_json(remote: bool) -> String
```

- `Device` holds the backend's view (name, vendor and device ids, `Kind`, API version, `device_local_bytes`,
  `host_visible_device_local`, `compute_queues`) plus `count` and `usd_per_hour` for rented hardware, and an optional
  `Sysfs` record.
- `Kind` is `Integrated`, `Discrete`, `Virtual`, `Cpu` (a software renderer, never used), `Remote` (rented, never
  started automatically) or `Other`.
- `Device::usable()` is true only for `Integrated` or `Discrete` with at least one compute queue.
- `discover()` asks every local backend and attaches the kernel's view of the card from `/sys/class/drm/card*/device`:
  driver, VRAM and GTT totals, PCI address, NUMA node. A card is matched by vendor and device id, so it is described
  by what the driver says, not only by what the API reports.
- `device_local_bytes` is the sum of the device-local heaps, the memory the device addresses fastest.
  `host_visible_device_local` is true when some device-local memory is also host-visible (integrated memory,
  resizable BAR), so weights need no staging copy.
- `selected()` applies `BANKML_GPU`, keeps the usable devices, and orders them discrete first, then by device-local
  memory, largest first.
- A new backend (CUDA, ROCm, Metal, …) is a new module and one line in `BACKENDS`.
- A remote card is never selected automatically, because it costs money.

### `vulkan.rs` — Vulkan through `dlopen`

```rust
pub fn devices() -> Result<Vec<Device>, String>
pub(super) fn instance() -> Result<(GetInstanceProcAddr, *mut c_void), String>
```

One API covers every GPU a Vulkan driver exposes: AMD (RADV, AMDVLK), NVIDIA, Intel (ANV), Arm, Qualcomm and Apple
(MoltenVK). The loader is opened at run time: `libvulkan.so.1`, `libvulkan.so`, `libvulkan.1.dylib`, then
`libMoltenVK.dylib`.
Every entry point comes from `vkGetInstanceProcAddr`. The few C structs bankML needs are declared in the file from the
Vulkan 1.x headers. `devices()` creates an instance, enumerates the physical devices, reads their properties, memory
heaps and queue families, and destroys the instance. With no loader it returns
`no Vulkan loader (libvulkan.so.1) on this machine`, and bankML carries on with the CPU. `instance()` creates the
instance compute uses and never destroys it; it lives for the rest of the process. The device name is read from the
first 276 bytes of `VkPhysicalDeviceProperties`, written into a 4 KiB buffer larger than the whole struct.

### `spirv.rs` — bankML's own SPIR-V assembler

bankML writes its compute shaders as SPIR-V words itself, so no shader compiler is needed at build or run time (the
development machine has none, and bankML takes no crates).
`Module` keeps the sections apart and joins them in the order the spec requires (`Module::words()`). It covers only
what compute kernels use: scalar and vector types, storage buffers, push constants, the invocation ids, workgroup
memory and a control barrier, integer and float arithmetic, GLSL.std.450 `Fma`, and structured control flow
(selections and loops). Steps the CPU kernels fuse are written as explicit `Fma`.

```rust
pub fn op(&mut self, opcode: u16, result_ty: u32, operands: &[u32]) -> u32
pub fn two_sum(&mut self, f32t: u32, a: u32, b: u32) -> (u32, u32)
pub fn fma_exact(&mut self, tys: (u32, u32, u32), a: u32, b: u32, c: u32, c_mask: u32, c1: u32, c0: u32, cf0: u32) -> u32
```

- `op` decorates every `OpFMul`, `OpFAdd`, `OpFSub` and extended instruction `NoContraction`, so a driver cannot fuse
  what the CPU kernels keep separate.
- `fma_exact` builds a correctly rounded `fma(a, b, c)` from plain multiplies and adds, whatever the driver does with
  `Fma`. `a` must carry at most 12 significant bits (an f16 value does). `b` is split into two 12-bit halves so both
  partial products are exact, Knuth's TwoSum keeps every rounding error, and Boldo and Melquiond's
  `RN(th + RO(tl + ul))` (rounding to odd, 2008) gives the FMA's single rounding.
- Why it exists: RADV on the Radeon Vega 3 does not fuse `Fma`. On real layer weights the 0.2.13 kernels differed
  from the CPU by about one ulp on 70 % of rows, and decorating the `Fma` `NoContraction` did not change it
  (CHANGELOG 0.2.14).

### `kernels.rs` — the Q1_0 kernels

```rust
pub const Q1_0_BINDINGS: u32 = 5;
pub const LOCAL_SIZE: u32 = 64;
pub fn q1_0_mat_vec() -> Vec<u32>
pub fn q1_0_mat_vec8() -> Vec<u32>
pub fn verify_q1_0(gpu: &super::compute::Gpu) -> Result<Vec<String>, String>
pub fn pack_q1_0(w: &[u8], rows: usize, n: usize) -> (Vec<f32>, Vec<u32>)
pub fn pack_act(a: &crate::q1_0::Q8Act) -> (Vec<f32>, Vec<i32>)
```

- Both kernels follow `q1_0::vec_dot_ref` step by step: per 128-weight block, four 32-element sub-blocks; eight lanes
  of exact integer sums (ggml's i8 negation wraps −128 to −128); `ab = d1·s` on the first sub-block and
  `fma(d1, s, ab)` after; the outer `fma(d0, ab, acc)` through `fma_exact`; then the horizontal sum
  `(a0+a4 + a2+a6) + (a1+a5 + a3+a7)`.
- `q1_0_mat_vec` uses one invocation per output row (dispatch `ceil(rows / 64)` workgroups). `q1_0_mat_vec8` uses
  eight per row, one per accumulation lane, which is valid because each lane's chain over the blocks is independent
  in the CPU kernel too; a workgroup of 64 covers 8 rows. The lane sums meet in workgroup memory and lane 0 adds them
  in the CPU's order, giving the same bits as `q1_0_mat_vec` with eight times the parallelism. Dispatch
  `ceil(rows · 8 / 64)` workgroups.
- Bindings: weight scales (f32), weight bits (u32, four per block), activation scales (f32 per 32), activation quants
  (i32 words of four i8), output (f32). Push constants: rows, blocks per row.
- `pack_q1_0` repacks weights once (scales f16→f32, exact; bits as u32 words). `pack_act` repacks the q8_0 activation
  per call. The numbers are unchanged, only aligned.

### `compute.rs` — Vulkan compute

```rust
impl Gpu {
    pub fn open(index: usize) -> Result<Gpu, String>
    pub fn buffer(&self, bytes: usize) -> Result<Buffer, String>
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<Buffer, String>
    pub fn write<T: Copy>(&self, b: &Buffer, data: &[T])
    pub fn read_f32(&self, b: &Buffer, n: usize) -> Vec<f32>
    pub fn allocated(&self) -> u64                    // bytes bankML's buffers hold (0.3.7, for the limiter)
    pub fn free(&self, b: Buffer)
    pub fn pipeline(&self, spirv: &[u32], bindings: u32, push_bytes: u32) -> Result<Pipeline, String>
    pub fn run(&self, p: &Pipeline, bufs: &[&Buffer], push: &[u32], groups: u32) -> Result<(), String>
    pub fn submit(&self, p: &Pipeline, bufs: &[&Buffer], push: &[u32], groups: u32) -> Result<(), String>
    pub fn wait(&self) -> Result<(), String>
}
```

A device with one compute queue, host-visible coherent buffers mapped for their whole life (device-local and
host-visible memory when the card has it, as an integrated GPU does), pipelines from bankML's own SPIR-V, and a
dispatch that waits on a fence. `run` is `submit` then `wait`. One submission is in flight at a time (one command
buffer, one fence, one descriptor set per pipeline), so `wait` must follow each `submit` before the next; `wait`
without a pending `submit` blocks indefinitely. Every Vulkan call's result is checked.

On a card without device-local host-visible memory, buffers live in host-visible memory the CPU can write. `Gpu` is
`Send` but not `Sync`: one thread drives it at a time (the forward pass holds the worker behind a `Mutex`).

Lifetimes (0.4.1):
- Every object is released on drop. The Vulkan instance, the device and its entry points live in one shared `Device`
  (`Arc`), held by the `Gpu` and by every `Buffer` and `Pipeline` made on it.
- Dropping a `Buffer` destroys it and frees its memory (`Gpu::free` is now just that).
- Dropping a `Pipeline` destroys its pipeline, layout, descriptor pool and set layout. The shader module is destroyed
  as soon as the pipeline is built.
- Dropping a `Gpu` waits for the card, then destroys its fence and command pool.
- The device, and then the instance, go with the last of them. So whatever the drop order, nothing is used after its
  device is gone.
- A `Buffer` is `Send` but no longer `Sync`. `write` and `submit` assert that every buffer and pipeline belongs to their
  own device.

`gpu_objects_are_released_on_drop` measures it with the driver's own counters (sysfs VRAM + GTT): 30 rounds of open,
five 16 MB buffers, a pipeline, a run and drop.
- Before: card memory grew 65 MB, about 2.2 MB a round. Each `Gpu::open` made a Vulkan instance and never destroyed
  it, and nothing else was destroyed either.
- After: 221.4 → 223.4 MB, within the desktop's own noise.

### `worker.rs` — the GPU beside the CPU's threads

```rust
impl Worker {
    pub fn open(pool: &Pool) -> Result<Option<Worker>, String>
    pub fn begin(&mut self, name: &str, w: &[u8], rows: usize, a: &Q8Act) -> Result<usize, String>
    pub fn finish(&mut self, out: &mut [f32]) -> Result<(), String>
}
```

- `open` takes the first selected card, runs `verify_q1_0` on it, and builds the `q1_0_mat_vec8` pipeline. It
  returns `None` when there is no usable card or the share is below 0.02.
- The share comes from a calibration: the same 4096×4096 product timed on the card and on the CPU pool (best of
  three each), `share = card rate / (card rate + CPU rate)`. `BANKML_GPU_SHARE` overrides it.
- In `Weights::mv`, for each `Q1_0` product, `begin` starts the card on the first `share` of the rows and returns
  how many it took; the CPU pool computes the rest with `q1_0::mat_vec_par`; `finish` waits and copies the card's
  rows in. `finish` is called after every `begin`, even when the card took no rows, because it also records the
  shape's timing.
- Only the card's share of each matrix is copied to it, repacked exactly, on first use. Activations wider than
  16,384 are left to the CPU.
- A failure to open or verify is logged (`bankml: GPU not used: …`) and the CPU path runs alone.

### `hf.rs` — Hugging Face's rented GPUs

```rust
pub const HARDWARE_URL: &str = "https://huggingface.co/api/jobs/hardware";
pub fn devices() -> Result<Vec<Device>, String>
pub fn parse(text: &str) -> Result<Vec<Device>, String>
```

Reads the public Jobs hardware list (no token) through the system `curl`, because bankML has no TLS stack of its own.
Each GPU flavor becomes one `Kind::Remote` device with its card count, memory and price per hour (the API's
per-minute price × 60). A multi-card flavor such as `h200x8` is one device with `count` 8. Nothing is started, rented
or paid for. Using a rented card means running bankML there as a Hugging Face Job, where the Vulkan backend finds it
like any local card. See [huggingface.md](../huggingface.md).

The flavors range from an Nvidia T4 to 8× H200. Hugging Face provisions them on the large clouds (its Inference
Endpoints name AWS, Azure and GCP); NVIDIA agreed on 2026-09-02 to acquire Hugging Face. `curl` is used as the
installer's downloads use it; without `curl` or the network the backend reports why and bankML carries on with local
cards. A price per hour is reported only when the API's price is per minute.

### The limiter and per-shape calibration (0.3.7)

```rust
pub fn limit() -> f64                 // BANKML_GPU_LIMIT, clamped 0.05–1, default 0.8
pub fn status() -> Option<Status>     // None when no card works in this process
pub fn status_json() -> String        // GET /bankml/usage's "gpu_limiter"
```

- **`BANKML_GPU_LIMIT`** = L: bankML's buffers stay within L of the heap they come from (the `Gpu.heap_bytes` field,
  against `Gpu::allocated()`), and on an integrated card within L of the RAM they could use (what they hold plus what
  the system has available). After a dispatch of `d` the card rests `d·(1−L)/L`. A matrix that does not fit, or a
  product arriving while the card rests, runs whole on the CPU — never a wait, the same bits.
- **Per-shape calibration**: each matrix shape's first products alternate between the calibrated share and the CPU
  alone, timed `begin` → `finish` in the real pipeline (the CPU's rows included). After a warm-up (which includes
  the upload) and 6 timings each (`TRIALS`) the faster median is kept, and a shape decided for the CPU frees its
  buffers. A product the card skipped because it was resting or out of memory is not counted as a timing of the
  card. The bits are the same either way.
- Why per shape: the one-off calibration measures a 4096×4096 product in isolation. In the forward pass a smaller
  matrix may not repay the submit-and-wait, and on an integrated card (whose heap is system RAM) the card and the CPU
  share the memory bandwidth that decode is bound by.
- `status_json` reports `card`, `limit`, `share`, `allocated_bytes`, `heap_bytes`, `integrated`, `busy`,
  `products_on_card`, `products_on_cpu_resting`, `products_on_cpu_memory`, `shapes_on_card`, `shapes_on_cpu` and
  `shapes_tuning`. The console's Admin tab charts it ([console.md](console.md)).
- Measured (CHANGELOG 0.3.7; Vega 3, Bonsai-1.7B, interleaved A/B, 4 rounds, the same answer every time): with one
  global share the card cost decode 18 % (10.40 → 8.51–8.58 tok/s); per shape, 10.36 tok/s off against 10.30–10.38
  on — neutral — with ~13 MB on the card instead of 66 MB.

## How it is verified

- **`gpu_q1_0_mat_vec_bit_exact`** (`kernels.rs`, ignored by default: needs a Vulkan GPU; in the release gate). Both
  kernels on every usable card against bankML's CPU kernel, which is bit-exact against ggml, on random weights and
  activations with −128 quants. Shapes 64×128 to 12288×4096. On the Vega 3, every row of every shape is bit-exact
  ([oracles.md](../oracles.md), 0.2.13).
- **`bankml gpu --verify`** runs `verify_q1_0`: the same comparison, plus a layer-shaped regime (scales and magnitudes
  that vary per block, so products are inexact in f32). With the driver's `Fma` put back, it refuses the Vega 3
  ("row 1 differs … bankml will not use this card"). The worker runs the same check before it takes any work.
- **The forward pass with the card working.** Every token oracle runs through `Weights::mv`, so with a verified card
  present they all run with the GPU computing its share. In 0.2.14 the whole 1-bit model passed at 1,064 of 1,064
  rows with the Vega 3 on 26 % of every matrix, and the greedy and sampling checks went through the same path.
- Unit tests: `software_renderers_are_never_selected_and_discrete_cards_come_first`, `discovery_never_panics`
  (`mod.rs`), `a_multi_card_flavor_is_one_device_with_its_count` (`hf.rs`), `header_and_string_words` (`spirv.rs`).

## Advantages and efficiency

- **The same bits, then speed.** A card is trusted only after it reproduces the CPU kernel's bits on the card, and it
  works on whole rows, so no element is split between devices. The answer and the receipt do not depend on whether a
  GPU was present.
- **No SDK, no crate, no shader compiler.** Vulkan is opened with `dlopen` and every entry point comes from
  `vkGetInstanceProcAddr`; shaders are assembled in Rust by `spirv.rs`. The same binary runs on a machine with or
  without a GPU. One API covers AMD, NVIDIA, Intel, Arm and Qualcomm, and a rented NVIDIA card needs no CUDA toolkit
  ([huggingface.md](../huggingface.md)).
- **The card and the CPU finish together.** The calibrated share gives the card the fraction of rows that matches its
  rate, and the CPU pool computes the rest while the card works (`submit` returns at once; `wait` is called after the
  CPU part).
- **Exact FMA only where needed.** The inner `fma(d1, s, ab)` stays a plain `Fma`, because `d1·s` is exact in f32;
  only the outer `fma(d0, ab, acc)` uses `fma_exact` (CHANGELOG 0.2.14).
- **Weights copied once, and only the card's share.** On an integrated GPU, buffers are allocated in device-local,
  host-visible memory when the card offers it.
- **Measured.** On the Vega 3 the eight-lane kernel matches one CPU thread, 1.7–2.0 ms per 4096×4096 (CHANGELOG
  0.2.13). The calibrated share is 26–35 % there. Decode is unchanged within noise: 1.93–2.00 tokens/s with the card
  against 1.96–1.97 without (CHANGELOG 0.2.14).
- **Rust practice visible in the code.** No external crate (the manifest's `[dependencies]` is empty). `unsafe` is
  confined to the FFI files `vulkan.rs` and `compute.rs`, with `SAFETY` comments, behind safe methods on `Gpu`; the
  registry, kernels, assembler, worker and Hugging Face backend have none. Errors are `Result<_, String>` with the
  failing call named, and a failure falls back to the CPU instead of stopping. Verification fails closed. The
  toolchain is pinned to Rust 1.99.0 in `rust-toolchain.toml`.
- **Where optimization goes next** ([TODO.md](../TODO.md), 0.5.0): batched submissions (Q/K/V and gate/up in one
  command buffer, one wait per group, persistent descriptor sets), the Q2_0 (ternary) GPU kernel in `--verify`,
  several cards each taking a share of every matrix's rows, and a discrete-card measurement (a rented T4 or L4 run,
  only with the owner's approval).

## Limitations

- **Speed-neutral on the APU.** On the Vega 3 the card is worth about one CPU core, and each of the 253 matrix calls
  per token pays one submit and wait, which cancels the gain (CHANGELOG 0.2.14). Per-shape calibration (0.3.7) keeps
  the card from costing decode, but does not make it faster there. One submission is in flight at a time.
- **`Q1_0` only.** The worker is opened only for 1-bit models; there is no Q2_0 (ternary) or F16 GPU kernel yet.
- **One card.** The worker uses the first selected card; several cards are not yet used together. The design for
  several cards gives each a share of every matrix's rows, which keeps each output element one card's exact dot
  product.
- **No Vulkan, no GPU.** Without a Vulkan loader, a usable card, or a passing verification, bankML runs on the CPU.
  Software renderers are never used.
- **Remote cards are listed, never run.** `bankml gpu --remote` needs `curl` and the network; nothing is provisioned.
- No discrete card has been measured yet, and llama.cpp's own Vulkan build has not been measured as a comparison
  ([TODO.md](../TODO.md)).

## See also

- [usage.md](../usage.md) — `bankml gpu`, `BANKML_GPU`, `BANKML_GPU_SHARE`, `BANKML_GPU_LIMIT`
- [sys.md](sys.md) and [metrics.md](metrics.md) — `/bankml/usage` (GPU busy, VRAM, GTT, the limiter) and the answers' measurements
- [oracles.md](../oracles.md) — 0.2.13 and 0.2.14, the GPU oracles
- [PERFORMANCE.md](../PERFORMANCE.md) — decode speed
- [huggingface.md](../huggingface.md) — the rented GPUs
- [TODO.md](../TODO.md) — 0.5.0, hardware
- [forward.md](forward.md) — `Weights::open` and `Weights::mv`
- [q1_0.md](q1_0.md) — the CPU kernel the GPU reproduces
