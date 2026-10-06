# Hugging Face — the Hub's GPUs as bankML devices (addendum, 2026-09-29)

An addendum to bankML's interaction with Hugging Face: bankML already imports models from the Hub (the importer's
catalogue and any Hugging Face GGUF, each pinned by the publisher's sha256), and since 0.2.12 its video-card
component lists the GPUs Hugging Face rents (`bankml gpu --remote`). This page records what those GPUs are, who
provisions them, and the news that changes who owns them.

### The deal, as announced

On **2026-09-02** NVIDIA entered a definitive agreement to acquire Hugging Face (NVIDIA Form 8-K, filed the same
week; confirmed publicly 2026-09-03). The terms are about **$11.9 billion** to Hugging Face stockholders plus an
equity retention program of up to about **$1.0 billion** for employees joining NVIDIA, reported as **$12.9 billion**
in all. ⚠ **It has not closed.** Closing is expected in the **first half of 2027**, subject to regulatory approvals,
so Hugging Face is still an independent company until then.
Sources: [NVIDIA 8-K](https://www.sec.gov/Archives/edgar/data/0001045810/000104581026000078/nvda-20260902.htm) ·
[NVIDIA blog](https://blogs.nvidia.com/blog/nvidia-to-acquire-hugging-face/) ·
[TechCrunch](https://techcrunch.com/2026/09/03/nvidia-confirms-it-will-buy-hugging-face-for-12-9-billion/) ·
[CNBC](https://www.cnbc.com/2026/09/03/nvidia-agrees-to-buy-hugging-face-for-almost-13-billion-ai-expansion.html) ·
[Bloomberg](https://www.bloomberg.com/news/articles/2026-09-03/nvidia-agrees-to-13-billion-deal-for-ai-platform-hugging-face)

What changed is who will own the rented compute, and that the rented compute is, already, **all NVIDIA**.

### The provisioners and the hardware, read live today

- **Provisioners.** Inference Endpoints' public provider list
  (`GET https://api.endpoints.huggingface.cloud/v2/provider`) names **AWS** (eu-west-1, us-east-1, us-east-2,
  us-west-2), **Azure** (eastus) and **GCP** (us-east4).
- **Jobs hardware** (`GET https://huggingface.co/api/jobs/hardware`, public, no token): **26 flavors, 22 of them
  GPU**, every GPU an NVIDIA part. Prices are the API's per-minute figures × 60.

| flavor | GPUs | VRAM | $/h |
|---|---|--:|--:|
| t4-small / t4-medium | 1× T4 | 16 GB | 0.40 / 0.60 |
| l4x1 / l4x4 | 1× / 4× L4 | 24 / 96 GB | 0.80 / 3.80 |
| a10g-small / a10g-large | 1× A10G | 24 GB | 1.00 / 1.50 |
| a10g-largex2 / a10g-largex4 | 2× / 4× A10G | 48 / 96 GB | 3.00 / 5.00 |
| l40sx1 / l40sx4 / l40sx8 | 1× / 4× / 8× L40S | 48 / 192 / 384 GB | 1.80 / 8.30 / 23.50 |
| a100-large / a100x4 / a100x8 | 1× / 4× / 8× A100 | 80 / 320 / 640 GB | 2.50 / 10.00 / 20.00 |
| rtx-pro-6000 ×1 / ×2 / ×4 / ×8 | RTX PRO 6000 | 96 / 192 / 384 / 768 GB | 2.75 / 5.50 / 11.00 / 22.00 |
| h200 ×1 / ×2 / ×4 / ×8 | H200 | 141 / 282 / 564 / 1128 GB | 5.00 / 10.00 / 20.00 / 40.00 |

- **Spaces hardware** (`GET https://huggingface.co/api/spaces/hardware`) lists 14 GPU flavors plus ZeroGPU (free,
  dynamic). ⚠ **The Spaces list mislabels the L40S flavors:** `l40sx1/x4/x8` report `"model": "L4"` there,
  while the Jobs list says L40S with the right memory (48/192/384 GB). Read the model from the Jobs list.

### bankml sees them (bankml 0.2.12)

bankml grew a video-card component (`bankML/gpu/`, a registry of backends with no crates). **`bankml gpu`** lists
the local cards through Vulkan, merged with the kernel's `/sys/class/drm` view. On this node that is a Radeon
Vega 3, and Mesa's llvmpipe is refused as a software renderer. **`bankml gpu --remote`** adds the Hugging Face
backend (`gpu/hf.rs`), which reads the Jobs hardware list above through the system `curl` (bankml has no TLS of its
own) and reports every GPU flavor as a **remote** device: its card count, memory and price per hour. A
multi-card flavor such as `h200x8` is one device with count 8.

The rules, stated in code and tested:
- **A rented card is never selected automatically and never provisioned by bankml.** Listing costs nothing;
  running costs money, and that is the owner's decision (mindX's cost doctrine for Hugging Face is in its `docs/HUGGINGFACE_INTEGRATION.md`).
- **Using one means running bankml where the card is:** a Hugging Face Job on the chosen flavor. There bankml's
  Vulkan backend finds the NVIDIA card like any local one; NVIDIA's driver ships a Vulkan ICD. No CUDA toolkit is
  needed.
- **Several cards share each matrix by rows**, so every output element stays one card's exact dot product. The
  GPU kernels must reproduce bankml's CPU bits, which match llama.cpp's, on the card before they are trusted: a GPU
  changes the speed, never the tokens.
- The CUDA Rust tracks of 2026-09-18 (`cutile-rs`, `cuda-oxide`, both Apache-2.0) remain the candidate for a
  native NVIDIA backend module (`gpu/cuda.rs`) if Vulkan compute leaves speed on the table on these parts.
  Vulkan comes first because it covers AMD and Intel as well, with nothing linked.

### Where it stands (updated 2026-10-06)

- Discovery is done: local cards via Vulkan and sysfs, remote cards via the Jobs API.
- The Q1_0 kernel is done in bankML's own SPIR-V: bit-exact against the CPU reference (0.2.13), working inside the
  forward pass (0.2.14), and checked on each card by `bankml gpu --verify` before it is used. Since 0.3.7
  (unreleased) `BANKML_GPU_LIMIT` caps the card's share of memory and time, and each matrix shape is calibrated
  between card and CPU ([modules/gpu.md](modules/gpu.md), CHANGELOG 0.3.7). The only card verified so far is the
  integrated Radeon Vega 3.
- Q2_0 (ternary) and F16 still run on the CPU. Their GPU kernels, and the vendor matrix (AMD discrete, NVIDIA,
  Intel), are in [TODO.md 0.5.0](TODO.md#050--hardware).
- A Job recipe that runs `bankml` on a chosen flavor, with its oracle on the card, comes with the first rented run.
- No money has been spent and none will be spent without the owner starting a Job.
