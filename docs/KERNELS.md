# Reference-Equivalent Low-Bit Kernels across x86, WebAssembly and WebGPU

*bankML · kernel treatise · 2026-10-14 · engine at `f618309` (branch `browseml`), browseML 0.6.9*

## Abstract

Inference engines for quantized language models are usually validated by tolerance: a kernel passes if its outputs
lie within some normalised error of a reference. This treatise argues for a stricter criterion, which we call
**reference equivalence**: every floating-point result of a kernel must equal, bit for bit, the result of a
designated reference binary (here, ggml as shipped in llama.cpp release b11192), on every platform the engine runs on.
We show that this criterion is attainable for 1-bit (Q1_0) and ternary (Q2_0) weight formats on four execution
models — x86 AVX2, WebAssembly SIMD (with and without relaxed SIMD), Vulkan/SPIR-V and WebGPU/WGSL — at speeds at or
above the reference's. The method treats the reference's floating-point operation order as part of the
specification, and confines optimisation to operations that are exact by construction: integer reductions, products
whose operands' significands provably fit, and lane decompositions the reference already contains. Where a target
lacks an exact fused multiply-add, we emulate one with the Boldo–Melquiond round-to-odd construction, and we report a
compiler hazard on a production GPU driver that silently defeated the error-free transformation it depends on. The
evidence is exhaustive where the input space allows, and otherwise randomised plus real-tensor oracles. All of it is
reproducible from the repository.

---

## I. The problem

### I.1 Why bit equality, and not tolerance

Let *R* be a reference implementation of a function *f* over IEEE 754 binary32 (IEEE 2019), and *K* a candidate
kernel. The conventional acceptance test asks whether ‖K(x) − R(x)‖ / ‖R(x)‖ < ε for sampled *x*. llama.cpp's own
backend tests take this form, with a normalised mean-squared-error threshold per operation. This is sound engineering
for an engine whose outputs are consumed only by a human reader. It is insufficient for an engine whose outputs are
**attested**. bankML puts a receipt on every answer: hashes of the request, the model and the generated text. Its
claim is that a third party who replays the request against the reference obtains the same text. In autoregressive
decoding a difference in the last bit of one logit can change an argmax, and every later token is then conditioned
on a different prefix. Tolerance-based equivalence therefore does not compose over a generation, and bitwise
equivalence does: equal logits at every step give equal tokens at every step, under any deterministic sampler and any
seeded stochastic sampler.

### I.2 Why this is hard

Floating-point addition is not associative (Goldberg 1991; Higham 2002, ch. 2). A kernel's result is therefore a
function of its *operation order*, not only of its mathematical definition, and the operation order is what every
optimisation changes: vectorisation, threading, tiling, the GPU's parallel reduction. The fused multiply-add rounds
once where a multiply followed by an add rounds twice, so whether a compiler contracts `a*b + c` is itself part of the
result. The reference here is a *binary*, not a source file. bankML found that a faithful port of ggml's generic C
code disagreed with the shipped library in the last bit, because GCC had contracted the C expression into
`vfmadd231ss` under its default floating-point contraction ([modules/q1_0.md](modules/q1_0.md)). The specification
is the instruction stream.

### I.3 Definitions

- **Reference equivalence.** *K* is reference-equivalent to *R* on domain *D* when `bits(K(x)) = bits(R(x))` for all
  *x* in *D*, NaN payloads excepted. *D* is stated per kernel.
- **Exact operation.** A floating-point operation whose rounding error is zero on its whole domain of use, so its
  result is the real-number result. Exact operations may be reordered, fused or split freely; inexact ones may not.
- **Order-defining operation.** An inexact operation of *R*; *K* must perform it on the same operands, in the same
  dependency order, with the same rounding.
- **Q1_0.** ggml type 41: blocks of 128 weights, an f16 scale *d₀* and 16 bytes of signs (bit *j* of byte *j/8*,
  LSB first; 1 → +*d₀*, 0 → −*d₀*), 18 bytes in all.
- **q8_0.** The activation format: blocks of 32 int8 quants with an f16 scale *d₁*, 34 bytes, produced by ggml's AVX2
  quantiser (*id* = 127/amax, round half to even).
- **Q2_0_g64.** Ternary: blocks of 64 two-bit codes *c* ∈ {0, 1, 2} meaning *c* − 1, with an f16 scale.

---

## II. The literature, and where this work sits

### II.1 Reproducible and accurate summation

The most developed school on order-dependence aims at **order-independence**. Demmel and Nguyen's reproducible
summation (Demmel & Nguyen 2015) pre-rounds summands to a common grid so that any order gives the same bits.
Collange, Defour, Graillat and Iakymchuk (2015) combine error-free transformations with a long accumulator in the
tradition of Kulisch (2013) to make parallel reductions both reproducible and correctly rounded. Ogita, Rump and Oishi
(2005) give dot products as accurate as if computed in twice the working precision, from error-free transformations
alone. In its strongest form this school says that the right target is a result independent of the machine, and
ideally the correctly rounded one. Any particular order is an accident of implementation and should not be enshrined.

We accept that argument where the reference is ours to choose. It does not settle the case where the reference
already exists and is deployed. A correctly rounded dot product is *more* accurate than ggml's AVX2 kernel and for
that very reason *different* from it: the ReproBLAS result would not reproduce llama-server's answers. The present
work therefore takes the complementary position, **reference conformance**: we reproduce a fixed, deployed, imperfect
order everywhere, so that the deployed system's outputs become checkable. The two positions compose. A future
reference built on reproducible summation would make conformance cheaper, because every order would then conform.

### II.2 Non-determinism in inference

That deployed inference is non-deterministic is well documented. NVIDIA's cuBLAS documentation warns that results are
not guaranteed bitwise reproducible across GPU architectures, or for some routines across runs (NVIDIA n.d.). He
(2025) traces much of the run-to-run variation of served LLMs to batch-size-dependent reduction orders, and argues for
*batch-invariant* kernels. The present work shares that diagnosis and widens the invariance demanded. bankML's kernels
are invariant to batch, to thread count (the thread pool's results do not depend on its size; [modules/par.md](modules/par.md)),
to instruction set (scalar, AVX2, WebAssembly SIMD, relaxed SIMD), and to processor type (CPU, Vulkan GPU, WebGPU).

### II.3 Low-bit language models

1-bit and ternary weights entered mainstream language modelling with BitNet (Wang et al. 2023) and the 1.58-bit
variant (Ma et al. 2024). Post-training quantisation such as GPTQ (Frantar et al. 2023), and ggml's block formats,
made low-bit inference practical on commodity hardware. This literature asks how much model quality survives
quantisation. It does not ask whether two implementations of the same quantised model agree. Our weights are
prism-ml's Bonsai models in ggml's Q1_0 and Q2_0 formats, and our question is purely the second one.

### II.4 Error-free transformations

The tools are classical. Møller (1965) and Knuth (1998, §4.2.2) give **TwoSum**: for binary floating-point *a*, *b*,
the pair *(s, e)* with *s* = fl(*a* + *b*) and *a* + *b* = *s* + *e* exactly, computed in six operations without
branches. Dekker (1971) gives the exact product by **Veltkamp splitting**. Kahan's compensated summation (Kahan 1965)
is the first use of such an error term. Boldo and Melquiond (2008) show that **rounding to odd** makes double
rounding innocuous, and use it to emulate a correctly rounded FMA from TwoSum and an exact product. The systematic
treatment is Muller et al. (2018). These results all assume that the arithmetic performed is the arithmetic written.
§III.6 records a production compiler for which that assumption failed.

### II.5 Performance models

We use the roofline model (Williams, Waterman & Patterson 2009) to classify each kernel as compute- or
memory-bound, and register-level tiling in the sense of Goto and van de Geijn (2008) as the main lever for prompt
processing, a matrix–matrix product. Our variation on Goto–van de Geijn is that a tile is admissible only if each
output's accumulation chain is unchanged by it.

---

## III. The contribution: kernels designed from their exact operations outward

The design rule throughout: **identify the reference's order-defining operations; reproduce them exactly; reorganise
everything else.**

### III.1 The reference's order (Q1_0 · q8_0)

ggml's AVX2 `ggml_vec_dot_q1_0_q8_0`, modelled by bankML's scalar `vec_dot_ref` ([modules/q1_0.md](modules/q1_0.md)),
computes for an *n*-vector:

1. eight f32 accumulators *acc*₀…*acc*₇ (the eight lanes of a 256-bit register), zero;
2. for each Q1_0 block *i* (128 weights) and each of its four q8_0 sub-blocks *k* (32 activations), for each lane *l*:
   the integer *s*ₖ,ₗ = Σ ±q over elements 4*l* … 4*l*+3, the signs taken from the weight bits, with ggml's int8
   negation wrapping −(−128) to −128;
3. *ab*ₗ = *d₁*⁽⁰⁾·*s*₀,ₗ, then *ab*ₗ = fma(*d₁*⁽ᵏ⁾, *s*ₖ,ₗ, *ab*ₗ) for *k* = 1, 2, 3;
4. *acc*ₗ = fma(*d₀*⁽ⁱ⁾, *ab*ₗ, *acc*ₗ);
5. finally `hsum_float_8`: ((*acc*₀ + *acc*₄) + (*acc*₂ + *acc*₆)) + ((*acc*₁ + *acc*₅) + (*acc*₃ + *acc*₇)).

### III.2 Three lemmas that free most of the work

**Lemma 1 (integer freedom).** Step 2 is integer arithmetic on bounded values, |*s*ₖ,ₗ| ≤ 4·128 = 512. It may be
computed in any order, width or instruction sequence that respects the wrap of −128. *Use:* AVX2 sign expansion with
`pshufb`, AND and compare, then `(q ⊕ m) − m`; the selection form Σ±q = 2·Σ₊q − Σq with Σq precomputed per activation
(llama.cpp's "sel" kernel); WebAssembly's `relaxed_dot_i8x16_i7x16` against weight bits as {0, 1}, deterministic on
every implementation because the second operand never sets its top bit. With ±1 it is not deterministic: V8 on x86
reads that operand as unsigned.

**Lemma 2 (inner exactness).** *d₁* is an f16, so its significand has at most 11 bits. |*s*ₖ,ₗ| ≤ 512 has at most 10.
Their product has at most 21 significant bits and is exact in binary32 (24), barring underflow, which f16
exponents cannot reach. **Corollary:** in step 3, fma(*d₁*, *s*, *ab*) = RN(*d₁s* + *ab*) = RN(fl(*d₁s*) + *ab*), because
fl(*d₁s*) = *d₁s*. The fused and unfused forms round once at the same point, whatever the size of *ab*, so a target
without a deterministic SIMD FMA may use a multiply then an add there. *This does not transfer to Q2_0:*
its per-32 sums satisfy |*s*| < 2¹⁴, so *d₁*·*s* may need 25 bits, and the GPU ternary kernel must use the emulated
FMA in its inner step as well.

**Lemma 3 (lane independence).** The eight chains *acc*ₗ are independent until step 5. Assigning one chain to each
of eight GPU invocations, then letting invocation 0 perform step 5 in the reference's bracket order, is
reference-equivalent. Splitting across *blocks* is not, because it would reassociate step 4.

The only order-defining operations left are the outer fma of step 4, once per block per lane, and the fixed
horizontal sum of step 5. Each target's problem reduces to performing *those* exactly.

### III.3 x86-64 AVX2

The FMA is native, and contraction is pinned by writing the intrinsics explicitly. On the laptop measured (Ryzen 3
3200U), whole-token 1-bit matrix–vector products run at 0.93–1.13× of ggml's own time, and the prefill tile at 1.27–1.33×.
The ternary kernel ggml lacks on x86 (its scalar fallback) runs at 9.1–10.6× the reference's speed with the
reference's bits ([modules/q2_0.md](modules/q2_0.md)). It reorders only integer sums (Lemma 1) and keeps ggml's float
chain `sumi = fma(d1₁, s₁, fma(d1₀, s₀, 0))`, `sumf = fma(d0, sumi, sumf)`. The oracle distinguishes orders: two
ggml builds (haswell and the baseline x64) disagree with each other on 19 of 762 real-tensor dot products, and bankML
agrees with the haswell build on 762 of 762.

### III.4 WebAssembly SIMD: an exact FMA from f64

WebAssembly SIMD has no deterministic FMA (Haas et al. 2017 defines the deterministic core; FMA arrived only in the
relaxed proposal, whose semantics explicitly permit either rounding). For step 4, bankML evaluates *d₀*·*ab*ₗ + *acc*ₗ
in f64 SIMD. The product of two binary32 values is exact in binary64 (48 ≤ 53 significand bits), so one rounding of
an exact f64 sum to f32 is the fused result. Exactness of the f64 sum is checked with TwoSum's error term. Two cases
remain rare: lanes whose f64 sum is inexact *and* lands on an f32 rounding midpoint, where rounding twice errs, and
subnormal or non-finite lanes. Both fall back to the scalar `mul_add`. The scalar `fmaf` would have cost a third of
the kernel's time. The midpoint case is not hypothetical: the test suite constructs it, and with the fallback removed
the test fails (`tools/build.sh test`, about a million lanes).

### III.5 Relaxed SIMD, admitted by probe

`f32x4.relaxed_madd` may be fused or unfused, at the implementation's choice. Fused, it *is* the reference's FMA. The
engine therefore probes the machine at open. (1 + 2⁻²³)(1 − 2⁻²³) − 1 equals −2⁻⁴⁶ when fused, and 0 when the product
is rounded first. The relaxed build refuses to open where the probe reports 0, and the page then loads the plain
build. Combined with Lemma 1's selection form, relaxed SIMD bought +51% reading and +33% writing at four threads in
the browser, with identical answers (browseML `docs/efficiency.md`).

### III.6 SPIR-V and WGSL: an emulated FMA, and a compiler that rewrote it

On GPUs, bankML writes its own SPIR-V ([modules/gpu.md](modules/gpu.md)). SPIR-V can mark operations
`NoContraction`. Whether a driver honours a requested fused operation varies: RADV did not fuse `Fma` as the CPU does.
The verify step therefore checks before trusting, and the kernel computes step 4 by emulation. WGSL has neither an
exact `fma` (its `fma()` may round twice) nor `NoContraction`. bankML's WGSL (`gpu/wgsl.rs`) emulates fma(*a*, *b*, *c*)
as follows.

1. *Exact product.* Split *b* = *b*ₕ + *b*ₗ, with *b*ₕ its top 12 significand bits (a mask). *a* = *d₀* is an f16
   (≤ 11 bits), so *a*·*b*ₕ and *a*·*b*ₗ are both exact in binary32. TwoSum gives (*u*ₕ, *u*ₗ) with
   *u*ₕ + *u*ₗ = *ab* exactly. This is a specialised Veltkamp–Dekker product (Dekker 1971). It is exact *because* of
   the operand's provenance, and the kernel would be wrong for a full-precision *a*.
2. *Boldo–Melquiond.* (*t*ₕ, *t*ₗ) = TwoSum(*c*, *u*ₕ); *v* = RO(*t*ₗ + *u*ₗ), computed as TwoSum followed by setting
   the last bit when the error is non-zero and the significand is even; the result is RN(*t*ₕ + *v*). Boldo and
   Melquiond (2008) prove that this composition yields the correctly rounded fused result, barring underflow and
   overflow.

**The hazard.** On an AMD Radeon Vega 3 under Mesa 23.2 (RADV, through Chrome's Dawn), this emulation returned wrong
bits. Mesa's compiler had simplified TwoSum's `(a + b) − a` to `b`, an identity in the reals and false in floating
point, which destroys the error term. WGSL provides no way to forbid this. The countermeasure, `hide`, passes each
intermediate through a bitwise OR with a uniform the host sets to zero. The value is unchanged, but the compiler
cannot prove that, so it can no longer apply the identity. With `hide`, every result on every tested shape is
bit-exact on that card (browseML `testing/gpu.mjs`). The general lesson for error-free transformations on
JIT-compiled shading languages: **the arithmetic written is not the arithmetic performed until a test on the target
hardware says so.** bankML acts on this lesson with an *on-card oracle*: before a card is given any rows, layer-shaped
products are computed on it and compared bit for bit with the CPU kernel. A card that fails is never used.

### III.7 Parallel shape on the GPU

By Lemma 3, eight invocations per output row compute the eight lane chains, and invocation 0 performs step 5 from
workgroup memory in the reference's bracket order. This is 6.5× the one-invocation-per-row kernel on the Vega 3, with
identical bits. A second dispatch dimension carries a prompt's tokens (`q1_0_mat_mul8` in SPIR-V; `gid.y` in WGSL),
so a micro-batch is one dispatch. The workgroup size follows the vendor's SIMD width (64 for AMD wavefronts, 32 for
NVIDIA warps). Any multiple of 8 gives the same bits, as checked at 32, 64 and 128.

**Token tiling** (`Q1_0_MM4`, 2026-10-14). Each invocation computes *T* tokens, decoding each weight word once for
all *T*; each token's chain is untouched, so the tile is admissible by construction, and every result was
bit-identical to the untiled kernel, partial tiles included. Interleaved medians of five rounds on the Vega 3 through
Dawn chose *T* = 4: 1.00–1.18× from 4 to 512 tokens (1.14–1.18× at 64 and 512, the sizes prompt reading sends), a
geometric mean of 1.09–1.10× over the shapes and token counts measured. *T* = 8 and 16 lost on short batches (0.97×
and 0.78× geometric means) as the invocations of a partial tile compute clamped duplicate columns; a guard that skips
them helped short batches but cost long ones. Batches of two tokens lose at any *T* and stay on the CPU. The whole
engine's prompt reading, with the card taking ~38% of the rows, moved within the measurement noise under load.

### III.8 Transcendentals: porting the reference's libm

Rotary position embeddings need sin and cos, and the reference takes them from glibc. Rust's WebAssembly libm
disagreed with glibc in the last bit on 1,987 cosines and 2,917 sines among the 262,144 angles of positions 0–4095.
That was enough to part one of nine oracle conversations from the reference late in an answer. bankML ports glibc's
algorithm (ARM's optimized-routines, which glibc adopted) and matches glibc's x86-64 FMA build on **all 2³² binary32
inputs** (`sincosf_matches_glibc_on_every_float`): an exhaustive proof, not a sample.

---

## IV. Evidence and testable propositions

| proposition | evidence | kind |
|---|---|---|
| f16 ↔ f32 conversions equal the hardware's | all 65,536 f16 values; 2,000,000 random f32 against F16C | exhaustive / sampled |
| sin, cos equal glibc's | all 2³² inputs | exhaustive |
| Q1_0 AVX2 = reference | 3,500 randomised cases incl. q = −128; Bonsai-1.7B: 197 tensors, 1,719,904,256 weights, 788/788 dot products; Bonsai-8B: 762/762 | randomised + real tensors |
| Q2_0 AVX2 = reference | 3,300 randomised cases; 254 tensors, 8,188,239,872 weights, 762/762 | randomised + real tensors |
| WebAssembly f64 FMA route = `mul_add` | ~10⁶ lanes including constructed midpoints; fails without the fallback | randomised, adversarial |
| relaxed build = reference | probe at open + the oracle's 9 conversations on 2 threads | system |
| WGSL (with `hide`) = CPU on AMD | every result of 4 shapes × up to 8 tokens; workgroups 32/64/128 | on-card |
| SPIR-V = CPU on AMD | `bankml gpu --verify`: mat_vec, mat_vec8, mat_mul8, q2_0, layer-shaped | on-card |
| whole model = llama-server b11192 | 9 of 9 conversations token-identical, every release (browseML `testing/release.sh`) | end to end |

Propositions that remain open and are falsifiable: (P1) the `hide` countermeasure suffices on NVIDIA and Intel
compilers. It has not been tested on either. (P2) The emulated FMA's correctness holds on every input the forward pass
produces. It follows from Boldo and Melquiond's theorem plus the operand-provenance condition of §III.6, but neither
the condition nor its use here is machine-checked. (P3) Token tiling's gain survives on discrete cards, where the
submit overhead is smaller relative to the arithmetic.

---

## V. Objections, answered

**"The reference is arbitrary; you have enshrined one compiler's rounding."** Yes, deliberately. The deployed system
is the reference that matters to anyone checking an answer. Reproducing it turns llama-server's outputs into a
public, replayable standard. Choosing a better reference (§II.1) is compatible with this method and would only change
the target.

**"Exactness costs speed."** The measurements say otherwise. The kernels run at parity or above the reference on x86
(and 9–10× it for ternary), and exactness on WebAssembly and WebGPU came *with* the large speed-ups, not against them.
The lemmas show why: most of the work in a low-bit dot product is integer, and integer work is free to reorganise.

**"Testing is not proof."** Agreed. Where the domain is small enough we test it entirely (f16, sincosf). Elsewhere we
rely on lemmas whose premises are checked, randomised adversarial tests, real-tensor oracles and end-to-end oracles.
Machine-checked proofs, as Boldo and Melquiond gave for round-to-odd, are not attempted here. That is the obvious next
step for the emulated FMA, together with the operand-provenance condition of §III.6.

**"Edge cases."** Stated: NaN payloads are excluded from equivalence. On the GPU an infinite activation scale yields
NaN through the emulated FMA where the CPU yields ∞, so such products stay on the CPU. No real activation has one.
Subnormals take the scalar path on WebAssembly.

---

## References

Boldo, S. & Melquiond, G. (2008). "Emulation of FMA and correctly rounded sums: proved algorithms using rounding to
odd." *IEEE Transactions on Computers* 57(4): 462–471.

Collange, S., Defour, D., Graillat, S. & Iakymchuk, R. (2015). "Numerical reproducibility for the parallel reduction
on multi- and many-core architectures." *Parallel Computing* 49: 83–97.

Dekker, T. J. (1971). "A floating-point technique for extending the available precision." *Numerische Mathematik*
18(3): 224–242.

Demmel, J. & Nguyen, H. D. (2015). "Parallel reproducible summation." *IEEE Transactions on Computers* 64(7):
2060–2070.

Frantar, E., Ashkboos, S., Hoefler, T. & Alistarh, D. (2023). "GPTQ: Accurate post-training quantization for
generative pre-trained transformers." *International Conference on Learning Representations (ICLR)*.

Goldberg, D. (1991). "What every computer scientist should know about floating-point arithmetic." *ACM Computing
Surveys* 23(1): 5–48.

Goto, K. & van de Geijn, R. A. (2008). "Anatomy of high-performance matrix multiplication." *ACM Transactions on
Mathematical Software* 34(3): article 12.

Haas, A., Rossberg, A., Schuff, D. L., Titzer, B. L., Holman, M., Gohman, D., Wagner, L., Zakai, A. & Bastien, J. F.
(2017). "Bringing the web up to speed with WebAssembly." *Proceedings of the 38th ACM SIGPLAN Conference on
Programming Language Design and Implementation (PLDI)*.

He, H. (2025). "Defeating nondeterminism in LLM inference." Thinking Machines Lab, *Connectionism* (blog).

Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. Philadelphia: SIAM.

IEEE (2019). *IEEE Standard for Floating-Point Arithmetic*, IEEE Std 754-2019.

Kahan, W. (1965). "Further remarks on reducing truncation errors." *Communications of the ACM* 8(1): 40.

Knuth, D. E. (1998). *The Art of Computer Programming, Vol. 2: Seminumerical Algorithms*, 3rd ed. Reading, MA:
Addison-Wesley.

Kulisch, U. (2013). *Computer Arithmetic and Validity: Theory, Implementation, and Applications*, 2nd ed. Berlin:
De Gruyter.

Ma, S., Wang, H., Ma, L., Wang, L., Wang, W., Huang, S., Dong, L., Wang, R., Xue, J. & Wei, F. (2024). "The era of
1-bit LLMs: All large language models are in 1.58 bits." arXiv:2402.17764.

Møller, O. (1965). "Quasi double-precision in floating point addition." *BIT Numerical Mathematics* 5: 37–50.

Muller, J.-M., Brunie, N., de Dinechin, F., Jeannerod, C.-P., Joldes, M., Lefèvre, V., Melquiond, G., Revol, N. &
Torres, S. (2018). *Handbook of Floating-Point Arithmetic*, 2nd ed. Cham: Birkhäuser.

NVIDIA (n.d.). *cuBLAS Library documentation*, section "Results reproducibility."

Ogita, T., Rump, S. M. & Oishi, S. (2005). "Accurate sum and dot product." *SIAM Journal on Scientific Computing*
26(6): 1955–1988.

Wang, H., Ma, S., Dong, L., Huang, S., Wang, H., Ma, L., Yang, F., Wang, R., Wu, Y. & Wei, F. (2023). "BitNet:
Scaling 1-bit transformers for large language models." arXiv:2310.11453.

Williams, S., Waterman, A. & Patterson, D. (2009). "Roofline: an insightful visual performance model for multicore
architectures." *Communications of the ACM* 52(4): 65–76.

**Primary sources in this repository:** `bankML/q1_0.rs` (`vec_dot_ref`, `simd128_core`, `relaxed_core`,
`relaxed_madd_is_fused`), `bankML/q2_0.rs`, `bankML/wasm_simd.rs`, `bankML/sincosf.rs`, `bankML/gpu/kernels.rs`,
`bankML/gpu/spirv.rs`, `bankML/gpu/wgsl.rs`; [modules/q1_0.md](modules/q1_0.md), [modules/q2_0.md](modules/q2_0.md),
[modules/gpu.md](modules/gpu.md), [oracles.md](oracles.md), [PERFORMANCE.md](PERFORMANCE.md); browseML
`testing/gpu.mjs`, `testing/release.sh`, `docs/technical.md`, `docs/efficiency.md`.
