// SPDX-License-Identifier: MIT
// An AVX2 kernel for ggml's Q2_0 × Q8_0 dot product, offered to llama.cpp (ggml-org/llama.cpp, MIT).
// At b11192 x86 has no Q2_0 kernel: arch-fallback.h maps ggml_vec_dot_q2_0_q8_0 to the generic C, which GCC builds
// scalar (64 integer multiplies per block). This kernel returns the SAME BITS as the shipped haswell build of that
// generic C (which GCC contracts to FMAs): the two integer sums per block are exact in any order, and the float
// operations are the generic code's, in its order, with the fused multiply-adds made explicit.
//   sumi = fma(d1_0, s_0, 0); sumi = fma(d1_1, s_1, sumi); sumf = fma(d0, sumi, sumf)   — block after block
// Proven bit-exact against the shipped libggml-cpu-haswell.so (llama-b11192-bin-ubuntu-x64) by test_q2_0_avx2.c,
// and in bankml (github.com/cryptoAGI/bankml) on all 8,188,239,872 weights of Ternary-Bonsai-8B.
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define QK2_0 64
#define QK8_0 32
typedef struct { uint16_t d; uint8_t qs[QK2_0 / 4]; } block_q2_0;   // 18 bytes, as ggml-common.h
typedef struct { uint16_t d; int8_t qs[QK8_0]; } block_q8_0;         // 34 bytes

static inline float h2f(uint16_t h) { return _cvtsh_ss(h); }        // exact, as ggml's table/F16C conversion

// the Σ (c − 1)·y of one q8_0 block against the 8 code bytes that meet it
static inline int32_t dot32(const uint8_t *codes8, const int8_t *y) {
    uint32_t lo, hi;
    memcpy(&lo, codes8, 4);
    memcpy(&hi, codes8 + 4, 4);
    // lane L, dword r, byte j holds code r of byte 4L+j, i.e. element 16L + 4j + r
    const __m256i v = _mm256_setr_epi32(lo, lo, lo, lo, hi, hi, hi, hi);
    const __m256i c = _mm256_and_si256(_mm256_srlv_epi32(v, _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6)), _mm256_set1_epi8(3));
    // the activation, transposed 4×4 inside each 128-bit lane so that position 4r+j holds element 4j+r
    const __m256i t = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                       0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m256i q = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)y), t);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256i cy = _mm256_madd_epi16(_mm256_maddubs_epi16(c, q), ones16);               // Σ c·y   (c ∈ 0..3)
    __m256i sy = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(1), q), ones16); // Σ y
    __m256i d = _mm256_sub_epi32(cy, sy);                                            // Σ (c−1)·y, exact
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(d), _mm256_extracti128_si256(d, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

// the ggml signature: ggml_vec_dot_q2_0_q8_0(n, s, bs, vx, bx, vy, by, nrc), nrc == 1
void ggml_vec_dot_q2_0_q8_0_avx2(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    const block_q2_0 *x = (const block_q2_0 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)vy;
    const int nb = n / QK2_0;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d0 = h2f(x[i].d);
        float sumi = 0.0f;
        for (int k = 0; k < 2; k++) {
            const block_q8_0 *yb = &y[2 * i + k];
            sumi = fmaf(h2f(yb->d), (float)dot32(x[i].qs + 8 * k, yb->qs), sumi);
        }
        sumf = fmaf(d0, sumi, sumf);
    }
    *s = sumf;
}
