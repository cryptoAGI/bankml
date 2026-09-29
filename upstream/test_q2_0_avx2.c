// SPDX-License-Identifier: MIT
// Bit-exactness and speed of q2_0_avx2.c against the SHIPPED ggml_vec_dot_q2_0_q8_0 (dlopen of the llama.cpp b11192
// release's libggml-cpu-haswell.so): random blocks (every code 0..3, random scales, extreme activations), many
// lengths; every result compared by its bits. Build: cc -O2 -mavx2 -mfma -mf16c q2_0_avx2.c test_q2_0_avx2.c -ldl -o t
//   ./t /path/to/llama-b11192
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
typedef void (*vec_dot_t)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
void ggml_vec_dot_q2_0_q8_0_avx2(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
static uint64_t st = 0x9e3779b97f4a7c15ull;
static uint64_t rnd(void) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s LLAMA_B11192_DIR\n", argv[0]); return 2; }
    char p[4096];
    snprintf(p, sizeof p, "%s/libggml-base.so", argv[1]);
    if (!dlopen(p, RTLD_NOW | RTLD_GLOBAL)) { fprintf(stderr, "%s\n", dlerror()); return 2; }
    snprintf(p, sizeof p, "%s/libggml-cpu-haswell.so", argv[1]);
    void *h = dlopen(p, RTLD_NOW | RTLD_GLOBAL);
    if (!h) { fprintf(stderr, "%s\n", dlerror()); return 2; }
    ((void (*)(void))dlsym(h, "ggml_cpu_init"))();
    vec_dot_t ref = (vec_dot_t)dlsym(h, "ggml_vec_dot_q2_0_q8_0");
    const int maxn = 4096;
    uint8_t *x = malloc(maxn / 64 * 18), *y = malloc(maxn / 32 * 34);
    long cases = 0, bad = 0;
    for (int it = 0; it < 200000; it++) {
        int n = 64 * (1 + rnd() % (maxn / 64));
        for (int b = 0; b < n / 64; b++) {
            uint16_t d = _cvtss_sh((float)((int)(rnd() % 2001) - 1000) / 7919.0f, 0);
            memcpy(x + 18 * b, &d, 2);
            for (int j = 0; j < 16; j++) x[18 * b + 2 + j] = (uint8_t)rnd();   // every code 0..3, including +2
        }
        for (int b = 0; b < n / 32; b++) {
            uint16_t d = _cvtss_sh((float)(rnd() % 100000) / 1e6f, 0);
            memcpy(y + 34 * b, &d, 2);
            for (int j = 0; j < 32; j++) y[34 * b + 2 + j] = (it % 7 == 0) ? (uint8_t)((rnd() & 1) ? 127 : 0x81) : (uint8_t)rnd();
        }
        float a, r;
        ggml_vec_dot_q2_0_q8_0_avx2(n, &a, 0, x, 0, y, 0, 1);
        ref(n, &r, 0, x, 0, y, 0, 1);
        uint32_t ab, rb;
        memcpy(&ab, &a, 4); memcpy(&rb, &r, 4);
        cases++;
        if (ab != rb) { if (bad++ < 5) fprintf(stderr, "n=%d: avx2 %.9g (%08x) != ggml %.9g (%08x)\n", n, a, ab, r, rb); }
    }
    printf("bit-exact against the shipped ggml_vec_dot_q2_0_q8_0: %ld of %ld random cases\n", cases - bad, cases);
    // speed: one 4096-wide row, many times
    int n = 4096; float s1 = 0, s2 = 0, t;
    double t0 = now(); for (int k = 0; k < 200000; k++) { ref(n, &t, 0, x, 0, y, 0, 1); s1 += t; } double tr = now() - t0;
    t0 = now(); for (int k = 0; k < 200000; k++) { ggml_vec_dot_q2_0_q8_0_avx2(n, &t, 0, x, 0, y, 0, 1); s2 += t; } double ta = now() - t0;
    printf("4096-wide row, 200,000 times: ggml (shipped, scalar) %.3f s · this kernel %.3f s · %.1fx (%s)\n", tr, ta, tr / ta, s1 == s2 ? "same sums" : "SUMS DIFFER");
    return bad != 0;
}
