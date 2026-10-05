// SPDX-License-Identifier: MIT OR Apache-2.0
// The std::sort oracle (0.3.7): libstdc++'s own std::sort (introsort) on index arrays keyed by floats with many
// ties, as llama.cpp's typical-p sorts its shifted scores. std::sort is not stable, so the order equal keys end in is
// the algorithm's own; bankML's port (sampler::sort_by) must leave every array exactly as this does.
// build: g++ -O2 -o sort_oracle testing/sort_oracle.cpp   run: ./sort_oracle > .models/oracle-sort/cases.txt
// format: one case per line — n, the n keys (as hex float bits), then the n sorted indices.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint64_t s = 0x9E3779B97F4A7C15ull;
static uint32_t rnd() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 11); }

int main() {
    std::vector<int> sizes;
    for (int n = 0; n <= 140; ++n) sizes.push_back(n);
    for (int n : {150, 200, 256, 300, 1000}) sizes.push_back(n);
    for (int n : sizes) {
        for (int shape = 0; shape < 6; ++shape) {
            std::vector<float> key(n);
            for (int i = 0; i < n; ++i) {
                switch (shape) {
                    case 0: key[i] = (float)(rnd() % 4); break;                 // heavy ties
                    case 1: key[i] = (float)(rnd() % 1000) / 7.0f; break;       // few ties
                    case 2: key[i] = (float)i; break;                           // sorted
                    case 3: key[i] = (float)(n - i); break;                     // reversed
                    case 4: key[i] = 1.0f; break;                               // all equal
                    default: key[i] = (float)((i * 7919) % 13) * 0.25f; break;  // periodic
                }
            }
            std::vector<size_t> idx(n);
            for (int i = 0; i < n; ++i) idx[i] = i;
            std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return key[a] < key[b]; });
            printf("%d", n);
            for (float k : key) { uint32_t b; std::memcpy(&b, &k, 4); printf(" %08x", b); }
            for (size_t i : idx) printf(" %zu", i);
            printf("\n");
        }
    }
}
