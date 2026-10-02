// SPDX-License-Identifier: MIT OR Apache-2.0
// The grammar oracle's other half (0.3.3): llama.cpp b11192's own grammar sampler, driven through libllama's public C
// API (llama_sampler_init_grammar / _apply / _accept), on the pinned model's vocabulary (loaded vocab-only). Built and
// driven by testing/grammar_oracle.py; reads commands on stdin, one per line, text arguments hex-encoded:
//   pieces          every token's piece (llama_token_to_piece, special = true) and whether it ends generation
//   tokenize HEX    llama_tokenize(text, add_special = false, parse_special = true)
//   grammar HEX     a fresh grammar sampler (root "root"): "ok" or "fail"
//   prefill IDS     accept these tokens without printing masks
//   run IDS         before each token and after the last: the mask over the whole vocabulary ("m <allowed> <fnv64>"),
//                   then accept the token; "x" (and stop) if llama.cpp throws on it (or, for an end-of-generation
//                   token, would abort: one the grammar's own apply rejects)
// The mask is what llama_sampler_apply leaves above -INFINITY when every logit is 0: allowed ids, ascending, hashed
// with 64-bit FNV-1a over their little-endian u32 bytes.
#include "llama.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::string unhex(const std::string & h) {
    std::string s;
    for (size_t i = 0; i + 1 < h.size(); i += 2) s.push_back((char) std::stoi(h.substr(i, 2), nullptr, 16));
    return s;
}

static std::string hex(const std::string & s) {
    static const char * d = "0123456789abcdef";
    std::string o;
    for (unsigned char c : s) { o.push_back(d[c >> 4]); o.push_back(d[c & 15]); }
    return o;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: grammar_oracle MODEL.gguf < commands\n"); return 2; }
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_sampler * g = nullptr;
    std::vector<llama_token_data> cur(n_vocab);
    auto mask = [&]() {
        for (int i = 0; i < n_vocab; i++) cur[i] = {i, 0.0f, 0.0f};
        llama_token_data_array a = {cur.data(), (size_t) n_vocab, -1, false};
        llama_sampler_apply(g, &a);
        uint64_t h = 1469598103934665603ULL;
        long n = 0;
        for (size_t i = 0; i < a.size; i++) {
            if (a.data[i].logit == -INFINITY) continue;
            n++;
            uint32_t id = (uint32_t) a.data[i].id;
            for (int b = 0; b < 4; b++) { h ^= (id >> (8 * b)) & 0xff; h *= 1099511628211ULL; }
        }
        printf("m %ld %llu\n", n, (unsigned long long) h);
    };
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        if (cmd == "pieces") {
            std::vector<char> buf(512);
            for (int i = 0; i < n_vocab; i++) {
                int n = llama_token_to_piece(vocab, i, buf.data(), (int) buf.size(), 0, true);
                if (n < 0) { buf.resize(-n); n = llama_token_to_piece(vocab, i, buf.data(), (int) buf.size(), 0, true); }
                printf("%d %s %d\n", i, hex(std::string(buf.data(), n)).c_str(), llama_vocab_is_eog(vocab, i) ? 1 : 0);
            }
        } else if (cmd == "tokenize") {
            std::string h; in >> h;
            std::string t = unhex(h);
            std::vector<llama_token> ids(t.size() + 16);
            int n = llama_tokenize(vocab, t.data(), (int) t.size(), ids.data(), (int) ids.size(), false, true);
            for (int i = 0; i < n; i++) printf(i ? " %d" : "%d", ids[i]);
            printf("\n");
        } else if (cmd == "grammar") {
            std::string h; in >> h;
            if (g) llama_sampler_free(g);
            g = llama_sampler_init_grammar(vocab, unhex(h).c_str(), "root");
            printf(g ? "ok\n" : "fail\n");
        } else if (cmd == "prefill" || cmd == "run") {
            bool run = cmd == "run";
            llama_token id;
            bool dead = false;
            while (in >> id) {
                if (run) mask();
                if (llama_vocab_is_eog(vocab, id)) {
                    // llama.cpp aborts (it does not throw) on an end token the grammar cannot take: ask first
                    llama_token_data one = {id, 0.0f, 0.0f};
                    llama_token_data_array a1 = {&one, 1, -1, false};
                    llama_sampler_apply(g, &a1);
                    if (one.logit == -INFINITY) { printf("x\n"); dead = true; break; }
                }
                try {
                    llama_sampler_accept(g, id);
                } catch (const std::exception &) {
                    printf("x\n");
                    dead = true;
                    break;
                }
            }
            if (run && !dead) mask();
            printf("done\n");
        } else if (!cmd.empty()) {
            printf("? %s\n", cmd.c_str());
        }
        fflush(stdout);
    }
    if (g) llama_sampler_free(g);
    llama_model_free(model);
    return 0;
}
