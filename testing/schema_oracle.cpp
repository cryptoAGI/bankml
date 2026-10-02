// SPDX-License-Identifier: MIT OR Apache-2.0
// The JSON-schema grammar oracle (O6b, testing/schema_oracle.py): llama.cpp b11192's own json_schema_to_grammar and its chat-template wrapping,
// called in libllama-common.so from the b11192 release — no model is loaded. Reads one JSON value per line (hex) on
// stdin; prints, per line: "R <hex grammar>" or "RE <hex error>" (json_schema_to_grammar), then "C <hex grammar>"
// "P <hex generation prompt>" or "CE <hex error>" (common_chat_templates_apply on the jinja path, the way
// oaicompat_chat_params_parse feeds it with Savante's flags: --jinja --reasoning off), then "." .
#include "chat.h"
#include "json.h"
#include "json-schema-to-grammar.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

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
    if (argc < 2) { fprintf(stderr, "usage: schema_oracle TEMPLATE.jinja < hex lines\n"); return 2; }
    std::ifstream tf(argv[1]);
    std::stringstream ts; ts << tf.rdbuf();
    auto tmpls = common_chat_templates_init(nullptr, ts.str());
    std::string line;
    while (std::getline(std::cin, line)) {
        std::string text = unhex(line);
        try {
            auto j = common_json::parse(text);
            printf("R %s\n", hex(json_schema_to_grammar(j, true)).c_str());
        } catch (const std::exception & e) {
            printf("RE %s\n", hex(e.what()).c_str());
        }
        try {
            auto j = common_json::parse(text);
            common_chat_templates_inputs in;
            common_chat_msg m; m.role = "user"; m.content = "hi";
            in.messages = {m};
            std::string js = j.is_null() ? "" : j.dump();
            if (j.is_object() && j.empty()) js = "{\"type\":\"object\"}";
            in.json_schema = js;
            in.use_jinja = true;
            in.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            in.enable_thinking = false;
            in.chat_template_kwargs["enable_thinking"] = "false";
            auto p = common_chat_templates_apply(tmpls.get(), in);
            printf("C %s\nP %s\n", hex(p.grammar).c_str(), hex(p.generation_prompt).c_str());
        } catch (const std::exception & e) {
            printf("CE %s\n", hex(e.what()).c_str());
        }
        printf(".\n");
        fflush(stdout);
    }
}
