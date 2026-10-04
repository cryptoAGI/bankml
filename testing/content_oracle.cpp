// SPDX-License-Identifier: MIT OR Apache-2.0
// The JSON-content oracle (0.3.5, testing/content_oracle.py): how llama-server b11192 turns a constrained answer's raw
// text into the message `content` — `common_chat_parse(raw, is_partial = false, …)` with the PEG chat parser that
// `common_chat_templates_apply` builds for the request (the template, `--jinja --reasoning off`, the JSON schema; an
// empty schema line means JSON mode, `{"type": "object"}`), exactly as `server_task_result_cmpl_final::update` calls
// it — inside the release's libllama-common.so, no model. Reads lines "<hex schema> <hex raw>"; prints
// "C <hex content>" or "E <hex error>" per line.
#include "chat.h"
#include "json.h"
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
    if (argc < 2) { fprintf(stderr, "usage: content_oracle TEMPLATE.jinja < lines\n"); return 2; }
    std::ifstream tf(argv[1]);
    std::stringstream ts; ts << tf.rdbuf();
    auto tmpls = common_chat_templates_init(nullptr, ts.str());
    std::string line;
    while (std::getline(std::cin, line)) {
        auto sp = line.find(' ');
        std::string schema = unhex(line.substr(0, sp)), raw = unhex(line.substr(sp + 1));
        try {
            common_chat_templates_inputs in;
            common_chat_msg m; m.role = "user"; m.content = "hi";
            in.messages = {m};
            in.json_schema = schema.empty() ? "{\"type\":\"object\"}" : schema;
            in.use_jinja = true;
            in.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            in.enable_thinking = false;
            in.chat_template_kwargs["enable_thinking"] = "false";
            auto p = common_chat_templates_apply(tmpls.get(), in);
            common_chat_parser_params pp(p);
            pp.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            if (!p.parser.empty()) pp.parser.load(p.parser);
            auto msg = common_chat_parse(raw, false, pp);
            printf("C %s\n", hex(msg.content).c_str());
        } catch (const std::exception & e) {
            printf("E %s\n", hex(e.what()).c_str());
        }
        fflush(stdout);
    }
}
