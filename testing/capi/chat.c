/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/*
 * The C side of the chat oracle (0.3.2): a program that embeds libbankml the way an application would.
 *   chat MODEL.gguf FORK.json N_CTX
 * Opens the model through bankml_open (the guard, the sha256 pin, the native check) and prints one JSON line:
 *   {"open": true, "version": "…"}   or   {"open": false, "error": "…"} (exit 2)
 * then answers one request per stdin line through bankml_chat, printing one JSON line per request:
 *   {"rc": 0, "pieces": N, "streamed": "<the pieces the callback received, joined>", "result": <result_json>}
 * The library's own log messages go through bankml_set_log to stderr, prefixed "[bankml log L] ".
 * testing/capi/chat_oracle.py drives it and compares it with `bankml serve --native` and llama-server's record.
 */
#include <bankml.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct buf {
  char *p;
  size_t n, cap;
};

static void put(struct buf *b, const char *s, size_t n) {
  if (b->n + n + 1 > b->cap) {
    b->cap = (b->n + n + 1) * 2;
    b->p = realloc(b->p, b->cap);
    if (!b->p) { fprintf(stderr, "out of memory\n"); exit(1); }
  }
  memcpy(b->p + b->n, s, n);
  b->n += n;
  b->p[b->n] = '\0';
}

static void json_string(FILE *f, const char *s, size_t n) {
  fputc('"', f);
  for (size_t i = 0; i < n; i++) {
    unsigned char c = (unsigned char)s[i];
    if (c == '"') fputs("\\\"", f);
    else if (c == '\\') fputs("\\\\", f);
    else if (c < 0x20) fprintf(f, "\\u%04x", c);
    else fputc(c, f);
  }
  fputc('"', f);
}

static size_t pieces;

static int on_piece(const char *piece, size_t len, void *user) {
  if (piece[len] != '\0') { fprintf(stderr, "piece not NUL-terminated\n"); exit(1); }
  put((struct buf *)user, piece, len);
  pieces++;
  return 1;
}

static void on_log(int level, const char *msg, size_t len, void *user) {
  (void)user;
  fprintf(stderr, "[bankml log %d] %.*s\n", level, (int)len, msg);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: chat MODEL.gguf FORK.json N_CTX < requests.jsonl\n");
    return 1;
  }
  bankml_set_log(on_log, NULL);
  bankml_log(BANKML_LOG_INFO, "chat oracle: libbankml %s, opening %s (context %s)", bankml_version(), argv[1], argv[3]);
  char *err = NULL;
  bankml_t *h = bankml_open(argv[1], argv[2], (uint32_t)strtoul(argv[3], NULL, 10), &err);
  if (!h) {
    printf("{\"open\": false, \"error\": ");
    json_string(stdout, err ? err : "", err ? strlen(err) : 0);
    printf("}\n");
    bankml_free(err);
    return 2;
  }
  printf("{\"open\": true, \"version\": \"%s\"}\n", bankml_version());
  fflush(stdout);
  struct buf line = {0}, streamed = {0};
  int c;
  for (;;) {
    line.n = 0;
    while ((c = getchar()) != EOF && c != '\n') {
      char ch = (char)c;
      put(&line, &ch, 1);
    }
    if (line.n == 0 && c == EOF) break;
    if (line.n == 0) continue;
    streamed.n = 0;
    put(&streamed, "", 0);
    pieces = 0;
    char *result = NULL;
    int rc = bankml_chat(h, line.p, on_piece, &streamed, &result);
    printf("{\"rc\": %d, \"pieces\": %zu, \"streamed\": ", rc, pieces);
    json_string(stdout, streamed.p, streamed.n);
    printf(", \"result\": %s}\n", result ? result : "null");
    fflush(stdout);
    bankml_free(result);
    if (c == EOF) break;
  }
  bankml_close(h);
  free(line.p);
  free(streamed.p);
  return 0;
}
