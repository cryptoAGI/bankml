/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/*
 * The printf oracle for bankml_log (0.3.2): for every case, the message bankml_log delivers to the sink must equal,
 * byte for byte (length included, an embedded NUL included), what this machine's libc snprintf writes for the same
 * format and arguments. Then the cases bankml_log does not support: each must come out as its explicit marker, read
 * no argument it cannot type, and never write through %n.
 *   cc -O2 -Wall -Wextra -Wno-format -Werror printf_oracle.c -I capi/include -L target/release -lbankml
 * (-Wno-format: some cases are deliberately unusual formats.) Exit 0 only if every case passes.
 */
#include <bankml.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>

static char got[1 << 16];
static size_t got_len;
static int got_level, calls, total, failed;

static void sink(int level, const char *msg, size_t len, void *user) {
  (void)user;
  if (len >= sizeof got) len = sizeof got - 1;
  memcpy(got, msg, len);
  got_len = len;
  got_level = level;
  calls++;
  if (msg[len] != '\0') { printf("  sink: message not NUL-terminated\n"); failed++; }
}

static void show(const char *what, const char *s, size_t n) {
  printf("    %s (%zu bytes): \"", what, n);
  for (size_t i = 0; i < n; i++) {
    unsigned char c = (unsigned char)s[i];
    if (c < 32 || c == '"' || c == '\\' || c >= 127) printf("\\x%02x", c); else putchar(c);
  }
  printf("\"\n");
}

static void check(const char *label, const char *want, size_t want_len) {
  total++;
  if (got_level != BANKML_LOG_INFO || got_len != want_len || memcmp(got, want, want_len) != 0) {
    failed++;
    printf("  DIFFERS: %s\n", label);
    show("libc snprintf", want, want_len);
    show("bankml_log   ", got, got_len);
  }
}

/* the same format and arguments through snprintf and bankml_log */
#define SAME(...)                                                     \
  do {                                                                \
    static char want_[1 << 16];                                       \
    int n_ = snprintf(want_, sizeof want_, __VA_ARGS__);              \
    int before_ = calls;                                              \
    bankml_log(BANKML_LOG_INFO, __VA_ARGS__);                         \
    if (calls != before_ + 1) { failed++; printf("  no message: %s\n", #__VA_ARGS__); } \
    check(#__VA_ARGS__, want_, (size_t)n_);                           \
  } while (0)

/* a case bankml_log does not support: the exact marker text */
#define MARKED(want, ...)                                             \
  do {                                                                \
    bankml_log(BANKML_LOG_INFO, __VA_ARGS__);                         \
    check(#__VA_ARGS__, want, strlen(want));                          \
  } while (0)

int main(void) {
  bankml_set_log(sink, NULL);
  int local = 0;
  const char *s = "bankml";
  const char unterminated[2] = {'a', 'b'};

  /* integers, at their limits and in every supported length */
  SAME("%d|%d|%d|%d|%d", 0, 1, -1, INT_MIN, INT_MAX);
  SAME("%i|%i", INT_MIN, INT_MAX);
  SAME("%u|%u|%u", 0u, 1u, UINT_MAX);
  SAME("%ld|%ld|%ld", LONG_MIN, LONG_MAX, 0L);
  SAME("%lu|%lu", ULONG_MAX, 0UL);
  SAME("%lld|%lld|%lld", LLONG_MIN, LLONG_MAX, -1LL);
  SAME("%llu|%llu", ULLONG_MAX, 0ULL);
  SAME("%zu|%zu|%zd|%zd", SIZE_MAX, (size_t)0, (ssize_t)-1, (ssize_t)PTRDIFF_MAX);
  SAME("%hhd|%hhd|%hd|%hd|%hhu|%hu", 300, -129, 70000, -32769, 511u, 65537u);
  SAME("%x|%X|%x|%x|%X", 0xdeadbeefu, 0xdeadbeefu, 0u, UINT_MAX, 0xabcu);
  SAME("%#x|%#X|%#x", 255u, 255u, 0u);
  SAME("%lx|%llX|%zx|%hhx|%hx", ULONG_MAX, ULLONG_MAX, SIZE_MAX, 0x1ffu, 0x12345u);
  SAME("[%5d|%-5d|%05d|%+d|% d|%+ d|%+d]", 42, 42, -42, 7, 7, 7, -7);
  SAME("[%.3d|%.0d|%.0d|%8.3d|%-8.3d|%08.3d|%.10d]", 5, 0, 1, -5, -5, 5, INT_MIN);
  SAME("[%08x|%#010x|%-#10x|%#.6x|%.0x|%#.0x]", 0xbeefu, 0xbeefu, 0xbeefu, 0xbeefu, 0u, 0u);
  SAME("[%*d|%-*d|%*d|%.*d|%.*d|%*.*d]", 6, 42, 6, 42, -6, 42, 4, 7, -1, 7, 8, 5, -3);
  SAME("[%+u|% u|%+x|% X]", 3u, 3u, 255u, 255u);
  SAME("[%020lld|%-20llu|%+lld]", LLONG_MIN, ULLONG_MAX, LLONG_MAX);

  /* %f: precision, rounding (ties to even on the exact binary value), signs, flags, extremes, inf and nan */
  SAME("%f|%f|%f|%f", 0.0, -0.0, 1.0, -1.0);
  SAME("%f|%.3f|%.3f|%.1f", 3.14258, -2.0005, 2.0005, 0.25);
  SAME("%.0f|%.0f|%.0f|%.0f|%.0f|%.0f", 0.5, 1.5, 2.5, -0.5, 0.49999999999999994, 1e15 + 0.5);
  SAME("%.20f|%.17f|%.10f|%.30f", 0.1, 0.1 + 0.2, 1.0 / 3.0, 1e-20);
  SAME("%f|%f", 1e300, -1e300);
  SAME("%f|%.320f", DBL_MIN, DBL_MIN);
  SAME("%.1100f", 4.9406564584124654e-324);
  SAME("%f", DBL_MAX);
  SAME("[%#.0f|%#f|%#.2f]", 3.0, 3.0, 3.0);
  SAME("[%10.2f|%-10.2f|%010.2f|%+.1f|% .1f|%+010.3f|% 010.3f]", -1.005, 1.0, -3.25, 0.0, 0.0, 2.5, 2.5);
  SAME("[%*.*f|%-*.*f|%.*f]", 12, 4, 2.718281828, 12, 4, 2.718281828, -1, 2.718281828);
  SAME("[%lf|%F|%F]", 2.5, 1.25, -1.25);
  SAME("[%f|%f|%f|%f]", INFINITY, -INFINITY, NAN, -NAN);
  SAME("[%F|%F|%F|%F]", INFINITY, -INFINITY, NAN, -NAN);
  SAME("[%5f|%-6f|%05f|%+f|% f|%010F]", INFINITY, NAN, INFINITY, INFINITY, NAN, -INFINITY);
  SAME("%.3f|%.3f|%.3f", 1e-4, 9.9995, 0.0005);

  /* strings */
  SAME("[%s|%s|%10s|%-10s|%.3s|%.0s|%.10s]", s, "", s, s, s, s, s);
  SAME("[%.*s|%*s|%-*s|%*.*s]", 2, s, -9, s, 9, s, 8, 3, s);
  SAME("[%s|%.5s|%.6s]", "Sav\xc3\xa9 \xf0\x9f\x98\x80", "Sav\xc3\xa9 \xf0\x9f\x98\x80", "Sav\xc3\xa9 \xf0\x9f\x98\x80");
  SAME("[%.2s]", unterminated);                 /* a precision bounds the read: no NUL needed */
  SAME("[%s|%.3s|%.6s|%10s]", (char *)NULL, (char *)NULL, (char *)NULL, (char *)NULL);

  /* characters, an embedded NUL and a high byte included */
  SAME("[%c%c|%3c|%-3c|%c|%c]", 'o', 'k', 'x', 'y', 0, 255);

  /* pointers */
  SAME("[%p|%p|%p]", (void *)0x1234, (void *)NULL, (void *)&local);
  SAME("[%20p|%-20p|%8p|%-8p]", (void *)s, (void *)s, (void *)NULL, (void *)NULL);
  SAME("[%p]", (void *)UINTPTR_MAX);

  /* %% and a mix */
  SAME("100%%");
  SAME("%s=%d (%.2f%%) %c %zu %#x %p end", "turn", -3, 99.5, 'Z', (size_t)4096, 48879u, (void *)0xff);
  SAME("no conversions at all");
  SAME("%s", "");

  int supported = total, supported_failed = failed;

  /* what bankml_log does not support: an explicit marker, never a guess */
  MARKED("%<unsupported:e>", "%e", 2.0);
  MARKED("1 %<unsupported:g> %<skipped:d> %<skipped:s> % end", "%d %g %d %s %% end", 1, 2.0, 3, s);
  MARKED("%<unsupported:o>", "%o", 8u);
  MARKED("%<unsupported:a>|%<skipped:d>", "%a|%d", 1.0, 2);
  MARKED("%<unsupported:Lf>", "%Lf", (long double)1.5);
  MARKED("%<unsupported:ls>", "%ls", L"wide");
  MARKED("%<unsupported:jd>", "%jd", (intmax_t)5);
  MARKED("%<unsupported:+s>|5", "%+s|%d", s, 5);   /* a known type: its argument is read and dropped, the rest goes on */
  MARKED("%<unsupported:05c>|6", "%05c|%d", 'q', 6);
  MARKED("%<unsupported:#p>|7", "%#p|%d", (void *)s, 7);
  MARKED("%<unsupported:#d>|8", "%#d|%d", 1, 8);
  MARKED("%<unsupported:.3c>|9", "%.3c|%d", 'q', 9);
  MARKED("%<unsupported:70000d>|10", "%70000d|%d", 1, 10);
  MARKED("%<unsupported:5%>|11", "%5%|%d", 11);
  MARKED("tail %<incomplete:>", "tail %");
  int sentinel = 12345;
  MARKED("before %<unsupported:n> %<skipped:d>", "before %n %d", &sentinel, 1);
  total++;
  if (sentinel != 12345) { failed++; printf("  DIFFERS: %%n wrote through its argument\n"); }

  printf("printf oracle: %d of %d formats byte-identical to libc snprintf (ints at their limits, every length, %%f "
         "precision and rounding, inf/nan, strings, %%c with NUL, %%p, %%%%, width/precision/*); %d of %d unsupported "
         "cases written as their marker, no argument guessed, %%n never written\n",
         supported - supported_failed, supported, (total - supported) - (failed - supported_failed), total - supported);
  bankml_set_log(NULL, NULL);
  return failed == 0 ? 0 : 1;
}
