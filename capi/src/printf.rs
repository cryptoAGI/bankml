// SPDX-License-Identifier: MIT OR Apache-2.0
//! The printf formatter behind `bankml_log`, reading its arguments from a C `va_list`.
//!
//! Supported conversions (`%d %i %u %x %X` with `hh h l ll z`, `%f %F %lf`, `%c %s %p %%`; flags `- + space # 0`;
//! width and precision as a number or `*`) are byte-identical to glibc's `snprintf`. Anything else is written as
//! `%<unsupported:SPEC>`, never guessed: after an unknown conversion or length no further argument is read (later
//! conversions become `%<skipped:SPEC>`); a known conversion with flags it does not vouch for, or a width or
//! precision above 65,536, has its argument read and dropped. `%n` never writes through its argument.
//! Details: docs/modules/capi.md.

use std::ffi::{c_char, c_int, c_long, c_longlong, c_uint, c_ulong, c_ulonglong, c_void, VaList};

/// The typed argument reads the formatter needs; implemented by `VaList` and, in the tests, a typed queue.
pub trait Args {
    /// # Safety
    /// The next argument must have been passed with this C type (after the default promotions).
    unsafe fn int(&mut self) -> c_int;
    /// # Safety
    /// As [`Args::int`].
    unsafe fn uint(&mut self) -> c_uint;
    /// # Safety
    /// As [`Args::int`].
    unsafe fn long(&mut self) -> c_long;
    /// # Safety
    /// As [`Args::int`].
    unsafe fn ulong(&mut self) -> c_ulong;
    /// # Safety
    /// As [`Args::int`].
    unsafe fn llong(&mut self) -> c_longlong;
    /// # Safety
    /// As [`Args::int`].
    unsafe fn ullong(&mut self) -> c_ulonglong;
    /// # Safety
    /// As [`Args::int`].
    unsafe fn isize(&mut self) -> isize;
    /// # Safety
    /// As [`Args::int`].
    unsafe fn usize(&mut self) -> usize;
    /// # Safety
    /// As [`Args::int`].
    unsafe fn double(&mut self) -> f64;
    /// # Safety
    /// As [`Args::int`].
    unsafe fn ptr(&mut self) -> *const c_void;
}

impl Args for VaList<'_> {
    unsafe fn int(&mut self) -> c_int {
        unsafe { self.next_arg::<c_int>() }
    }
    unsafe fn uint(&mut self) -> c_uint {
        unsafe { self.next_arg::<c_uint>() }
    }
    unsafe fn long(&mut self) -> c_long {
        unsafe { self.next_arg::<c_long>() }
    }
    unsafe fn ulong(&mut self) -> c_ulong {
        unsafe { self.next_arg::<c_ulong>() }
    }
    unsafe fn llong(&mut self) -> c_longlong {
        unsafe { self.next_arg::<c_longlong>() }
    }
    unsafe fn ullong(&mut self) -> c_ulonglong {
        unsafe { self.next_arg::<c_ulonglong>() }
    }
    unsafe fn isize(&mut self) -> isize {
        unsafe { self.next_arg::<isize>() }
    }
    unsafe fn usize(&mut self) -> usize {
        unsafe { self.next_arg::<usize>() }
    }
    unsafe fn double(&mut self) -> f64 {
        unsafe { self.next_arg::<f64>() }
    }
    unsafe fn ptr(&mut self) -> *const c_void {
        unsafe { self.next_arg::<*const c_void>() }
    }
}

/// The largest width or precision honoured.
///
/// A larger one is marked unsupported and its argument read and dropped, so a log call cannot make the library
/// allocate gigabytes.
const LIMIT: usize = 1 << 16;

#[derive(Clone, Copy, PartialEq)]
enum Len {
    None,
    Hh,
    H,
    L,
    Ll,
    Z,
    /// A length modifier the formatter does not read (`j t L q`).
    Other,
}

#[derive(Clone, Copy, PartialEq)]
enum Num {
    None,
    Lit(usize),
    Star,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Signed,
    Unsigned,
    Float,
    Char,
    Str,
    Ptr,
    Percent,
}

#[derive(Default, Clone, Copy)]
struct Flags {
    left: bool,
    plus: bool,
    space: bool,
    alt: bool,
    zero: bool,
}

fn marker(out: &mut Vec<u8>, what: &str, spec: &[u8]) {
    out.extend_from_slice(b"%<");
    out.extend_from_slice(what.as_bytes());
    out.push(b':');
    out.extend_from_slice(spec);
    out.push(b'>');
}

/// `prefix` (sign or `0x`) and `body`, padded to `width`: spaces on the right with `-`, zeros after the prefix with
/// `0`, otherwise spaces on the left.
fn put(out: &mut Vec<u8>, prefix: &[u8], body: &[u8], width: usize, left: bool, zero: bool) {
    let fill = width.saturating_sub(prefix.len() + body.len());
    if left {
        out.extend_from_slice(prefix);
        out.extend_from_slice(body);
        out.resize(out.len() + fill, b' ');
    } else if zero {
        out.extend_from_slice(prefix);
        out.resize(out.len() + fill, b'0');
        out.extend_from_slice(body);
    } else {
        out.resize(out.len() + fill, b' ');
        out.extend_from_slice(prefix);
        out.extend_from_slice(body);
    }
}

/// The digits of `v` in `base` with C's integer precision: at least `prec` digits, none for zero at precision 0.
fn digits(v: u128, base: u32, upper: bool, prec: Option<usize>) -> Vec<u8> {
    let set: &[u8; 16] = if upper { b"0123456789ABCDEF" } else { b"0123456789abcdef" };
    let mut d = Vec::new();
    let mut x = v;
    while x > 0 {
        d.push(set[(x % base as u128) as usize]);
        x /= base as u128;
    }
    match prec {
        Some(0) if v == 0 => {}
        Some(p) => d.resize(d.len().max(p), b'0'),
        None if v == 0 => d.push(b'0'),
        None => {}
    }
    d.reverse();
    d
}

/// Format `fmt` with arguments from `a`, byte-identical to glibc's `snprintf` for every supported conversion.
///
/// # Safety
/// Each argument `a` yields must have the C type its conversion names (as for `printf`), and a `%s` argument must be
/// NULL or point to a string readable up to its NUL or its precision.
pub unsafe fn format(fmt: &[u8], a: &mut impl Args) -> Vec<u8> {
    let mut out = Vec::with_capacity(fmt.len() + 16);
    let mut lost = false;
    let mut i = 0;
    while i < fmt.len() {
        if fmt[i] != b'%' {
            out.push(fmt[i]);
            i += 1;
            continue;
        }
        let start = i + 1;
        i += 1;
        let mut fl = Flags::default();
        while let Some(&c) = fmt.get(i) {
            match c {
                b'-' => fl.left = true,
                b'+' => fl.plus = true,
                b' ' => fl.space = true,
                b'#' => fl.alt = true,
                b'0' => fl.zero = true,
                _ => break,
            }
            i += 1;
        }
        let number = |i: &mut usize| -> Num {
            if fmt.get(*i) == Some(&b'*') {
                *i += 1;
                return Num::Star;
            }
            let s = *i;
            let mut n = 0usize;
            while let Some(&c @ b'0'..=b'9') = fmt.get(*i) {
                n = n.saturating_mul(10).saturating_add((c - b'0') as usize);
                *i += 1;
            }
            if *i > s { Num::Lit(n) } else { Num::None }
        };
        let width = number(&mut i);
        let prec = if fmt.get(i) == Some(&b'.') {
            i += 1;
            match number(&mut i) {
                Num::None => Num::Lit(0), // "." alone is precision 0
                n => n,
            }
        } else {
            Num::None
        };
        let len = match (fmt.get(i), fmt.get(i + 1)) {
            (Some(b'h'), Some(b'h')) => {
                i += 2;
                Len::Hh
            }
            (Some(b'l'), Some(b'l')) => {
                i += 2;
                Len::Ll
            }
            (Some(b'h'), _) => {
                i += 1;
                Len::H
            }
            (Some(b'l'), _) => {
                i += 1;
                Len::L
            }
            (Some(b'z'), _) => {
                i += 1;
                Len::Z
            }
            (Some(b'j' | b't' | b'L' | b'q'), _) => {
                i += 1;
                Len::Other
            }
            _ => Len::None,
        };
        let Some(&conv) = fmt.get(i) else {
            marker(&mut out, "incomplete", &fmt[start..]);
            break;
        };
        i += 1;
        let spec = &fmt[start..i];
        let int_len = matches!(len, Len::None | Len::Hh | Len::H | Len::L | Len::Ll | Len::Z);
        let kind = match (conv, len) {
            (b'd' | b'i', _) if int_len => Some(Kind::Signed),
            (b'u' | b'x' | b'X', _) if int_len => Some(Kind::Unsigned),
            (b'f' | b'F', Len::None | Len::L) => Some(Kind::Float),
            (b'c', Len::None) => Some(Kind::Char),
            (b's', Len::None) => Some(Kind::Str),
            (b'p', Len::None) => Some(Kind::Ptr),
            (b'%', Len::None) => Some(Kind::Percent),
            _ => None,
        };
        if kind == Some(Kind::Percent) {
            // `%%` prints `%`; anything between the two (`%5%`) is marked but reads no argument.
            if spec == b"%" { out.push(b'%') } else { marker(&mut out, "unsupported", spec) }
            continue;
        }
        if lost {
            marker(&mut out, "skipped", spec);
            continue;
        }
        let Some(kind) = kind else {
            // The argument's type is unknown: read none, here or after.
            marker(&mut out, "unsupported", spec);
            lost = true;
            continue;
        };
        // `*` arguments come first, width then precision, each an `int`.
        let width = match width {
            Num::Star => {
                let w = unsafe { a.int() };
                if w < 0 {
                    fl.left = true;
                }
                w.unsigned_abs() as usize
            }
            Num::Lit(w) => w,
            Num::None => 0,
        };
        let prec = match prec {
            Num::Star => {
                let p = unsafe { a.int() };
                if p < 0 { None } else { Some(p as usize) }
            }
            Num::Lit(p) => Some(p),
            Num::None => None,
        };
        let vouched = width <= LIMIT
            && prec.is_none_or(|p| p <= LIMIT)
            && match kind {
                Kind::Signed => !fl.alt || conv == b'x',
                Kind::Unsigned => !fl.alt || conv != b'u',
                Kind::Float => true,
                Kind::Char | Kind::Ptr => !(fl.plus || fl.space || fl.alt || fl.zero) && prec.is_none(),
                Kind::Str => !(fl.plus || fl.space || fl.alt || fl.zero),
                Kind::Percent => true,
            };
        let zero = fl.zero && !fl.left;
        match kind {
            Kind::Signed => {
                let v: i128 = unsafe {
                    match len {
                        Len::Hh => a.int() as i8 as i128,
                        Len::H => a.int() as i16 as i128,
                        Len::L => a.long() as i128,
                        Len::Ll => a.llong() as i128,
                        Len::Z => a.isize() as i128,
                        _ => a.int() as i128,
                    }
                };
                if !vouched {
                    marker(&mut out, "unsupported", spec);
                    continue;
                }
                let sign: &[u8] = if v < 0 { b"-" } else if fl.plus { b"+" } else if fl.space { b" " } else { b"" };
                put(&mut out, sign, &digits(v.unsigned_abs(), 10, false, prec), width, fl.left, zero && prec.is_none());
            }
            Kind::Unsigned => {
                let v: u128 = unsafe {
                    match len {
                        Len::Hh => a.uint() as u8 as u128,
                        Len::H => a.uint() as u16 as u128,
                        Len::L => a.ulong() as u128,
                        Len::Ll => a.ullong() as u128,
                        Len::Z => a.usize() as u128,
                        _ => a.uint() as u128,
                    }
                };
                if !vouched {
                    marker(&mut out, "unsupported", spec);
                    continue;
                }
                // C ignores `+` and space for unsigned conversions; `#` prefixes 0x/0X to a non-zero hex value.
                let (base, upper) = match conv {
                    b'x' => (16, false),
                    b'X' => (16, true),
                    _ => (10, false),
                };
                let prefix: &[u8] = match (fl.alt && v != 0, conv) {
                    (true, b'x') => b"0x",
                    (true, b'X') => b"0X",
                    _ => b"",
                };
                put(&mut out, prefix, &digits(v, base, upper, prec), width, fl.left, zero && prec.is_none());
            }
            Kind::Float => {
                let v = unsafe { a.double() };
                if !vouched {
                    marker(&mut out, "unsupported", spec);
                    continue;
                }
                let sign: &[u8] = if v.is_sign_negative() { b"-" } else if fl.plus { b"+" } else if fl.space { b" " } else { b"" };
                if !v.is_finite() {
                    let body: &[u8] = match (v.is_nan(), conv == b'F') {
                        (true, false) => b"nan",
                        (true, true) => b"NAN",
                        (false, false) => b"inf",
                        (false, true) => b"INF",
                    };
                    put(&mut out, sign, body, width, fl.left, false); // C pads inf and nan with spaces, never zeros
                    continue;
                }
                let p = prec.unwrap_or(6);
                // Rust rounds the exact decimal expansion to nearest, ties to even, as glibc does in the default
                // rounding mode.
                let mut body = format!("{:.*}", p, v.abs()).into_bytes();
                if fl.alt && p == 0 {
                    body.push(b'.');
                }
                put(&mut out, sign, &body, width, fl.left, zero);
            }
            Kind::Char => {
                let c = unsafe { a.int() } as u8;
                if !vouched {
                    marker(&mut out, "unsupported", spec);
                    continue;
                }
                put(&mut out, b"", &[c], width, fl.left, false);
            }
            Kind::Str => {
                let p = unsafe { a.ptr() } as *const c_char;
                if !vouched {
                    marker(&mut out, "unsupported", spec);
                    continue;
                }
                let body: Vec<u8> = if p.is_null() {
                    // glibc prints "(null)" when it fits the precision, else nothing.
                    if prec.is_none_or(|p| p >= 6) { b"(null)".to_vec() } else { Vec::new() }
                } else {
                    // At most `prec` bytes and never past the NUL: an unterminated buffer with a precision is legal C.
                    let mut b = Vec::new();
                    let max = prec.unwrap_or(usize::MAX);
                    while b.len() < max {
                        let c = unsafe { *p.add(b.len()) } as u8;
                        if c == 0 {
                            break;
                        }
                        b.push(c);
                    }
                    b
                };
                put(&mut out, b"", &body, width, fl.left, false);
            }
            Kind::Ptr => {
                let p = unsafe { a.ptr() } as usize;
                if !vouched {
                    marker(&mut out, "unsupported", spec);
                    continue;
                }
                let body = if p == 0 { b"(nil)".to_vec() } else { format!("0x{p:x}").into_bytes() };
                put(&mut out, b"", &body, width, fl.left, false);
            }
            Kind::Percent => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    /// A typed argument queue: the test fails if the formatter reads a type other than the one passed.
    #[derive(Debug, Clone, Copy)]
    pub enum A {
        I(c_int),
        U(c_uint),
        L(c_long),
        Ul(c_ulong),
        Ll(c_longlong),
        Ull(c_ulonglong),
        Z(usize),
        Zd(isize),
        D(f64),
        P(*const c_void),
    }

    pub struct Q(pub VecDeque<A>);

    macro_rules! take {
        ($self:ident, $v:ident) => {
            match $self.0.pop_front() {
                Some(A::$v(x)) => x,
                other => panic!("read {} but the next argument is {other:?}", stringify!($v)),
            }
        };
    }

    impl Args for Q {
        unsafe fn int(&mut self) -> c_int { take!(self, I) }
        unsafe fn uint(&mut self) -> c_uint { take!(self, U) }
        unsafe fn long(&mut self) -> c_long { take!(self, L) }
        unsafe fn ulong(&mut self) -> c_ulong { take!(self, Ul) }
        unsafe fn llong(&mut self) -> c_longlong { take!(self, Ll) }
        unsafe fn ullong(&mut self) -> c_ulonglong { take!(self, Ull) }
        unsafe fn isize(&mut self) -> isize { take!(self, Zd) }
        unsafe fn usize(&mut self) -> usize { take!(self, Z) }
        unsafe fn double(&mut self) -> f64 { take!(self, D) }
        unsafe fn ptr(&mut self) -> *const c_void { take!(self, P) }
    }

    fn f(fmt: &str, args: &[A]) -> (String, usize) {
        let mut q = Q(args.iter().copied().collect());
        let out = unsafe { format(fmt.as_bytes(), &mut q) };
        (String::from_utf8(out).unwrap(), q.0.len())
    }

    fn eq(fmt: &str, args: &[A], want: &str) {
        let (got, left) = f(fmt, args);
        assert_eq!(got, want, "{fmt}");
        assert_eq!(left, 0, "{fmt}: every argument read");
    }

    #[test]
    fn integers_as_c_prints_them() {
        use A::*;
        eq("%d|%i|%u", &[I(-42), I(c_int::MIN), U(c_uint::MAX)], "-42|-2147483648|4294967295");
        eq("%5d|%-5d|%05d|%+d|% d|%.3d|%.0d|%8.3d", &[I(42), I(42), I(-42), I(7), I(7), I(5), I(0), I(-5)], "   42|42   |-0042|+7| 7|005||    -005");
        eq("%x|%X|%#x|%#X|%#x|%08x|%#010x", &[U(255), U(255), U(255), U(255), U(0), U(0xbeef), U(0xbeef)], "ff|FF|0xff|0XFF|0|0000beef|0x0000beef");
        eq("%ld|%lu|%lld|%llu|%zu|%zd", &[L(c_long::MIN), Ul(c_ulong::MAX), Ll(c_longlong::MIN), Ull(c_ulonglong::MAX), Z(usize::MAX), Zd(-1)],
           "-9223372036854775808|18446744073709551615|-9223372036854775808|18446744073709551615|18446744073709551615|-1");
        eq("%hhd|%hd|%hhu|%hx", &[I(300), I(70000), U(511), U(0x12345)], "44|4464|255|2345");
        eq("%*d|%-*d|%.*d|%*d", &[I(4), I(1), I(4), I(2), I(3), I(9), I(-4), I(3)], "   1|2   |009|3   ");
        eq("%+u|% u", &[U(3), U(3)], "3|3"); // C ignores + and space for unsigned
    }

    #[test]
    fn floats_strings_chars_pointers() {
        use A::*;
        eq("%f|%.3f|%.0f|%.0f|%.0f|%#.0f|%10.2f|%-10.2f|%010.2f|%+.1f", &[D(3.14258), D(-2.0005), D(0.5), D(1.5), D(2.5), D(3.0), D(-1.005), D(1.0), D(-3.25), D(0.0)],
           "3.142580|-2.001|0|2|2|3.|     -1.00|1.00      |-000003.25|+0.0");
        eq("%f|%F|%f|%5f|%05f|%.2f", &[D(f64::INFINITY), D(f64::NEG_INFINITY), D(f64::NAN), D(f64::INFINITY), D(f64::INFINITY), D(-0.0)], "inf|-INF|nan|  inf|  inf|-0.00");
        let s = c"bankml";
        eq("[%s|%8s|%-8s|%.4s|%.*s]", &[P(s.as_ptr().cast()), P(s.as_ptr().cast()), P(s.as_ptr().cast()), P(s.as_ptr().cast()), I(2), P(s.as_ptr().cast())],
           "[bankml|  bankml|bankml  |bank|ba]");
        eq("%s|%.3s", &[P(std::ptr::null()), P(std::ptr::null())], "(null)|");
        eq("%c%c|%3c|%-3c|", &[I(b'o' as c_int), I(b'k' as c_int), I(b'x' as c_int), I(b'y' as c_int)], "ok|  x|y  |");
        eq("%p|%p|%8p", &[P(0x1234 as *const c_void), P(std::ptr::null()), P(std::ptr::null())], "0x1234|(nil)|   (nil)");
        eq("100%%", &[], "100%");
    }

    #[test]
    fn what_is_not_supported_is_marked_never_guessed() {
        use A::*;
        // An unknown conversion reads nothing, and nothing after it is read.
        let (got, left) = f("a %d %w %d %s %% end", &[I(1), I(2), P(std::ptr::null())]);
        assert_eq!(got, "a 1 %<unsupported:w> %<skipped:d> %<skipped:s> % end");
        assert_eq!(left, 2);
        for spec in ["e", "g", "o", "n", "Lf", "ls", "lc", "jd", "a", "hf", "zc"] {
            let (got, left) = f(&format!("%{spec}"), &[I(1)]);
            assert_eq!(got, format!("%<unsupported:{spec}>"));
            assert_eq!(left, 1, "{spec}: nothing read");
        }
        // A known conversion with unvouched flags is marked; its argument (of known type) is read and dropped.
        eq("%+s|%05c|%#p|%#d|%.3p|%d", &[P(c"x".as_ptr().cast()), I(65), P(std::ptr::null()), I(1), P(std::ptr::null()), I(9)],
           "%<unsupported:+s>|%<unsupported:05c>|%<unsupported:#p>|%<unsupported:#d>|%<unsupported:.3p>|9");
        eq("%70000d|%d", &[I(1), I(2)], "%<unsupported:70000d>|2");
        // floats too: a width or precision past LIMIT is marked, not formatted (and not allocated)
        eq("%.100000000f|%70000f|%.1f", &[D(1.0), D(2.0), D(0.25)], "%<unsupported:.100000000f>|%<unsupported:70000f>|0.2");
        eq("%5%|tail %", &[], "%<unsupported:5%>|tail %<incomplete:>");
    }
}
