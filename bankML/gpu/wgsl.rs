// SPDX-License-Identifier: MIT OR Apache-2.0
//! The GPU kernels in WGSL (WebGPU), learned in browseML and kept here as bankML's own: the twins of the SPIR-V
//! Q1_0 kernels ([`super::kernels`]), each performing the CPU reference's float operations in its order, so a verified
//! card returns the CPU's bits.
//!
//! What browseML taught (docs/modules/gpu.md, "WebGPU"):
//! - **WGSL has no exact fused multiply-add and no NoContraction.** `fma_exact` builds the fused result from an
//!   exact product split (Veltkamp on 12 bits) and TwoSum, then rounds once to odd and lets the last add round.
//! - **Drivers simplify TwoSum away.** Mesa's NIR on AMD rewrote `(a + b) − a` to `b`, so on a real Radeon the
//!   kernel was wrong until every TwoSum step passed through `hide`: an OR with a uniform the host sets to zero,
//!   which no compiler can see through. Vulkan says the same with NoContraction; WGSL cannot.
//! - **Eight invocations per row**, one per accumulation lane of the CPU kernel, then lane 0 adds the eight in the
//!   CPU's order (6.5× the one-per-row kernel on a Vega 3), and **a token dimension** (`gid.y`): a micro-batch of a
//!   prompt in one dispatch.
//! - **The workgroup follows the vendor**: 64 for AMD (wavefronts of 64), 32 for NVIDIA (warps of 32); the size never
//!   changes a bit (checked at 32, 64 and 128).
//!
//! browseML's page and tests (`web/browseml-gpu.js`) carry the same text; its gate checks that the two are identical.

/// The Q1_0 × q8_0 kernel for one activation or many: bindings weight scales (f32 per block), weight bits (four u32
/// per block), activation scales (f32 per 32), activation quants (u32, four i8), output `O[col · rows + row]`, and a
/// uniform (rows, blocks per row, columns, zero). Dispatch `(ceil(rows · 8 / WG), columns)`; `WG` an override.
pub const Q1_0_MM: &str = r#"
override WG: u32 = 64u;
struct P { rows: u32, nb: u32, cols: u32, zero: u32 }
@group(0) @binding(0) var<storage, read> Wd: array<f32>;
@group(0) @binding(1) var<storage, read> Wb: array<u32>;
@group(0) @binding(2) var<storage, read> Ad: array<f32>;
@group(0) @binding(3) var<storage, read> Aq: array<u32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> part: array<f32, WG>;

fn hide(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) | p.zero); }  // see Q1_DOT_WGSL's hide
fn two_sum(a: f32, b: f32) -> vec2<f32> {
  let s = hide(a + b);
  let bb = hide(s - a);
  let sb = hide(s - bb);
  return vec2<f32>(s, hide(a - sb) + hide(b - bb));
}
fn fma_exact(a: f32, b: f32, c: f32) -> f32 {
  let bhi = bitcast<f32>(bitcast<u32>(b) & 0xfffff000u);
  let blo = b - bhi;
  let u = two_sum(a * bhi, a * blo);
  let t = two_sum(c, u.x);
  let v = two_sum(t.y, u.y);
  var vb = bitcast<u32>(v.x);
  if ((vb & 1u) == 0u && v.y != 0.0) { vb = select(vb - 1u, vb + 1u, (v.x > 0.0) == (v.y > 0.0)); }
  return t.x + bitcast<f32>(vb);
}

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = gid.x / 8u;
  let l = gid.x % 8u;
  let c = gid.y;
  let live = r < p.rows && c < p.cols;
  var acc = 0.0;
  if (live) {
    let nb = p.nb;
    for (var i = 0u; i < nb; i++) {
      let d0 = Wd[r * nb + i];
      var ab = 0.0;
      for (var k = 0u; k < 4u; k++) {
        let b = i * 4u + k;
        let d1 = Ad[c * nb * 4u + b];
        let bits = (Wb[(r * nb + i) * 4u + k] >> (4u * l)) & 0xfu;
        let qw = Aq[(c * nb * 4u + b) * 8u + l];
        var s: i32 = 0;
        for (var j = 0u; j < 4u; j++) {
          let q = bitcast<i32>(qw << (24u - 8u * j)) >> 24u;
          // the reference negates as an i8 (wrapping): -(-128) stays -128
          let nq = select(-q, -128, q == -128);
          s += select(nq, q, ((bits >> j) & 1u) == 1u);
        }
        let prod = d1 * f32(s);
        ab = select(prod + ab, prod, k == 0u);
      }
      acc = fma_exact(d0, ab, acc);
    }
  }
  part[lid.x] = acc;
  workgroupBarrier();
  if (live && l == 0u) {
    let o = lid.x;
    let h = ((part[o] + part[o + 4u]) + (part[o + 2u] + part[o + 6u])) + ((part[o + 1u] + part[o + 5u]) + (part[o + 3u] + part[o + 7u]));
    O[c * p.rows + r] = h;
  }
}"#;

/// The workgroup size for a vendor: 64 for AMD, 32 for NVIDIA, else the largest subgroup rounded up to a multiple of
/// 8 (64 when unknown). Any multiple of 8 gives the same bits; this one is the fastest measured or expected.
pub fn workgroup_for(vendor: &str, subgroup_max: u32) -> u32 {
    let v = vendor.to_lowercase();
    if v.contains("nvidia") {
        32
    } else if v.contains("amd") || v.contains("ati") {
        64
    } else if (8..=128).contains(&subgroup_max) {
        subgroup_max.div_ceil(8) * 8
    } else {
        64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workgroups_follow_the_vendor() {
        assert_eq!(workgroup_for("amd", 64), 64);
        assert_eq!(workgroup_for("NVIDIA", 32), 32);
        assert_eq!(workgroup_for("intel", 16), 16);
        assert_eq!(workgroup_for("apple", 0), 64);
        assert_eq!(workgroup_for("qualcomm", 12), 16);
    }

    #[test]
    fn the_kernel_hides_two_sum_and_has_a_token_dimension() {
        // the AMD lesson: every TwoSum step through hide (an OR with the zero uniform)
        assert!(Q1_0_MM.contains("let s = hide(a + b);") && Q1_0_MM.contains("let bb = hide(s - a);") && Q1_0_MM.contains("hide(s - bb)"));
        assert!(Q1_0_MM.contains("let c = gid.y;") && Q1_0_MM.contains("O[c * p.rows + r] = h;"));
        assert!(Q1_0_MM.starts_with("\noverride WG: u32 = 64u;"));
    }
}
