// SPDX-License-Identifier: MIT OR Apache-2.0
//! bankml's GPU kernels, generated as SPIR-V by `spirv.rs`. Each reproduces its CPU kernel's float order exactly, so
//! the GPU result is the CPU result, which is ggml's (§III.4): same bits first, then speed.
//!
//! `q1_0_mat_vec`: one invocation per output row, following `q1_0::vec_dot_ref` step by step — per 128-weight block,
//! four 32-element sub-blocks; per sub-block eight lanes of an exact integer sum of ±q over four elements (ggml's
//! i8 negation wraps −128 to −128); per lane `ab = d1·s` on the first sub-block and `fma(d1, s, ab)` after;
//! `acc = fma(d0, ab, acc)`; then the horizontal sum `(a0+a4 + a2+a6) + (a1+a5 + a3+a7)`. Weights are repacked
//! once on upload (scales f16→f32, exact; bits as u32 words) and activations per call (scales f32, quants packed
//! four to a word) — the numbers are unchanged, only aligned.

use super::spirv::{dec, op, sc, Module, BUILTIN_GLOBAL_INVOCATION_ID, BUILTIN_LOCAL_INVOCATION_INDEX, GLSL_FMA};

/// Bindings of `q1_0_mat_vec`: weight scales (f32), weight bits (u32 ×4 per block), activation scales (f32 per 32),
/// activation quants (i32 words of four i8), output (f32). Push constants: rows, blocks per row.
pub const Q1_0_BINDINGS: u32 = 5;
pub const LOCAL_SIZE: u32 = 64;

pub fn q1_0_mat_vec() -> Vec<u32> {
    let mut m = Module::new();
    m.capability(1); // Shader
    let glsl = m.ext_import("GLSL.std.450");
    m.memory_model_glsl450();
    let void = m.ty(op::TYPE_VOID, &[]);
    let fn_ty = m.ty(op::TYPE_FUNCTION, &[void]);
    let tbool = m.ty(op::TYPE_BOOL, &[]);
    let u32t = m.ty(op::TYPE_INT, &[32, 0]);
    let i32t = m.ty(op::TYPE_INT, &[32, 1]);
    let f32t = m.ty(op::TYPE_FLOAT, &[32]);
    let v3u = m.ty(op::TYPE_VECTOR, &[u32t, 3]);
    // storage buffers: struct { T data[]; } per element type
    let buffer_ty = |m: &mut Module, elem: u32| {
        let arr = m.ty(op::TYPE_RUNTIME_ARRAY, &[elem]);
        m.decorate(arr, &[dec::ARRAY_STRIDE, 4]);
        let st = m.ty(op::TYPE_STRUCT, &[arr]);
        m.decorate(st, &[dec::BLOCK]);
        m.member_decorate(st, 0, &[dec::OFFSET, 0]);
        let p = m.ty(op::TYPE_POINTER, &[sc::STORAGE_BUFFER, st]);
        let pe = m.ty(op::TYPE_POINTER, &[sc::STORAGE_BUFFER, elem]);
        (p, pe)
    };
    let (p_sf, pe_f) = buffer_ty(&mut m, f32t);
    let (p_su, pe_u) = buffer_ty(&mut m, u32t);
    let (p_si, pe_i) = buffer_ty(&mut m, i32t);
    let bind = |m: &mut Module, p: u32, b: u32| {
        let v = m.global(p, sc::STORAGE_BUFFER);
        m.decorate(v, &[dec::DESCRIPTOR_SET, 0]);
        m.decorate(v, &[dec::BINDING, b]);
        v
    };
    let (wd, wbits, ad, aq, outb) = (bind(&mut m, p_sf, 0), bind(&mut m, p_su, 1), bind(&mut m, p_sf, 2), bind(&mut m, p_si, 3), bind(&mut m, p_sf, 4));
    // push constants { u32 rows; u32 nb; }
    let pc_st = m.ty(op::TYPE_STRUCT, &[u32t, u32t]);
    m.decorate(pc_st, &[dec::BLOCK]);
    m.member_decorate(pc_st, 0, &[dec::OFFSET, 0]);
    m.member_decorate(pc_st, 1, &[dec::OFFSET, 4]);
    let p_pc = m.ty(op::TYPE_POINTER, &[sc::PUSH_CONSTANT, pc_st]);
    let pe_pcu = m.ty(op::TYPE_POINTER, &[sc::PUSH_CONSTANT, u32t]);
    let pc = m.global(p_pc, sc::PUSH_CONSTANT);
    let p_in = m.ty(op::TYPE_POINTER, &[sc::INPUT, v3u]);
    let gid = m.global(p_in, sc::INPUT);
    m.decorate(gid, &[dec::BUILTIN, BUILTIN_GLOBAL_INVOCATION_ID]);
    let pf_f = m.ty(op::TYPE_POINTER, &[sc::FUNCTION, f32t]);
    let pf_u = m.ty(op::TYPE_POINTER, &[sc::FUNCTION, u32t]);
    let cu: Vec<u32> = (0..40).map(|v| m.const_u32(u32t, v)).collect();
    let ci0 = m.const_u32(i32t, 0);
    let ci_m128 = m.const_u32(i32t, (-128i32) as u32);
    let cf0 = m.const_u32(f32t, 0);
    let c_mask = m.const_u32(u32t, 0xFFFF_F000);

    let func = m.id();
    m.entry_point_compute(func, "main", &[gid], [LOCAL_SIZE, 1, 1]);
    m.stmt(op::FUNCTION, &[void, func, 0, fn_ty]);
    let (l_entry, l_work, l_head, l_check, l_body, l_cont, l_merge, l_end) = (m.id(), m.id(), m.id(), m.id(), m.id(), m.id(), m.id(), m.id());
    m.label(l_entry);
    let acc: Vec<u32> = (0..8).map(|_| m.op(op::VARIABLE, pf_f, &[sc::FUNCTION])).collect();
    let ivar = m.op(op::VARIABLE, pf_u, &[sc::FUNCTION]);
    let g = m.op(op::LOAD, v3u, &[gid]);
    let r = m.op(op::COMPOSITE_EXTRACT, u32t, &[g, 0]);
    let prow = m.op(op::ACCESS_CHAIN, pe_pcu, &[pc, cu[0]]);
    let rows = m.op(op::LOAD, u32t, &[prow]);
    let pnb = m.op(op::ACCESS_CHAIN, pe_pcu, &[pc, cu[1]]);
    let nb = m.op(op::LOAD, u32t, &[pnb]);
    let inrange = m.op(op::U_LESS_THAN, tbool, &[r, rows]);
    m.stmt(op::SELECTION_MERGE, &[l_end, 0]);
    m.stmt(op::BRANCH_CONDITIONAL, &[inrange, l_work, l_end]);

    m.label(l_work);
    for &a in &acc {
        m.stmt(op::STORE, &[a, cf0]);
    }
    m.stmt(op::STORE, &[ivar, cu[0]]);
    m.stmt(op::BRANCH, &[l_head]);

    m.label(l_head);
    m.stmt(op::LOOP_MERGE, &[l_merge, l_cont, 0]);
    m.stmt(op::BRANCH, &[l_check]);

    m.label(l_check);
    let iv = m.op(op::LOAD, u32t, &[ivar]);
    let more = m.op(op::U_LESS_THAN, tbool, &[iv, nb]);
    m.stmt(op::BRANCH_CONDITIONAL, &[more, l_body, l_merge]);

    m.label(l_body);
    let rnb = m.op(op::I_MUL, u32t, &[r, nb]);
    let blk = m.op(op::I_ADD, u32t, &[rnb, iv]);
    let p = m.op(op::ACCESS_CHAIN, pe_f, &[wd, cu[0], blk]);
    let d0 = m.op(op::LOAD, f32t, &[p]);
    let blk4 = m.op(op::I_MUL, u32t, &[blk, cu[4]]);
    let iv4 = m.op(op::I_MUL, u32t, &[iv, cu[4]]);
    let mut ab = [0u32; 8];
    for k in 0..4u32 {
        let wi = m.op(op::I_ADD, u32t, &[blk4, cu[k as usize]]);
        let pw = m.op(op::ACCESS_CHAIN, pe_u, &[wbits, cu[0], wi]);
        let wword = m.op(op::LOAD, u32t, &[pw]);
        let ab_i = m.op(op::I_ADD, u32t, &[iv4, cu[k as usize]]);
        let pd = m.op(op::ACCESS_CHAIN, pe_f, &[ad, cu[0], ab_i]);
        let d1 = m.op(op::LOAD, f32t, &[pd]);
        let ab8 = m.op(op::I_MUL, u32t, &[ab_i, cu[8]]);
        for l in 0..8u32 {
            let qi = m.op(op::I_ADD, u32t, &[ab8, cu[l as usize]]);
            let pq = m.op(op::ACCESS_CHAIN, pe_i, &[aq, cu[0], qi]);
            let qword = m.op(op::LOAD, i32t, &[pq]);
            let mut s = ci0;
            for e in 0..4u32 {
                // sign-extend byte e: shift it to the top, then arithmetic-shift it back down
                let up = m.op(op::SHIFT_LEFT_LOGICAL, i32t, &[qword, cu[(24 - 8 * e) as usize]]);
                let q = m.op(op::SHIFT_RIGHT_ARITHMETIC, i32t, &[up, cu[24]]);
                let sh = m.op(op::SHIFT_RIGHT_LOGICAL, u32t, &[wword, cu[(4 * l + e) as usize]]);
                let bit = m.op(op::BITWISE_AND, u32t, &[sh, cu[1]]);
                let set = m.op(op::I_EQUAL, tbool, &[bit, cu[1]]);
                let neg = m.op(op::S_NEGATE, i32t, &[q]);
                let is_min = m.op(op::I_EQUAL, tbool, &[q, ci_m128]);
                let negw = m.op(op::SELECT, i32t, &[is_min, q, neg]);
                let term = m.op(op::SELECT, i32t, &[set, q, negw]);
                s = m.op(op::I_ADD, i32t, &[s, term]);
            }
            let sf = m.op(op::CONVERT_S_TO_F, f32t, &[s]);
            ab[l as usize] = if k == 0 {
                m.op(op::F_MUL, f32t, &[d1, sf])
            } else {
                m.op(op::EXT_INST, f32t, &[glsl, GLSL_FMA, d1, sf, ab[l as usize]])
            };
        }
    }
    for l in 0..8 {
        let a = m.op(op::LOAD, f32t, &[acc[l]]);
        // the outer fma's product is not exact in f32, and a driver may not fuse it: computed exactly (spirv.rs)
        let n = m.fma_exact((f32t, u32t, tbool), d0, ab[l], a, c_mask, cu[1], cu[0], cf0);
        m.stmt(op::STORE, &[acc[l], n]);
    }
    m.stmt(op::BRANCH, &[l_cont]);

    m.label(l_cont);
    let iv2 = m.op(op::LOAD, u32t, &[ivar]);
    let inc = m.op(op::I_ADD, u32t, &[iv2, cu[1]]);
    m.stmt(op::STORE, &[ivar, inc]);
    m.stmt(op::BRANCH, &[l_head]);

    m.label(l_merge);
    let a: Vec<u32> = acc.iter().map(|&v| m.op(op::LOAD, f32t, &[v])).collect();
    let r4: Vec<u32> = (0..4).map(|i| m.op(op::F_ADD, f32t, &[a[i], a[i + 4]])).collect();
    let lo = m.op(op::F_ADD, f32t, &[r4[0], r4[2]]);
    let hi = m.op(op::F_ADD, f32t, &[r4[1], r4[3]]);
    let sum = m.op(op::F_ADD, f32t, &[lo, hi]);
    let po = m.op(op::ACCESS_CHAIN, pe_f, &[outb, cu[0], r]);
    m.stmt(op::STORE, &[po, sum]);
    m.stmt(op::BRANCH, &[l_end]);

    m.label(l_end);
    m.stmt(op::RETURN, &[]);
    m.stmt(op::FUNCTION_END, &[]);
    m.words()
}

/// `q1_0_mat_vec8`: the same arithmetic with eight invocations per row, one per accumulation lane of the CPU kernel
/// (each lane's chain over the blocks is independent there too), so a workgroup of 64 covers 8 rows. The eight lane
/// sums meet in workgroup memory and lane 0 adds them in the CPU's order, `(a0+a4 + a2+a6) + (a1+a5 + a3+a7)`: the
/// same bits as `q1_0_mat_vec`, with eight times the parallelism. Dispatch `ceil(rows · 8 / 64)` workgroups.
pub fn q1_0_mat_vec8() -> Vec<u32> {
    let mut m = Module::new();
    m.capability(1);
    let glsl = m.ext_import("GLSL.std.450");
    m.memory_model_glsl450();
    let void = m.ty(op::TYPE_VOID, &[]);
    let fn_ty = m.ty(op::TYPE_FUNCTION, &[void]);
    let tbool = m.ty(op::TYPE_BOOL, &[]);
    let u32t = m.ty(op::TYPE_INT, &[32, 0]);
    let i32t = m.ty(op::TYPE_INT, &[32, 1]);
    let f32t = m.ty(op::TYPE_FLOAT, &[32]);
    let v3u = m.ty(op::TYPE_VECTOR, &[u32t, 3]);
    let buffer_ty = |m: &mut Module, elem: u32| {
        let arr = m.ty(op::TYPE_RUNTIME_ARRAY, &[elem]);
        m.decorate(arr, &[dec::ARRAY_STRIDE, 4]);
        let st = m.ty(op::TYPE_STRUCT, &[arr]);
        m.decorate(st, &[dec::BLOCK]);
        m.member_decorate(st, 0, &[dec::OFFSET, 0]);
        (m.ty(op::TYPE_POINTER, &[sc::STORAGE_BUFFER, st]), m.ty(op::TYPE_POINTER, &[sc::STORAGE_BUFFER, elem]))
    };
    let (p_sf, pe_f) = buffer_ty(&mut m, f32t);
    let (p_su, pe_u) = buffer_ty(&mut m, u32t);
    let (p_si, pe_i) = buffer_ty(&mut m, i32t);
    let bind = |m: &mut Module, p: u32, b: u32| {
        let v = m.global(p, sc::STORAGE_BUFFER);
        m.decorate(v, &[dec::DESCRIPTOR_SET, 0]);
        m.decorate(v, &[dec::BINDING, b]);
        v
    };
    let (wd, wbits, ad, aq, outb) = (bind(&mut m, p_sf, 0), bind(&mut m, p_su, 1), bind(&mut m, p_sf, 2), bind(&mut m, p_si, 3), bind(&mut m, p_sf, 4));
    let pc_st = m.ty(op::TYPE_STRUCT, &[u32t, u32t]);
    m.decorate(pc_st, &[dec::BLOCK]);
    m.member_decorate(pc_st, 0, &[dec::OFFSET, 0]);
    m.member_decorate(pc_st, 1, &[dec::OFFSET, 4]);
    let p_pc = m.ty(op::TYPE_POINTER, &[sc::PUSH_CONSTANT, pc_st]);
    let pe_pcu = m.ty(op::TYPE_POINTER, &[sc::PUSH_CONSTANT, u32t]);
    let pc = m.global(p_pc, sc::PUSH_CONSTANT);
    let p_in3 = m.ty(op::TYPE_POINTER, &[sc::INPUT, v3u]);
    let p_in1 = m.ty(op::TYPE_POINTER, &[sc::INPUT, u32t]);
    let gid = m.global(p_in3, sc::INPUT);
    m.decorate(gid, &[dec::BUILTIN, BUILTIN_GLOBAL_INVOCATION_ID]);
    let lid = m.global(p_in1, sc::INPUT);
    m.decorate(lid, &[dec::BUILTIN, BUILTIN_LOCAL_INVOCATION_INDEX]);
    let cu: Vec<u32> = (0..40).map(|v| m.const_u32(u32t, v)).collect();
    let c64 = m.const_u32(u32t, LOCAL_SIZE);
    let c_sem = m.const_u32(u32t, 0x108); // AcquireRelease | WorkgroupMemory
    let shared_ty = m.ty(op::TYPE_ARRAY, &[f32t, c64]);
    let p_sh = m.ty(op::TYPE_POINTER, &[sc::WORKGROUP, shared_ty]);
    let pe_sh = m.ty(op::TYPE_POINTER, &[sc::WORKGROUP, f32t]);
    let shared = m.global(p_sh, sc::WORKGROUP);
    let pf_f = m.ty(op::TYPE_POINTER, &[sc::FUNCTION, f32t]);
    let pf_u = m.ty(op::TYPE_POINTER, &[sc::FUNCTION, u32t]);
    let ci0 = m.const_u32(i32t, 0);
    let ci_m128 = m.const_u32(i32t, (-128i32) as u32);
    let cf0 = m.const_u32(f32t, 0);
    let c_mask = m.const_u32(u32t, 0xFFFF_F000);

    let func = m.id();
    m.entry_point_compute(func, "main", &[gid, lid], [LOCAL_SIZE, 1, 1]);
    m.stmt(op::FUNCTION, &[void, func, 0, fn_ty]);
    let (l_entry, l_work, l_head, l_check, l_body, l_cont, l_merge, l_after, l_sum, l_end) =
        (m.id(), m.id(), m.id(), m.id(), m.id(), m.id(), m.id(), m.id(), m.id(), m.id());
    m.label(l_entry);
    let acc = m.op(op::VARIABLE, pf_f, &[sc::FUNCTION]);
    let ivar = m.op(op::VARIABLE, pf_u, &[sc::FUNCTION]);
    let g = m.op(op::LOAD, v3u, &[gid]);
    let gx = m.op(op::COMPOSITE_EXTRACT, u32t, &[g, 0]);
    let r = m.op(op::SHIFT_RIGHT_LOGICAL, u32t, &[gx, cu[3]]);
    let lane = m.op(op::BITWISE_AND, u32t, &[gx, cu[7]]);
    let lane4 = m.op(op::I_MUL, u32t, &[lane, cu[4]]);
    let li = m.op(op::LOAD, u32t, &[lid]);
    let prow = m.op(op::ACCESS_CHAIN, pe_pcu, &[pc, cu[0]]);
    let rows = m.op(op::LOAD, u32t, &[prow]);
    let pnb = m.op(op::ACCESS_CHAIN, pe_pcu, &[pc, cu[1]]);
    let nb = m.op(op::LOAD, u32t, &[pnb]);
    let inrange = m.op(op::U_LESS_THAN, tbool, &[r, rows]);
    m.stmt(op::STORE, &[acc, cf0]);
    m.stmt(op::SELECTION_MERGE, &[l_after, 0]);
    m.stmt(op::BRANCH_CONDITIONAL, &[inrange, l_work, l_after]);

    m.label(l_work);
    m.stmt(op::STORE, &[ivar, cu[0]]);
    m.stmt(op::BRANCH, &[l_head]);
    m.label(l_head);
    m.stmt(op::LOOP_MERGE, &[l_merge, l_cont, 0]);
    m.stmt(op::BRANCH, &[l_check]);
    m.label(l_check);
    let iv = m.op(op::LOAD, u32t, &[ivar]);
    let more = m.op(op::U_LESS_THAN, tbool, &[iv, nb]);
    m.stmt(op::BRANCH_CONDITIONAL, &[more, l_body, l_merge]);

    m.label(l_body);
    let rnb = m.op(op::I_MUL, u32t, &[r, nb]);
    let blk = m.op(op::I_ADD, u32t, &[rnb, iv]);
    let p = m.op(op::ACCESS_CHAIN, pe_f, &[wd, cu[0], blk]);
    let d0 = m.op(op::LOAD, f32t, &[p]);
    let blk4 = m.op(op::I_MUL, u32t, &[blk, cu[4]]);
    let iv4 = m.op(op::I_MUL, u32t, &[iv, cu[4]]);
    let mut ab = 0u32;
    for k in 0..4u32 {
        let wi = m.op(op::I_ADD, u32t, &[blk4, cu[k as usize]]);
        let pw = m.op(op::ACCESS_CHAIN, pe_u, &[wbits, cu[0], wi]);
        let wword = m.op(op::LOAD, u32t, &[pw]);
        let ab_i = m.op(op::I_ADD, u32t, &[iv4, cu[k as usize]]);
        let pd = m.op(op::ACCESS_CHAIN, pe_f, &[ad, cu[0], ab_i]);
        let d1 = m.op(op::LOAD, f32t, &[pd]);
        let ab8 = m.op(op::I_MUL, u32t, &[ab_i, cu[8]]);
        let qi = m.op(op::I_ADD, u32t, &[ab8, lane]);
        let pq = m.op(op::ACCESS_CHAIN, pe_i, &[aq, cu[0], qi]);
        let qword = m.op(op::LOAD, i32t, &[pq]);
        let mut s = ci0;
        for e in 0..4u32 {
            let up = m.op(op::SHIFT_LEFT_LOGICAL, i32t, &[qword, cu[(24 - 8 * e) as usize]]);
            let q = m.op(op::SHIFT_RIGHT_ARITHMETIC, i32t, &[up, cu[24]]);
            let amt = m.op(op::I_ADD, u32t, &[lane4, cu[e as usize]]);
            let sh = m.op(op::SHIFT_RIGHT_LOGICAL, u32t, &[wword, amt]);
            let bit = m.op(op::BITWISE_AND, u32t, &[sh, cu[1]]);
            let set = m.op(op::I_EQUAL, tbool, &[bit, cu[1]]);
            let neg = m.op(op::S_NEGATE, i32t, &[q]);
            let is_min = m.op(op::I_EQUAL, tbool, &[q, ci_m128]);
            let negw = m.op(op::SELECT, i32t, &[is_min, q, neg]);
            let term = m.op(op::SELECT, i32t, &[set, q, negw]);
            s = m.op(op::I_ADD, i32t, &[s, term]);
        }
        let sf = m.op(op::CONVERT_S_TO_F, f32t, &[s]);
        ab = if k == 0 { m.op(op::F_MUL, f32t, &[d1, sf]) } else { m.op(op::EXT_INST, f32t, &[glsl, GLSL_FMA, d1, sf, ab]) };
    }
    let a = m.op(op::LOAD, f32t, &[acc]);
    let n = m.fma_exact((f32t, u32t, tbool), d0, ab, a, c_mask, cu[1], cu[0], cf0);
    m.stmt(op::STORE, &[acc, n]);
    m.stmt(op::BRANCH, &[l_cont]);
    m.label(l_cont);
    let iv2 = m.op(op::LOAD, u32t, &[ivar]);
    let inc = m.op(op::I_ADD, u32t, &[iv2, cu[1]]);
    m.stmt(op::STORE, &[ivar, inc]);
    m.stmt(op::BRANCH, &[l_head]);
    m.label(l_merge);
    m.stmt(op::BRANCH, &[l_after]);

    // every invocation reaches the barrier (uniform control flow); lane 0 of each row sums in the CPU's order
    m.label(l_after);
    let mine = m.op(op::LOAD, f32t, &[acc]);
    let ps = m.op(op::ACCESS_CHAIN, pe_sh, &[shared, li]);
    m.stmt(op::STORE, &[ps, mine]);
    m.stmt(op::CONTROL_BARRIER, &[cu[2], cu[2], c_sem]);
    let lane0 = m.op(op::I_EQUAL, tbool, &[lane, cu[0]]);
    let do_sum = m.op(op::LOGICAL_AND, tbool, &[lane0, inrange]);
    m.stmt(op::SELECTION_MERGE, &[l_end, 0]);
    m.stmt(op::BRANCH_CONDITIONAL, &[do_sum, l_sum, l_end]);
    m.label(l_sum);
    let v: Vec<u32> = (0..8usize).map(|j| {
        let ij = m.op(op::I_ADD, u32t, &[li, cu[j]]);
        let pj = m.op(op::ACCESS_CHAIN, pe_sh, &[shared, ij]);
        m.op(op::LOAD, f32t, &[pj])
    }).collect();
    let r4: Vec<u32> = (0..4).map(|i| m.op(op::F_ADD, f32t, &[v[i], v[i + 4]])).collect();
    let lo = m.op(op::F_ADD, f32t, &[r4[0], r4[2]]);
    let hi = m.op(op::F_ADD, f32t, &[r4[1], r4[3]]);
    let sum = m.op(op::F_ADD, f32t, &[lo, hi]);
    let po = m.op(op::ACCESS_CHAIN, pe_f, &[outb, cu[0], r]);
    m.stmt(op::STORE, &[po, sum]);
    m.stmt(op::BRANCH, &[l_end]);
    m.label(l_end);
    m.stmt(op::RETURN, &[]);
    m.stmt(op::FUNCTION_END, &[]);
    m.words()
}

/// The on-card oracle `bankml gpu --verify` runs before a card is trusted: both Q1_0 kernels against the CPU kernel
/// (bit-exact against ggml) on seeded random weights and activations, −128 quants included. Returns one line per
/// kernel and shape, or the first difference.
pub fn verify_q1_0(gpu: &super::compute::Gpu) -> Result<Vec<String>, String> {
    use crate::q1_0::{mat_vec, Q8Act};
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut lines = Vec::new();
    for (kname, spv, per_row) in [("q1_0_mat_vec", q1_0_mat_vec(), 1u32), ("q1_0_mat_vec8", q1_0_mat_vec8(), 8)] {
        let pipe = gpu.pipeline(&spv, Q1_0_BINDINGS, 8)?;
        // two regimes: uniform data with a −128 quant (the wrapping negation), and data shaped like a real layer —
        // weight scales and activation magnitudes that vary per block, so the products are inexact in f32 and a
        // driver that does not fuse an FMA shows (the Vega 3's does not: 0.2.14 computes the FMA exactly instead)
        for &(rows, n, real) in &[(1000usize, 512usize, false), (4096, 4096, false), (1024, 12288, false), (4096, 4096, true), (2048, 12288, true)] {
            let nb = n / 128;
            let w: Vec<u8> = (0..rows * nb).flat_map(|_| {
                let mut b = [0u8; 18];
                let d = if real { (1.0 + (next() % 1900) as f32) * 1e-5 * if next() & 1 == 1 { -1.0 } else { 1.0 } } else { ((next() % 2000) as f32 - 1000.0) * 1e-5 };
                b[..2].copy_from_slice(&crate::q1_0::f32_to_f16(d).to_le_bytes());
                b[2..].iter_mut().for_each(|x| *x = next() as u8);
                b
            }).collect();
            let mut x: Vec<f32> = (0..n).map(|i| {
                let v = ((next() % 20001) as f32 - 10000.0) * 1e-4;
                if real { v * (1.0 + (i / 32 % 7) as f32 * 0.37) * (0.05 + ((i / 32) * 2654435761 % 97) as f32 * 0.01) } else { v * 10.0 }
            }).collect();
            if !real {
                x[3] = -1e9;
            }
            let a = Q8Act::quantize(&x);
            let mut cpu = vec![0.0f32; rows];
            mat_vec(&w, rows, &a, &mut cpu);
            let (wd, wb) = pack_q1_0(&w, rows, n);
            let (ad, aq) = pack_act(&a);
            let bufs = [gpu.upload(&wd)?, gpu.upload(&wb)?, gpu.upload(&ad)?, gpu.upload(&aq)?, gpu.buffer(rows * 4)?];
            let t = std::time::Instant::now();
            gpu.run(&pipe, &bufs.iter().collect::<Vec<_>>(), &[rows as u32, nb as u32], (rows as u32 * per_row).div_ceil(LOCAL_SIZE))?;
            let dt = t.elapsed().as_secs_f64() * 1e3;
            let got = gpu.read_f32(&bufs[4], rows);
            for b in bufs {
                gpu.free(b);
            }
            if let Some(i) = got.iter().zip(&cpu).position(|(g, c)| g.to_bits() != c.to_bits()) {
                return Err(format!("{kname} {rows}×{n}: row {i} differs from the CPU kernel (gpu {:e}, cpu {:e})", got[i], cpu[i]));
            }
            lines.push(format!("{kname} {rows}×{n}{}: {rows} of {rows} rows bit-exact ({dt:.2} ms)", if real { " (layer-shaped)" } else { "" }));
        }
    }
    Ok(lines)
}

/// A Q1_0 matrix repacked for the kernel: per block its scale as f32 (exact) and its 128 bits as four u32 words.
pub fn pack_q1_0(w: &[u8], rows: usize, n: usize) -> (Vec<f32>, Vec<u32>) {
    let nb = n / 128;
    let (mut d, mut bits) = (Vec::with_capacity(rows * nb), Vec::with_capacity(rows * nb * 4));
    for b in w[..rows * nb * 18].chunks_exact(18) {
        d.push(crate::q1_0::f16_to_f32(u16::from_le_bytes([b[0], b[1]])));
        bits.extend(b[2..18].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])));
    }
    (d, bits)
}

/// A q8_0 activation repacked for the kernel: its f32 scales and its quants four to a word.
pub fn pack_act(a: &crate::q1_0::Q8Act) -> (Vec<f32>, Vec<i32>) {
    let q: Vec<i32> = a.qs().chunks_exact(4).map(|c| i32::from_le_bytes([c[0] as u8, c[1] as u8, c[2] as u8, c[3] as u8])).collect();
    (a.d().to_vec(), q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q1_0::{mat_vec, Q8Act};

    /// The Q1_0 kernel on every usable local GPU against bankml's CPU kernel (itself bit-exact against ggml), on
    /// random weights and activations, including −128 quants (ggml's wrapping negation).
    #[test]
    #[ignore = "needs a Vulkan GPU"]
    fn gpu_q1_0_mat_vec_bit_exact() {
        let devs = crate::gpu::selected(&crate::gpu::discover().0);
        if devs.is_empty() {
            eprintln!("gpu oracle: no usable GPU on this machine; skipped (bankml runs on the CPU)");
            return;
        }
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for d in &devs {
            let gpu = crate::gpu::compute::Gpu::open(d.index).unwrap();
            for (kname, spv, per_row) in [("q1_0_mat_vec", q1_0_mat_vec(), 1u32), ("q1_0_mat_vec8", q1_0_mat_vec8(), 8)] {
            let pipe = gpu.pipeline(&spv, Q1_0_BINDINGS, 8).unwrap();
            for &(rows, n) in &[(64usize, 128usize), (1000, 512), (4096, 4096), (1024, 12288), (12288, 4096)] {
                let nb = n / 128;
                let w: Vec<u8> = (0..rows * nb).flat_map(|_| {
                    let mut b = vec![0u8; 18];
                    let d = crate::q1_0::f32_to_f16(((next() % 2000) as f32 - 1000.0) * 1e-5);
                    b[..2].copy_from_slice(&d.to_le_bytes());
                    for x in b[2..].iter_mut() {
                        *x = next() as u8;
                    }
                    b
                }).collect();
                let mut x: Vec<f32> = (0..n).map(|_| ((next() % 20001) as f32 - 10000.0) * 1e-3).collect();
                x[3] = -1e9; // a block whose quants saturate to −128
                let a = Q8Act::quantize(&x);
                let mut cpu = vec![0.0f32; rows];
                mat_vec(&w, rows, &a, &mut cpu);
                let (wd, wb) = pack_q1_0(&w, rows, n);
                let (ad, aq) = pack_act(&a);
                let bufs = [gpu.upload(&wd).unwrap(), gpu.upload(&wb).unwrap(), gpu.upload(&ad).unwrap(), gpu.upload(&aq).unwrap(), gpu.buffer(rows * 4).unwrap()];
                let t = std::time::Instant::now();
                gpu.run(&pipe, &bufs.iter().collect::<Vec<_>>(), &[rows as u32, nb as u32], (rows as u32 * per_row).div_ceil(LOCAL_SIZE)).unwrap();
                let dt = t.elapsed();
                let tc = std::time::Instant::now();
                mat_vec(&w, rows, &a, &mut cpu);
                let dc = tc.elapsed();
                let got = gpu.read_f32(&bufs[4], rows);
                let same = got.iter().zip(&cpu).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
                eprintln!("gpu {}: {kname} {rows}×{n}: {same} of {rows} rows bit-exact against the CPU kernel ({:.2} ms; CPU 1 thread {:.2} ms)",
                          d.name, dt.as_secs_f64() * 1e3, dc.as_secs_f64() * 1e3);
                if same != rows {
                    let i = got.iter().zip(&cpu).position(|(a, b)| a.to_bits() != b.to_bits()).unwrap();
                    eprintln!("  first difference at row {i}: gpu {:e} cpu {:e}", got[i], cpu[i]);
                }
                assert_eq!(same, rows);
                for b in bufs {
                    gpu.free(b);
                }
            }
            }
        }
    }
}

