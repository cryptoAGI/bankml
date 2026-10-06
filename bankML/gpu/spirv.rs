// SPDX-License-Identifier: MIT OR Apache-2.0
//! A minimal SPIR-V assembler for bankML's compute kernels, so no shader compiler is needed at build or run time.
//!
//! It covers what the kernels use. Every `OpFMul`, `OpFAdd`, `OpFSub` and extended instruction it emits is
//! decorated `NoContraction`, so a driver cannot fuse operations the CPU kernels keep separate.
//! Details: docs/modules/gpu.md.

const MAGIC: u32 = 0x0723_0203;
const VERSION_1_3: u32 = 0x0001_0300;

/// Opcodes (SPIR-V 1.3 unified specification).
pub mod op {
    pub const EXT_INST_IMPORT: u16 = 11;
    pub const EXT_INST: u16 = 12;
    pub const MEMORY_MODEL: u16 = 14;
    pub const ENTRY_POINT: u16 = 15;
    pub const EXECUTION_MODE: u16 = 16;
    pub const CAPABILITY: u16 = 17;
    pub const TYPE_VOID: u16 = 19;
    pub const TYPE_BOOL: u16 = 20;
    pub const TYPE_INT: u16 = 21;
    pub const TYPE_FLOAT: u16 = 22;
    pub const TYPE_VECTOR: u16 = 23;
    pub const TYPE_ARRAY: u16 = 28;
    pub const TYPE_RUNTIME_ARRAY: u16 = 29;
    pub const TYPE_STRUCT: u16 = 30;
    pub const TYPE_POINTER: u16 = 32;
    pub const TYPE_FUNCTION: u16 = 33;
    pub const CONSTANT: u16 = 43;
    pub const FUNCTION: u16 = 54;
    pub const FUNCTION_END: u16 = 56;
    pub const VARIABLE: u16 = 59;
    pub const LOAD: u16 = 61;
    pub const STORE: u16 = 62;
    pub const ACCESS_CHAIN: u16 = 65;
    pub const DECORATE: u16 = 71;
    pub const MEMBER_DECORATE: u16 = 72;
    pub const COMPOSITE_EXTRACT: u16 = 81;
    pub const CONVERT_S_TO_F: u16 = 111;
    pub const S_NEGATE: u16 = 126;
    pub const BITCAST: u16 = 124;
    pub const I_ADD: u16 = 128;
    pub const I_SUB: u16 = 130;
    pub const F_SUB: u16 = 131;
    pub const F_ADD: u16 = 129;
    pub const I_MUL: u16 = 132;
    pub const F_MUL: u16 = 133;
    pub const LOGICAL_EQUAL: u16 = 164;
    pub const LOGICAL_AND: u16 = 167;
    pub const I_EQUAL: u16 = 170;
    pub const U_LESS_THAN: u16 = 176;
    pub const F_ORD_NOT_EQUAL: u16 = 182;
    pub const F_ORD_GREATER_THAN: u16 = 186;
    pub const SELECT: u16 = 169;
    pub const SHIFT_RIGHT_LOGICAL: u16 = 194;
    pub const SHIFT_RIGHT_ARITHMETIC: u16 = 195;
    pub const SHIFT_LEFT_LOGICAL: u16 = 196;
    pub const BITWISE_AND: u16 = 199;
    pub const BIT_FIELD_S_EXTRACT: u16 = 202;
    pub const CONTROL_BARRIER: u16 = 224;
    pub const LOOP_MERGE: u16 = 246;
    pub const SELECTION_MERGE: u16 = 247;
    pub const LABEL: u16 = 248;
    pub const BRANCH: u16 = 249;
    pub const BRANCH_CONDITIONAL: u16 = 250;
    pub const RETURN: u16 = 253;
}

/// Decorations.
pub mod dec {
    pub const BLOCK: u32 = 2;
    pub const ARRAY_STRIDE: u32 = 6;
    pub const BUILTIN: u32 = 11;
    pub const NO_CONTRACTION: u32 = 42;
    pub const BINDING: u32 = 33;
    pub const DESCRIPTOR_SET: u32 = 34;
    pub const OFFSET: u32 = 35;
    pub const NON_WRITABLE: u32 = 24;
}

/// Storage classes.
pub mod sc {
    pub const UNIFORM_CONSTANT: u32 = 0;
    pub const INPUT: u32 = 1;
    pub const FUNCTION: u32 = 7;
    pub const PUSH_CONSTANT: u32 = 9;
    pub const WORKGROUP: u32 = 4;
    pub const STORAGE_BUFFER: u32 = 12;
}

pub const GLSL_FMA: u32 = 50;
pub const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;
pub const BUILTIN_LOCAL_INVOCATION_INDEX: u32 = 29;

/// A module under construction: sections kept apart and joined in the order the spec requires.
#[derive(Default)]
pub struct Module {
    next: u32,
    caps: Vec<u32>,
    imports: Vec<u32>,
    model: Vec<u32>,
    entry: Vec<u32>,
    modes: Vec<u32>,
    decorations: Vec<u32>,
    types: Vec<u32>,
    code: Vec<u32>,
}

fn inst(out: &mut Vec<u32>, opcode: u16, operands: &[u32]) {
    out.push(((operands.len() as u32 + 1) << 16) | opcode as u32);
    out.extend_from_slice(operands);
}

/// A literal string as SPIR-V words (NUL-terminated, zero-padded to a word).
pub fn string_words(s: &str) -> Vec<u32> {
    let mut b = s.as_bytes().to_vec();
    b.push(0);
    while !b.len().is_multiple_of(4) {
        b.push(0);
    }
    b.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

impl Module {
    pub fn new() -> Self {
        Module { next: 1, ..Default::default() }
    }
    pub fn id(&mut self) -> u32 {
        self.next += 1;
        self.next - 1
    }
    pub fn capability(&mut self, c: u32) {
        inst(&mut self.caps, op::CAPABILITY, &[c]);
    }
    pub fn ext_import(&mut self, name: &str) -> u32 {
        let id = self.id();
        let mut o = vec![id];
        o.extend(string_words(name));
        inst(&mut self.imports, op::EXT_INST_IMPORT, &o);
        id
    }
    pub fn memory_model_glsl450(&mut self) {
        inst(&mut self.model, op::MEMORY_MODEL, &[0, 1]); // Logical, GLSL450
    }
    pub fn entry_point_compute(&mut self, func: u32, name: &str, interface: &[u32], local: [u32; 3]) {
        let mut o = vec![5, func]; // GLCompute
        o.extend(string_words(name));
        o.extend_from_slice(interface);
        inst(&mut self.entry, op::ENTRY_POINT, &o);
        inst(&mut self.modes, op::EXECUTION_MODE, &[func, 17, local[0], local[1], local[2]]); // LocalSize
    }
    pub fn decorate(&mut self, target: u32, d: &[u32]) {
        let mut o = vec![target];
        o.extend_from_slice(d);
        inst(&mut self.decorations, op::DECORATE, &o);
    }
    pub fn member_decorate(&mut self, st: u32, member: u32, d: &[u32]) {
        let mut o = vec![st, member];
        o.extend_from_slice(d);
        inst(&mut self.decorations, op::MEMBER_DECORATE, &o);
    }
    /// A type or constant declaration: `opcode result operands…`, returning the result id.
    pub fn ty(&mut self, opcode: u16, operands: &[u32]) -> u32 {
        let id = self.id();
        let mut o = vec![id];
        o.extend_from_slice(operands);
        inst(&mut self.types, opcode, &o);
        id
    }
    /// An instruction with a result type in the types section (constants, global variables): unlike `OpType*`,
    /// these take the result *type* first, then the result id.
    fn typed(&mut self, opcode: u16, result_ty: u32, operands: &[u32]) -> u32 {
        let id = self.id();
        let mut o = vec![result_ty, id];
        o.extend_from_slice(operands);
        inst(&mut self.types, opcode, &o);
        id
    }
    /// A global variable (types section).
    pub fn global(&mut self, ptr_ty: u32, storage: u32) -> u32 {
        self.typed(op::VARIABLE, ptr_ty, &[storage])
    }
    pub fn const_u32(&mut self, ty: u32, v: u32) -> u32 {
        self.typed(op::CONSTANT, ty, &[v])
    }
    /// An instruction with a result, in the function body.
    pub fn op(&mut self, opcode: u16, result_ty: u32, operands: &[u32]) -> u32 {
        let id = self.id();
        let mut o = vec![result_ty, id];
        o.extend_from_slice(operands);
        inst(&mut self.code, opcode, &o);
        if opcode == op::F_MUL || opcode == op::F_ADD || opcode == op::F_SUB || opcode == op::EXT_INST {
            self.decorate(id, &[dec::NO_CONTRACTION]);
        }
        id
    }
    /// An instruction without a result, in the function body.
    pub fn stmt(&mut self, opcode: u16, operands: &[u32]) {
        inst(&mut self.code, opcode, operands);
    }
    pub fn label(&mut self, id: u32) {
        inst(&mut self.code, op::LABEL, &[id]);
    }

    /// Knuth's TwoSum: `s = a + b` rounded and the exact error `e` (`s + e = a + b`), without an FMA.
    pub fn two_sum(&mut self, f32t: u32, a: u32, b: u32) -> (u32, u32) {
        let s = self.op(op::F_ADD, f32t, &[a, b]);
        let bb = self.op(op::F_SUB, f32t, &[s, a]);
        let sb = self.op(op::F_SUB, f32t, &[s, bb]);
        let ea = self.op(op::F_SUB, f32t, &[a, sb]);
        let eb = self.op(op::F_SUB, f32t, &[b, bb]);
        (s, self.op(op::F_ADD, f32t, &[ea, eb]))
    }

    /// A correctly rounded `fma(a, b, c)` from plain multiplies and adds, independent of the driver's `Fma`.
    ///
    /// `a` must carry at most 12 significant bits (an f16 value does): `b` is split into two 12-bit halves so both
    /// partial products are exact, TwoSum keeps every rounding error, and Boldo and Melquiond's
    /// `RN(th + RO(tl + ul))` (rounding to odd, 2008) gives the FMA's single rounding. `c_mask` is the u32 constant
    /// `0xFFFF_F000`; `c1` and `c0` are u32 1 and 0; `cf0` is f32 0.
    #[allow(clippy::too_many_arguments)]
    pub fn fma_exact(&mut self, tys: (u32, u32, u32), a: u32, b: u32, c: u32, c_mask: u32, c1: u32, c0: u32, cf0: u32) -> u32 {
        let (f32t, u32t, tbool) = tys;
        let bb = self.op(op::BITCAST, u32t, &[b]);
        let hb = self.op(op::BITWISE_AND, u32t, &[bb, c_mask]);
        let bhi = self.op(op::BITCAST, f32t, &[hb]);
        let blo = self.op(op::F_SUB, f32t, &[b, bhi]);
        let p1 = self.op(op::F_MUL, f32t, &[a, bhi]);
        let p2 = self.op(op::F_MUL, f32t, &[a, blo]);
        let (uh, ul) = self.two_sum(f32t, p1, p2);
        let (th, tl) = self.two_sum(f32t, c, uh);
        let (v, ve) = self.two_sum(f32t, tl, ul);
        // Round v to odd: when inexact and its last bit is even, step one ulp toward the lost part.
        let vb = self.op(op::BITCAST, u32t, &[v]);
        let last = self.op(op::BITWISE_AND, u32t, &[vb, c1]);
        let even = self.op(op::I_EQUAL, tbool, &[last, c0]);
        let inexact = self.op(op::F_ORD_NOT_EQUAL, tbool, &[ve, cf0]);
        let vpos = self.op(op::F_ORD_GREATER_THAN, tbool, &[v, cf0]);
        let epos = self.op(op::F_ORD_GREATER_THAN, tbool, &[ve, cf0]);
        let away = self.op(op::LOGICAL_EQUAL, tbool, &[vpos, epos]);
        let up = self.op(op::I_ADD, u32t, &[vb, c1]);
        let down = self.op(op::I_SUB, u32t, &[vb, c1]);
        let step = self.op(op::SELECT, u32t, &[away, up, down]);
        let fix = self.op(op::LOGICAL_AND, tbool, &[even, inexact]);
        let vo = self.op(op::SELECT, u32t, &[fix, step, vb]);
        let v_odd = self.op(op::BITCAST, f32t, &[vo]);
        self.op(op::F_ADD, f32t, &[th, v_odd])
    }

    /// The finished module: the header (id bound = next id) and the sections in the order the spec requires.
    pub fn words(&self) -> Vec<u32> {
        let mut w = vec![MAGIC, VERSION_1_3, 0, self.next, 0];
        for s in [&self.caps, &self.imports, &self.model, &self.entry, &self.modes, &self.decorations, &self.types, &self.code] {
            w.extend_from_slice(s);
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_and_string_words() {
        assert_eq!(string_words("main"), vec![u32::from_le_bytes(*b"main"), 0]);
        assert_eq!(string_words("GLSL.std.450").len(), 4);
        let m = Module::new();
        assert_eq!(&m.words()[..2], &[MAGIC, VERSION_1_3]);
    }
}
