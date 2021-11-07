//! This file contains the functionality for lowering x86_64 MIR into
//! machine code

const Emit = @This();
const std = @import("std");
const log = std.log.scoped(.codegen);
const math = std.math;
const mem = std.mem;
const Mir = @import("Mir.zig");
const bits = @import("bits.zig");
const link = @import("../../link.zig");
const ErrorMsg = Module.ErrorMsg;
const Module = @import("../../Module.zig");
const assert = std.debug.assert;
const DW = std.dwarf;
const leb128 = std.leb;
const Instruction = bits.Instruction;
const Register = bits.Register;
const DebugInfoOutput = @import("../../codegen.zig").DebugInfoOutput;
const Encoder = bits.Encoder;

mir: Mir,
bin_file: *link.File,
debug_output: DebugInfoOutput,
target: *const std.Target,
err_msg: ?*ErrorMsg = null,
src_loc: Module.SrcLoc,
code: *std.ArrayList(u8),

prev_di_line: u32,
prev_di_column: u32,
/// Relative to the beginning of `code`.
prev_di_pc: usize,

const InnerError = error{
    OutOfMemory,
    EmitFail,
};

pub fn emitMir(emit: *Emit) InnerError!void {
    const mir_tags = emit.mir.instructions.items(.tag);

    for (mir_tags) |tag, index| {
        const inst = @intCast(u32, index);
        switch (tag) {
            .mov => try emit.mirMov(inst),
            .sub => try emit.mirSub(inst),
            .push => try emit.mirPush(inst),
            .pop => try emit.mirPop(inst),
            .ret => try emit.mirRet(inst),
            .movabs => try emit.mirMovabs(inst),
            else => {
                return emit.fail("Implement MIR->Isel lowering for x86_64 for pseudo-inst: {s}", .{tag});
            },
        }
    }
}

fn fail(emit: *Emit, comptime format: []const u8, args: anytype) InnerError {
    @setCold(true);
    assert(emit.err_msg == null);
    emit.err_msg = try ErrorMsg.create(emit.bin_file.allocator, emit.src_loc, format, args);
    return error.EmitFail;
}

fn mirPushPop(emit: *Emit, tag: Mir.Inst.Tag, inst: Mir.Inst.Index) InnerError!void {
    const ops = Mir.Ops.decode(emit.mir.instructions.items(.ops)[inst]);
    if (@truncate(u1, ops.flags) == 0b0) {
        // PUSH/POP reg
        const opc: u8 = switch (tag) {
            .push => 0x50,
            .pop => 0x58,
            else => unreachable,
        };
        const encoder = try Encoder.init(emit.code, 1);
        encoder.opcode_withReg(opc, ops.reg1.lowId());
    } else {
        // PUSH/POP r/m64
        const imm = emit.mir.instructions.items(.data)[inst].imm;
        const opc: u8 = switch (tag) {
            .push => 0xff,
            .pop => 0x8f,
            else => unreachable,
        };
        const modrm_ext: u3 = switch (tag) {
            .push => 0x6,
            .pop => 0x0,
            else => unreachable,
        };
        const encoder = try Encoder.init(emit.code, 6);
        encoder.opcode_1byte(opc);
        if (math.cast(i8, imm)) |imm_i8| {
            encoder.modRm_indirectDisp8(modrm_ext, ops.reg1.lowId());
            encoder.imm8(@intCast(i8, imm_i8));
        } else |_| {
            encoder.modRm_indirectDisp32(modrm_ext, ops.reg1.lowId());
            encoder.imm32(imm);
        }
    }
}

fn mirPush(emit: *Emit, inst: Mir.Inst.Index) InnerError!void {
    const tag = emit.mir.instructions.items(.tag)[inst];
    assert(tag == .push);
    return emit.mirPushPop(tag, inst);
}

fn mirPop(emit: *Emit, inst: Mir.Inst.Index) InnerError!void {
    const tag = emit.mir.instructions.items(.tag)[inst];
    assert(tag == .pop);
    return emit.mirPushPop(tag, inst);
}

fn mirRet(emit: *Emit, inst: Mir.Inst.Index) InnerError!void {
    const tag = emit.mir.instructions.items(.tag)[inst];
    const ops = Mir.Ops.decode(emit.mir.instructions.items(.ops)[inst]);
    assert(tag == .ret);
    const encoder = try Encoder.init(emit.code, 3);
    switch (ops.flags) {
        0b00 => {
            // RETF imm16
            const imm = emit.mir.instructions.items(.data)[inst].imm;
            encoder.opcode_1byte(0xca);
            encoder.imm16(@intCast(i16, imm));
        },
        0b01 => encoder.opcode_1byte(0xcb), // RETF
        0b10 => {
            // RET imm16
            const imm = emit.mir.instructions.items(.data)[inst].imm;
            encoder.opcode_1byte(0xc2);
            encoder.imm16(@intCast(i16, imm));
        },
        0b11 => encoder.opcode_1byte(0xc3), // RET
    }
}

const OpCode = struct {
    opc: u8,
    modrm_ext: u3,
};
fn getOpCode(tag: Mir.Inst.Tag, ops: Mir.Ops) OpCode {
    return switch (ops.flags) {
        0b00 => if (ops.reg2 == .none) switch (tag) {
            .mov => OpCode{ .opc = 0xc7, .modrm_ext = 0x0 },
            .sub => OpCode{ .opc = 0x81, .modrm_ext = 0x5 },
            else => unreachable,
        } else switch (tag) {
            .mov => OpCode{ .opc = 0x89, .modrm_ext = ops.reg2.lowId() },
            .sub => OpCode{ .opc = 0x29, .modrm_ext = ops.reg2.lowId() },
            else => unreachable,
        },
        0b01 => switch (tag) {
            .mov => OpCode{ .opc = 0x8b, .modrm_ext = ops.reg2.lowId() },
            else => unreachable,
        },
        0b10 => if (ops.reg2 == .none) switch (tag) {
            .mov => OpCode{ .opc = 0xc7, .modrm_ext = 0x0 },
            else => unreachable,
        } else switch (tag) {
            .mov => OpCode{ .opc = 0x89, .modrm_ext = ops.reg2.lowId() },
            else => unreachable,
        },
        0b11 => switch (tag) {
            .mov => OpCode{ .opc = 0xc7, .modrm_ext = 0x0 },
            else => unreachable,
        },
    };
}

fn mirCommonOp(emit: *Emit, tag: Mir.Inst.Tag, inst: Mir.Inst.Index) InnerError!void {
    const ops = Mir.Ops.decode(emit.mir.instructions.items(.ops)[inst]);
    const opcode = getOpCode(tag, ops);
    switch (ops.flags) {
        0b00 => blk: {
            if (ops.reg2 == .none) {
                // OP reg1, imm32
                // OP r/m, imm32
                const imm = emit.mir.instructions.items(.data)[inst].imm;
                const encoder = try Encoder.init(emit.code, 7);
                encoder.rex(.{
                    .w = ops.reg1.size() == 64,
                    .r = ops.reg1.isExtended(),
                });
                encoder.opcode_1byte(opcode.opc);
                encoder.modRm_direct(opcode.modrm_ext, ops.reg1.lowId());
                encoder.imm32(imm);
                break :blk;
            }
            // OP reg1, reg2
            // OP r/m, r
            const encoder = try Encoder.init(emit.code, 3);
            encoder.rex(.{
                .w = ops.reg1.size() == 64,
                .r = ops.reg2.isExtended(),
                .b = ops.reg1.isExtended(),
            });
            encoder.opcode_1byte(opcode.opc);
            encoder.modRm_direct(opcode.modrm_ext, ops.reg1.lowId());
        },
        0b01 => {
            // OP reg1, [reg2 + imm32]
            // OP r64, r/m64
            const imm = emit.mir.instructions.items(.data)[inst].imm;
            const encoder = try Encoder.init(emit.code, 7);
            encoder.rex(.{
                .w = ops.reg1.size() == 64,
                .r = ops.reg2.isExtended(),
                .b = ops.reg1.isExtended(),
            });
            encoder.opcode_1byte(opcode.opc);
            encoder.modRm_indirectDisp32(opcode.modrm_ext, ops.reg1.lowId());
            encoder.disp32(imm);
        },
        0b10 => blk: {
            if (ops.reg2 == .none) {
                // OP [reg1 + 0], imm32
                // OP r/m64, imm32
                const imm = emit.mir.instructions.items(.data)[inst].imm;
                const encoder = try Encoder.init(emit.code, 7);
                encoder.rex(.{
                    .w = ops.reg1.size() == 64,
                    .b = ops.reg1.isExtended(),
                });
                encoder.opcode_1byte(opcode.opc);
                encoder.modRm_indirectDisp0(opcode.modrm_ext, ops.reg1.lowId());
                encoder.imm32(imm);
                break :blk;
            }
            // OP [reg1 + imm32], reg2
            // OP r/m64, r64
            const imm = emit.mir.instructions.items(.data)[inst].imm;
            const encoder = try Encoder.init(emit.code, 7);
            encoder.rex(.{
                .w = ops.reg1.size() == 64,
                .r = ops.reg2.isExtended(),
                .b = ops.reg1.isExtended(),
            });
            encoder.opcode_1byte(opcode.opc);
            encoder.modRm_indirectDisp32(opcode.modrm_ext, ops.reg1.lowId());
            encoder.disp32(imm);
        },
        0b11 => {
            // OP [reg1 + imm32], imm32
            // OP r/m64, imm32
            const payload = emit.mir.instructions.items(.data)[inst].payload;
            const imm_pair = emit.mir.extraData(Mir.ImmPair, payload).data;

            if (imm_pair.dest_off <= math.maxInt(i8)) {
                const encoder = try Encoder.init(emit.code, 8);
                encoder.rex(.{
                    .w = ops.reg1.size() == 64,
                    .b = ops.reg1.isExtended(),
                });
                encoder.opcode_1byte(opcode.opc);
                encoder.modRm_indirectDisp8(opcode.modrm_ext, ops.reg1.lowId());
                encoder.disp8(@intCast(i8, imm_pair.dest_off));
                encoder.imm32(imm_pair.operand);
            } else {
                const encoder = try Encoder.init(emit.code, 11);
                encoder.rex(.{
                    .w = ops.reg1.size() == 64,
                    .b = ops.reg1.isExtended(),
                });
                encoder.opcode_1byte(opcode.opc);
                encoder.modRm_indirectDisp32(opcode.modrm_ext, ops.reg1.lowId());
                encoder.disp32(imm_pair.dest_off);
                encoder.imm32(imm_pair.operand);
            }
        },
    }
}

fn mirMov(emit: *Emit, inst: Mir.Inst.Index) InnerError!void {
    const tag = emit.mir.instructions.items(.tag)[inst];
    assert(tag == .mov);
    return emit.mirCommonOp(tag, inst);
}

fn mirSub(emit: *Emit, inst: Mir.Inst.Index) InnerError!void {
    const tag = emit.mir.instructions.items(.tag)[inst];
    assert(tag == .sub);
    return emit.mirCommonOp(tag, inst);
}

fn mirMovabs(emit: *Emit, inst: Mir.Inst.Index) InnerError!void {
    const tag = emit.mir.instructions.items(.tag)[inst];
    assert(tag == .movabs);
    const ops = Mir.Ops.decode(emit.mir.instructions.items(.ops)[inst]);

    if (ops.reg1.size() == 64) {
        const payload = emit.mir.instructions.items(.data)[inst].payload;
        const imm64 = emit.mir.extraData(Mir.Imm64, payload).data;
        const encoder = try Encoder.init(emit.code, 10);
        encoder.rex(.{
            .w = true,
            .b = ops.reg1.isExtended(),
        });
        encoder.opcode_withReg(0xb8, ops.reg1.lowId());
        encoder.imm64(imm64.decode());
        return;
    }

    const imm = emit.mir.instructions.items(.data)[inst].imm;
    if (imm <= math.maxInt(i8)) {
        const encoder = try Encoder.init(emit.code, 3);
        encoder.rex(.{
            .w = false,
            .b = ops.reg1.isExtended(),
        });
        encoder.opcode_withReg(0xb0, ops.reg1.lowId());
        encoder.imm8(@intCast(i8, imm));
    } else if (imm <= math.maxInt(i16)) {
        const encoder = try Encoder.init(emit.code, 4);
        encoder.rex(.{
            .w = false,
            .b = ops.reg1.isExtended(),
        });
        encoder.opcode_withReg(0xb8, ops.reg1.lowId());
        encoder.imm16(@intCast(i16, imm));
    } else {
        const encoder = try Encoder.init(emit.code, 6);
        encoder.rex(.{
            .w = false,
            .b = ops.reg1.isExtended(),
        });
        encoder.opcode_withReg(0xb8, ops.reg1.lowId());
        encoder.imm32(imm);
    }
}
