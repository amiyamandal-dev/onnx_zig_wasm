//! C ABI Exports for ONNX Zig Library
//!
//! This module provides C-compatible function exports for FFI integration
//! with other languages (Swift, Java, Node.js, etc.).

const std = @import("std");
const session_mod = @import("session.zig");
const tensor_mod = @import("tensor.zig");

const Session = session_mod.Session;
const SessionOptions = session_mod.SessionOptions;
const TensorF32 = tensor_mod.TensorF32;
const SimdOps = tensor_mod.SimdOps;

// =============================================================================
// Error Codes (must match onnx_zig.h)
// =============================================================================

pub const OnnxZigError = enum(c_int) {
    ok = 0,
    allocation_failed = 1,
    invalid_input = 2,
    invalid_output = 3,
    model_not_found = 4,
    invalid_model = 5,
    inference_failed = 6,
    shape_mismatch = 7,
    null_pointer = 8,
    unknown = 99,
};

// =============================================================================
// Global State
// =============================================================================

var global_allocator: std.mem.Allocator = std.heap.c_allocator;
var last_error_message: [512]u8 = [_]u8{0} ** 512;

fn setLastError(comptime fmt: []const u8, args: anytype) void {
    _ = std.fmt.bufPrint(&last_error_message, fmt, args) catch {};
}

// =============================================================================
// Session Handle Wrapper
// =============================================================================

const SessionWrapper = struct {
    session: Session,
    output_tensors: ?[]TensorF32 = null,
    output_handles: ?[]*TensorWrapper = null,
};

const TensorWrapper = struct {
    tensor: TensorF32,
    is_output: bool = false,
};

// =============================================================================
// Session API Exports
// =============================================================================

export fn onnx_zig_session_create(
    model_path: [*:0]const u8,
    out_session: *?*SessionWrapper,
) OnnxZigError {
    if (model_path == null) {
        setLastError("model_path is null", .{});
        return .null_pointer;
    }

    const wrapper = global_allocator.create(SessionWrapper) catch {
        setLastError("Failed to allocate session wrapper", .{});
        return .allocation_failed;
    };

    wrapper.* = SessionWrapper{
        .session = Session.init(global_allocator, std.mem.span(model_path)) catch |err| {
            setLastError("Failed to create session: {s}", .{@errorName(err)});
            global_allocator.destroy(wrapper);
            return switch (err) {
                session_mod.SessionError.OrtNoSuchFile => .model_not_found,
                session_mod.SessionError.OrtInvalidProtobuf, session_mod.SessionError.OrtInvalidGraph => .invalid_model,
                session_mod.SessionError.AllocationFailed => .allocation_failed,
                else => .unknown,
            };
        },
    };

    out_session.* = wrapper;
    return .ok;
}

export fn onnx_zig_session_create_with_options(
    model_path: [*:0]const u8,
    intra_op_threads: u32,
    inter_op_threads: u32,
    out_session: *?*SessionWrapper,
) OnnxZigError {
    if (model_path == null) {
        setLastError("model_path is null", .{});
        return .null_pointer;
    }

    const wrapper = global_allocator.create(SessionWrapper) catch {
        setLastError("Failed to allocate session wrapper", .{});
        return .allocation_failed;
    };

    const options = SessionOptions{
        .intra_op_num_threads = intra_op_threads,
        .inter_op_num_threads = inter_op_threads,
    };

    wrapper.* = SessionWrapper{
        .session = Session.initWithOptions(global_allocator, std.mem.span(model_path), options) catch |err| {
            setLastError("Failed to create session: {s}", .{@errorName(err)});
            global_allocator.destroy(wrapper);
            return switch (err) {
                session_mod.SessionError.OrtNoSuchFile => .model_not_found,
                session_mod.SessionError.OrtInvalidProtobuf, session_mod.SessionError.OrtInvalidGraph => .invalid_model,
                session_mod.SessionError.AllocationFailed => .allocation_failed,
                else => .unknown,
            };
        },
    };

    out_session.* = wrapper;
    return .ok;
}

export fn onnx_zig_session_destroy(session: ?*SessionWrapper) void {
    if (session) |s| {
        // Free any cached output tensors
        if (s.output_tensors) |tensors| {
            for (tensors) |*t| {
                var t_mut = t.*;
                t_mut.deinit();
            }
            global_allocator.free(tensors);
        }
        if (s.output_handles) |handles| {
            for (handles) |h| {
                global_allocator.destroy(h);
            }
            global_allocator.free(handles);
        }

        s.session.deinit();
        global_allocator.destroy(s);
    }
}

export fn onnx_zig_session_get_input_count(session: ?*SessionWrapper) usize {
    if (session) |s| {
        return s.session.getInputCount();
    }
    return 0;
}

export fn onnx_zig_session_get_output_count(session: ?*SessionWrapper) usize {
    if (session) |s| {
        return s.session.getOutputCount();
    }
    return 0;
}

export fn onnx_zig_session_get_input_name(session: ?*SessionWrapper, index: usize) ?[*:0]const u8 {
    if (session) |s| {
        const names = s.session.getInputNames();
        if (index < names.len) {
            return names[index].ptr;
        }
    }
    return null;
}

export fn onnx_zig_session_get_output_name(session: ?*SessionWrapper, index: usize) ?[*:0]const u8 {
    if (session) |s| {
        const names = s.session.getOutputNames();
        if (index < names.len) {
            return names[index].ptr;
        }
    }
    return null;
}

// =============================================================================
// Inference API Exports
// =============================================================================

export fn onnx_zig_session_run_f32_simple(
    session: ?*SessionWrapper,
    input_data: ?[*]const f32,
    input_shape: ?[*]const i64,
    shape_dims: usize,
    out_tensors: *?[*]*TensorWrapper,
    out_count: *usize,
) OnnxZigError {
    if (session == null) {
        setLastError("session is null", .{});
        return .null_pointer;
    }
    if (input_data == null or input_shape == null) {
        setLastError("input_data or input_shape is null", .{});
        return .null_pointer;
    }

    const s = session.?;
    const shape_slice = input_shape.?[0..shape_dims];

    // Calculate input size
    var numel: usize = 1;
    for (shape_slice) |dim| {
        numel *= @intCast(dim);
    }

    // Run inference
    const outputs = s.session.runF32Simple(input_data.?[0..numel], shape_slice) catch |err| {
        setLastError("Inference failed: {s}", .{@errorName(err)});
        return .inference_failed;
    };

    // Free previous outputs if any
    if (s.output_tensors) |tensors| {
        for (tensors) |*t| {
            var t_mut = t.*;
            t_mut.deinit();
        }
        global_allocator.free(tensors);
    }
    if (s.output_handles) |handles| {
        for (handles) |h| {
            global_allocator.destroy(h);
        }
        global_allocator.free(handles);
    }

    // Store new outputs
    s.output_tensors = outputs;

    // Create handles array
    const handles = global_allocator.alloc(*TensorWrapper, outputs.len) catch {
        setLastError("Failed to allocate output handles", .{});
        return .allocation_failed;
    };

    for (outputs, 0..) |output, i| {
        const wrapper = global_allocator.create(TensorWrapper) catch {
            setLastError("Failed to allocate tensor wrapper", .{});
            return .allocation_failed;
        };
        wrapper.* = TensorWrapper{
            .tensor = output,
            .is_output = true,
        };
        handles[i] = wrapper;
    }

    s.output_handles = handles;
    out_tensors.* = handles.ptr;
    out_count.* = outputs.len;

    return .ok;
}

// =============================================================================
// Tensor API Exports
// =============================================================================

export fn onnx_zig_tensor_create_f32(
    shape: ?[*]const usize,
    ndim: usize,
    out_tensor: *?*TensorWrapper,
) OnnxZigError {
    if (shape == null) {
        setLastError("shape is null", .{});
        return .null_pointer;
    }

    const wrapper = global_allocator.create(TensorWrapper) catch {
        setLastError("Failed to allocate tensor wrapper", .{});
        return .allocation_failed;
    };

    wrapper.* = TensorWrapper{
        .tensor = TensorF32.init(global_allocator, shape.?[0..ndim]) catch {
            setLastError("Failed to allocate tensor", .{});
            global_allocator.destroy(wrapper);
            return .allocation_failed;
        },
    };

    out_tensor.* = wrapper;
    return .ok;
}

export fn onnx_zig_tensor_create_f32_from_data(
    data: ?[*]const f32,
    shape: ?[*]const usize,
    ndim: usize,
    out_tensor: *?*TensorWrapper,
) OnnxZigError {
    if (data == null or shape == null) {
        setLastError("data or shape is null", .{});
        return .null_pointer;
    }

    const wrapper = global_allocator.create(TensorWrapper) catch {
        setLastError("Failed to allocate tensor wrapper", .{});
        return .allocation_failed;
    };

    const shape_slice = shape.?[0..ndim];
    wrapper.* = TensorWrapper{
        .tensor = TensorF32.fromSlice(global_allocator, data.?[0..tensor_mod.calcNumel(shape_slice)], shape_slice) catch {
            setLastError("Failed to allocate tensor", .{});
            global_allocator.destroy(wrapper);
            return .allocation_failed;
        },
    };

    out_tensor.* = wrapper;
    return .ok;
}

export fn onnx_zig_tensor_destroy(tensor: ?*TensorWrapper) void {
    if (tensor) |t| {
        if (!t.is_output) {
            t.tensor.deinit();
        }
        global_allocator.destroy(t);
    }
}

export fn onnx_zig_tensors_destroy(tensors: ?[*]*TensorWrapper, count: usize) void {
    if (tensors) |t| {
        for (0..count) |i| {
            onnx_zig_tensor_destroy(t[i]);
        }
    }
}

export fn onnx_zig_tensor_get_data_f32(tensor: ?*TensorWrapper) ?[*]const f32 {
    if (tensor) |t| {
        return t.tensor.data.ptr;
    }
    return null;
}

export fn onnx_zig_tensor_get_data_f32_mut(tensor: ?*TensorWrapper) ?[*]f32 {
    if (tensor) |t| {
        return t.tensor.data.ptr;
    }
    return null;
}

export fn onnx_zig_tensor_get_numel(tensor: ?*TensorWrapper) usize {
    if (tensor) |t| {
        return t.tensor.numel();
    }
    return 0;
}

export fn onnx_zig_tensor_get_ndim(tensor: ?*TensorWrapper) usize {
    if (tensor) |t| {
        return t.tensor.shape.ndim;
    }
    return 0;
}

export fn onnx_zig_tensor_get_dim(tensor: ?*TensorWrapper, dim: usize) usize {
    if (tensor) |t| {
        if (dim < t.tensor.shape.ndim) {
            return t.tensor.shape.dims[dim];
        }
    }
    return 0;
}

export fn onnx_zig_tensor_get_shape(tensor: ?*TensorWrapper, out_shape: ?[*]usize, max_dims: usize) usize {
    if (tensor == null or out_shape == null) return 0;

    const t = tensor.?;
    const copy_dims = @min(t.tensor.shape.ndim, max_dims);

    for (0..copy_dims) |i| {
        out_shape.?[i] = t.tensor.shape.dims[i];
    }

    return copy_dims;
}

// =============================================================================
// Tensor Operations Exports
// =============================================================================

export fn onnx_zig_matmul(
    a: ?*TensorWrapper,
    b: ?*TensorWrapper,
    out_result: *?*TensorWrapper,
) OnnxZigError {
    if (a == null or b == null) {
        setLastError("input tensor is null", .{});
        return .null_pointer;
    }

    const result_tensor = tensor_mod.matmul(global_allocator, &a.?.tensor, &b.?.tensor) catch |err| {
        setLastError("matmul failed: {s}", .{@errorName(err)});
        return .shape_mismatch;
    };

    const wrapper = global_allocator.create(TensorWrapper) catch {
        setLastError("Failed to allocate result wrapper", .{});
        return .allocation_failed;
    };

    wrapper.* = TensorWrapper{ .tensor = result_tensor };
    out_result.* = wrapper;
    return .ok;
}

export fn onnx_zig_transpose(
    tensor: ?*TensorWrapper,
    out_result: *?*TensorWrapper,
) OnnxZigError {
    if (tensor == null) {
        setLastError("input tensor is null", .{});
        return .null_pointer;
    }

    const result_tensor = tensor_mod.transpose(global_allocator, &tensor.?.tensor) catch |err| {
        setLastError("transpose failed: {s}", .{@errorName(err)});
        return .shape_mismatch;
    };

    const wrapper = global_allocator.create(TensorWrapper) catch {
        setLastError("Failed to allocate result wrapper", .{});
        return .allocation_failed;
    };

    wrapper.* = TensorWrapper{ .tensor = result_tensor };
    out_result.* = wrapper;
    return .ok;
}

export fn onnx_zig_softmax_inplace(tensor: ?*TensorWrapper) void {
    if (tensor) |t| {
        SimdOps.softmax(t.tensor.data, t.tensor.data);
    }
}

export fn onnx_zig_argmax(tensor: ?*TensorWrapper) usize {
    if (tensor) |t| {
        return SimdOps.argmax(t.tensor.data);
    }
    return 0;
}

// =============================================================================
// Utility Exports
// =============================================================================

export fn onnx_zig_version() [*:0]const u8 {
    return "0.1.0";
}

export fn onnx_zig_get_last_error() [*:0]const u8 {
    return @ptrCast(&last_error_message);
}
