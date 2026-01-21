//! Basic Inference Example
//!
//! This example demonstrates how to:
//! 1. Load an ONNX model
//! 2. Prepare input tensors
//! 3. Run inference
//! 4. Process output tensors
//!
//! Build: zig build example-basic
//! Run:   ./zig-out/bin/example-basic models/test/identity.onnx

const std = @import("std");
const onnx_zig = @import("onnx_zig");

const Session = onnx_zig.Session;
const TensorF32 = onnx_zig.TensorF32;

pub fn main() !void {
    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Get model path from command line
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <model.onnx>\n", .{args[0]});
        std.debug.print("\nExample models in models/test/:\n", .{});
        std.debug.print("  - identity.onnx  (Y = X)\n", .{});
        std.debug.print("  - add.onnx       (C = A + B)\n", .{});
        std.debug.print("  - relu.onnx      (Y = max(0, X))\n", .{});
        std.debug.print("  - matmul.onnx    (Y = A @ B)\n", .{});
        return;
    }

    const model_path = args[1];

    // Load the ONNX model
    std.debug.print("Loading model: {s}\n", .{model_path});

    var session = try Session.init(allocator, model_path);
    defer session.deinit();

    // Print model info
    std.debug.print("\nModel loaded successfully!\n", .{});
    std.debug.print("  Inputs:  {d}\n", .{session.getInputCount()});
    std.debug.print("  Outputs: {d}\n", .{session.getOutputCount()});

    for (session.getInputNames(), 0..) |name, i| {
        std.debug.print("  Input [{d}]: {s}\n", .{ i, name });
    }
    for (session.getOutputNames(), 0..) |name, i| {
        std.debug.print("  Output [{d}]: {s}\n", .{ i, name });
    }

    // Prepare input data based on model type
    // For simplicity, we'll create appropriate inputs for common test models

    const input_count = session.getInputCount();
    std.debug.print("\nPreparing {d} input(s)...\n", .{input_count});

    // Get input info and prepare data
    var input_data_list = std.ArrayList([]const f32).init(allocator);
    defer input_data_list.deinit();

    var input_shape_list = std.ArrayList([]const i64).init(allocator);
    defer input_shape_list.deinit();

    var allocated_data = std.ArrayList([]f32).init(allocator);
    defer {
        for (allocated_data.items) |d| allocator.free(d);
        allocated_data.deinit();
    }

    var allocated_shapes = std.ArrayList([]i64).init(allocator);
    defer {
        for (allocated_shapes.items) |s| allocator.free(s);
        allocated_shapes.deinit();
    }

    for (0..input_count) |i| {
        var info = try session.getInputInfo(i);
        defer info.deinit(allocator);

        // Calculate number of elements
        var numel: usize = 1;
        const shape_slice = try allocator.alloc(i64, info.shape.ndim);
        try allocated_shapes.append(shape_slice);

        for (0..info.shape.ndim) |j| {
            const dim = info.shape.dims[j];
            const resolved_dim = if (dim == 0) 1 else dim;
            shape_slice[j] = @intCast(resolved_dim);
            numel *= resolved_dim;
        }

        // Allocate and fill with sequential data
        const data = try allocator.alloc(f32, numel);
        try allocated_data.append(data);

        for (0..numel) |j| {
            data[j] = @as(f32, @floatFromInt(j + 1)) * 0.5;
        }

        try input_data_list.append(data);
        try input_shape_list.append(shape_slice);

        std.debug.print("  {s}: [{any}] = [", .{ info.name, shape_slice });
        const max_print = @min(numel, 6);
        for (0..max_print) |j| {
            if (j > 0) std.debug.print(", ", .{});
            std.debug.print("{d:.1}", .{data[j]});
        }
        if (numel > max_print) std.debug.print(", ...", .{});
        std.debug.print("]\n", .{});
    }

    // Run inference
    std.debug.print("\nRunning inference...\n", .{});

    const start = std.time.nanoTimestamp();
    const outputs = try session.runF32(
        input_data_list.items,
        input_shape_list.items,
    );
    const end = std.time.nanoTimestamp();
    defer session.freeOutputs(outputs);

    const elapsed_us = @as(f64, @floatFromInt(end - start)) / 1000.0;
    std.debug.print("Inference completed in {d:.2} us\n", .{elapsed_us});

    // Print outputs
    std.debug.print("\nOutputs:\n", .{});
    for (outputs, 0..) |output, i| {
        std.debug.print("  {s}: shape=[", .{session.getOutputNames()[i]});
        for (output.shape.slice(), 0..) |dim, j| {
            if (j > 0) std.debug.print(", ", .{});
            std.debug.print("{d}", .{dim});
        }
        std.debug.print("], {d} elements\n", .{output.numel()});

        // Print all values (up to 20)
        const max_print = @min(output.numel(), 20);
        std.debug.print("    Values: [", .{});
        for (0..max_print) |j| {
            if (j > 0) std.debug.print(", ", .{});
            std.debug.print("{d:.4}", .{output.data[j]});
        }
        if (output.numel() > max_print) std.debug.print(", ...", .{});
        std.debug.print("]\n", .{});
    }

    std.debug.print("\nDone!\n", .{});
}
