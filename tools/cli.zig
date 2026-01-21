//! ONNX Zig CLI Tool
//!
//! Command-line interface for ONNX model operations:
//! - Model inspection
//! - Inference benchmarking
//! - Model validation
//!
//! Usage:
//!   onnx-zig info <model.onnx>      - Show model information
//!   onnx-zig bench <model.onnx>     - Benchmark inference speed
//!   onnx-zig run <model.onnx>       - Run inference with random input
//!   onnx-zig validate <model.onnx>  - Validate model can be loaded

const std = @import("std");
const onnx_zig = @import("onnx_zig");

const Session = onnx_zig.Session;
const TensorF32 = onnx_zig.TensorF32;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        printUsage();
        return;
    }

    const command = args[1];

    if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        printUsage();
        return;
    }

    if (std.mem.eql(u8, command, "version") or std.mem.eql(u8, command, "--version") or std.mem.eql(u8, command, "-v")) {
        printVersion();
        return;
    }

    if (args.len < 3) {
        std.debug.print("Error: Missing model path\n\n", .{});
        printUsage();
        return;
    }

    const model_path = args[2];

    if (std.mem.eql(u8, command, "info")) {
        try cmdInfo(allocator, model_path);
    } else if (std.mem.eql(u8, command, "bench")) {
        const iterations = if (args.len > 3) std.fmt.parseInt(usize, args[3], 10) catch 100 else 100;
        try cmdBench(allocator, model_path, iterations);
    } else if (std.mem.eql(u8, command, "run")) {
        try cmdRun(allocator, model_path);
    } else if (std.mem.eql(u8, command, "validate")) {
        try cmdValidate(allocator, model_path);
    } else {
        std.debug.print("Error: Unknown command '{s}'\n\n", .{command});
        printUsage();
    }
}

fn printUsage() void {
    const usage =
        \\ONNX Zig CLI - ONNX inference toolkit
        \\
        \\USAGE:
        \\  onnx-zig <command> [options]
        \\
        \\COMMANDS:
        \\  info <model.onnx>              Show model metadata (inputs, outputs, shapes)
        \\  bench <model.onnx> [iterations] Benchmark inference speed (default: 100 iterations)
        \\  run <model.onnx>               Run single inference with random input
        \\  validate <model.onnx>          Validate model can be loaded
        \\  help                           Show this help message
        \\  version                        Show version information
        \\
        \\EXAMPLES:
        \\  onnx-zig info models/mnist.onnx
        \\  onnx-zig bench models/mnist.onnx 1000
        \\  onnx-zig run models/identity.onnx
        \\
        \\For more information: https://github.com/anthropics/onnx-zig
        \\
    ;
    std.debug.print("{s}", .{usage});
}

fn printVersion() void {
    std.debug.print("onnx-zig 0.1.0\n", .{});
    std.debug.print("Zig ONNX Runtime: {d}\n", .{onnx_zig.onnxruntime.ORT_API_VERSION});
}

fn cmdInfo(allocator: std.mem.Allocator, model_path: []const u8) !void {
    std.debug.print("Loading model: {s}\n", .{model_path});

    var session = Session.init(allocator, model_path) catch |err| {
        std.debug.print("Error: Failed to load model: {s}\n", .{@errorName(err)});
        return;
    };
    defer session.deinit();

    std.debug.print("\n=== Model Information ===\n\n", .{});

    // Input information
    std.debug.print("Inputs ({d}):\n", .{session.getInputCount()});
    for (session.getInputNames(), 0..) |name, i| {
        var info = session.getInputInfo(i) catch continue;
        defer info.deinit(allocator);

        std.debug.print("  [{d}] {s}\n", .{ i, name });
        std.debug.print("      Shape: [", .{});
        for (info.shape.slice(), 0..) |dim, j| {
            if (j > 0) std.debug.print(", ", .{});
            if (dim == 0) {
                std.debug.print("dynamic", .{});
            } else {
                std.debug.print("{d}", .{dim});
            }
        }
        std.debug.print("]\n", .{});
        std.debug.print("      Type: {s}\n", .{dtypeToString(info.dtype)});
    }

    std.debug.print("\nOutputs ({d}):\n", .{session.getOutputCount()});
    for (session.getOutputNames(), 0..) |name, i| {
        var info = session.getOutputInfo(i) catch continue;
        defer info.deinit(allocator);

        std.debug.print("  [{d}] {s}\n", .{ i, name });
        std.debug.print("      Shape: [", .{});
        for (info.shape.slice(), 0..) |dim, j| {
            if (j > 0) std.debug.print(", ", .{});
            if (dim == 0) {
                std.debug.print("dynamic", .{});
            } else {
                std.debug.print("{d}", .{dim});
            }
        }
        std.debug.print("]\n", .{});
        std.debug.print("      Type: {s}\n", .{dtypeToString(info.dtype)});
    }

    std.debug.print("\n", .{});
}

fn cmdBench(allocator: std.mem.Allocator, model_path: []const u8, iterations: usize) !void {
    std.debug.print("Loading model: {s}\n", .{model_path});

    var session = Session.init(allocator, model_path) catch |err| {
        std.debug.print("Error: Failed to load model: {s}\n", .{@errorName(err)});
        return;
    };
    defer session.deinit();

    // Get input shapes and create random data
    const input_count = session.getInputCount();
    const input_data = try allocator.alloc([]f32, input_count);
    defer {
        for (input_data) |d| allocator.free(d);
        allocator.free(input_data);
    }

    const input_shapes = try allocator.alloc([]i64, input_count);
    defer {
        for (input_shapes) |s| allocator.free(s);
        allocator.free(input_shapes);
    }

    var total_elements: usize = 0;
    for (0..input_count) |i| {
        var info = try session.getInputInfo(i);
        defer info.deinit(allocator);

        // Calculate shape (replace dynamic dims with 1)
        const ndim = info.shape.ndim;
        const shape = try allocator.alloc(i64, ndim);
        var numel: usize = 1;
        for (0..ndim) |j| {
            const dim = info.shape.dims[j];
            shape[j] = if (dim == 0) 1 else @intCast(dim);
            numel *= if (dim == 0) 1 else dim;
        }
        input_shapes[i] = shape;

        // Allocate and fill with random data
        const data = try allocator.alloc(f32, numel);
        var prng = std.Random.DefaultPrng.init(42);
        for (data) |*v| {
            v.* = prng.random().float(f32) * 2.0 - 1.0;
        }
        input_data[i] = data;
        total_elements += numel;
    }

    std.debug.print("Running {d} iterations...\n\n", .{iterations});

    // Warmup
    for (0..5) |_| {
        const outputs = session.runF32(input_data, input_shapes) catch continue;
        session.freeOutputs(outputs);
    }

    // Benchmark
    var total_time: u64 = 0;
    var min_time: u64 = std.math.maxInt(u64);
    var max_time: u64 = 0;

    for (0..iterations) |_| {
        const start = std.time.nanoTimestamp();
        const outputs = session.runF32(input_data, input_shapes) catch continue;
        const end = std.time.nanoTimestamp();

        session.freeOutputs(outputs);

        const elapsed: u64 = @intCast(end - start);
        total_time += elapsed;
        min_time = @min(min_time, elapsed);
        max_time = @max(max_time, elapsed);
    }

    const avg_time_us = @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(iterations)) / 1000.0;
    const min_time_us = @as(f64, @floatFromInt(min_time)) / 1000.0;
    const max_time_us = @as(f64, @floatFromInt(max_time)) / 1000.0;
    const throughput = 1000000.0 / avg_time_us;

    std.debug.print("=== Benchmark Results ===\n\n", .{});
    std.debug.print("Iterations:    {d}\n", .{iterations});
    std.debug.print("Total inputs:  {d} elements\n", .{total_elements});
    std.debug.print("\n", .{});
    std.debug.print("Average time:  {d:.2} us\n", .{avg_time_us});
    std.debug.print("Min time:      {d:.2} us\n", .{min_time_us});
    std.debug.print("Max time:      {d:.2} us\n", .{max_time_us});
    std.debug.print("Throughput:    {d:.2} inferences/sec\n", .{throughput});
}

fn cmdRun(allocator: std.mem.Allocator, model_path: []const u8) !void {
    std.debug.print("Loading model: {s}\n", .{model_path});

    var session = Session.init(allocator, model_path) catch |err| {
        std.debug.print("Error: Failed to load model: {s}\n", .{@errorName(err)});
        return;
    };
    defer session.deinit();

    // Get input shapes and create random data
    const input_count = session.getInputCount();
    const input_data = try allocator.alloc([]f32, input_count);
    defer {
        for (input_data) |d| allocator.free(d);
        allocator.free(input_data);
    }

    const input_shapes = try allocator.alloc([]i64, input_count);
    defer {
        for (input_shapes) |s| allocator.free(s);
        allocator.free(input_shapes);
    }

    std.debug.print("\n=== Inputs ===\n", .{});
    for (0..input_count) |i| {
        var info = try session.getInputInfo(i);
        defer info.deinit(allocator);

        const ndim = info.shape.ndim;
        const shape = try allocator.alloc(i64, ndim);
        var numel: usize = 1;
        for (0..ndim) |j| {
            const dim = info.shape.dims[j];
            shape[j] = if (dim == 0) 1 else @intCast(dim);
            numel *= if (dim == 0) 1 else dim;
        }
        input_shapes[i] = shape;

        const data = try allocator.alloc(f32, numel);
        var prng = std.Random.DefaultPrng.init(42);
        for (data) |*v| {
            v.* = prng.random().float(f32) * 2.0 - 1.0;
        }
        input_data[i] = data;

        std.debug.print("{s}: shape=[", .{session.getInputNames()[i]});
        for (shape, 0..) |dim, j| {
            if (j > 0) std.debug.print(", ", .{});
            std.debug.print("{d}", .{dim});
        }
        std.debug.print("], {d} elements (random)\n", .{numel});
    }

    std.debug.print("\nRunning inference...\n", .{});

    const start = std.time.nanoTimestamp();
    const outputs = try session.runF32(input_data, input_shapes);
    const end = std.time.nanoTimestamp();
    defer session.freeOutputs(outputs);

    const elapsed_us = @as(f64, @floatFromInt(end - start)) / 1000.0;

    std.debug.print("\n=== Outputs ({d:.2} us) ===\n", .{elapsed_us});
    for (outputs, 0..) |output, i| {
        std.debug.print("{s}: shape=[", .{session.getOutputNames()[i]});
        for (output.shape.slice(), 0..) |dim, j| {
            if (j > 0) std.debug.print(", ", .{});
            std.debug.print("{d}", .{dim});
        }
        std.debug.print("], {d} elements\n", .{output.numel()});

        // Print first few values
        const max_print = @min(output.numel(), 10);
        std.debug.print("  First {d} values: [", .{max_print});
        for (0..max_print) |j| {
            if (j > 0) std.debug.print(", ", .{});
            std.debug.print("{d:.4}", .{output.data[j]});
        }
        if (output.numel() > max_print) {
            std.debug.print(", ...", .{});
        }
        std.debug.print("]\n", .{});
    }
}

fn cmdValidate(allocator: std.mem.Allocator, model_path: []const u8) !void {
    std.debug.print("Validating model: {s}\n", .{model_path});

    var session = Session.init(allocator, model_path) catch |err| {
        std.debug.print("\nValidation FAILED: {s}\n", .{@errorName(err)});
        return;
    };
    defer session.deinit();

    std.debug.print("\nValidation PASSED\n", .{});
    std.debug.print("  Inputs:  {d}\n", .{session.getInputCount()});
    std.debug.print("  Outputs: {d}\n", .{session.getOutputCount()});
}

fn dtypeToString(dtype: c_uint) []const u8 {
    const c = onnx_zig.c;
    return switch (dtype) {
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => "float32",
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => "float64",
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => "int8",
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => "int16",
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => "int32",
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => "int64",
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => "uint8",
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => "uint16",
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => "uint32",
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => "uint64",
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => "bool",
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => "float16",
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => "bfloat16",
        else => "unknown",
    };
}
