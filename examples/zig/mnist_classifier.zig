//! MNIST Digit Classifier Example
//!
//! Demonstrates image classification using the MNIST model.
//! Reads a PGM image file and predicts the digit.
//!
//! Build: zig build example-mnist
//! Run:   ./zig-out/bin/example-mnist models/mnist.onnx digit.pgm

const std = @import("std");
const onnx_zig = @import("onnx_zig");

const Session = onnx_zig.Session;
const SimdOps = onnx_zig.SimdOps;

// MNIST normalization constants
const MNIST_MEAN: f32 = 0.1307;
const MNIST_STD: f32 = 0.3081;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        printUsage(args[0]);
        return;
    }

    const model_path = args[1];
    const image_path = if (args.len > 2) args[2] else null;

    // Load model
    std.debug.print("Loading MNIST model: {s}\n", .{model_path});

    var session = Session.init(allocator, model_path) catch |err| {
        std.debug.print("Error loading model: {s}\n", .{@errorName(err)});
        return;
    };
    defer session.deinit();

    std.debug.print("Model loaded successfully!\n", .{});
    std.debug.print("  Input:  {s} [1, 1, 28, 28]\n", .{session.getInputNames()[0]});
    std.debug.print("  Output: {s} [1, 10]\n", .{session.getOutputNames()[0]});

    // Prepare input image
    var image_data: [28 * 28]f32 = undefined;

    if (image_path) |path| {
        // Load image from file
        std.debug.print("\nLoading image: {s}\n", .{path});
        loadPgmImage(path, &image_data) catch |err| {
            std.debug.print("Error loading image: {s}\n", .{@errorName(err)});
            std.debug.print("Using synthetic digit instead.\n", .{});
            generateSyntheticDigit(&image_data, 7);
        };
    } else {
        // Generate a synthetic digit
        std.debug.print("\nNo image provided. Generating synthetic digit '7'...\n", .{});
        generateSyntheticDigit(&image_data, 7);
    }

    // Normalize for MNIST
    for (&image_data) |*v| {
        v.* = (v.* - MNIST_MEAN) / MNIST_STD;
    }

    // Run inference
    std.debug.print("\nRunning inference...\n", .{});

    const input_shape = [_]i64{ 1, 1, 28, 28 };
    const start = std.time.nanoTimestamp();

    const outputs = try session.runF32(
        &[_][]const f32{&image_data},
        &[_][]const i64{&input_shape},
    );
    defer session.freeOutputs(outputs);

    const end = std.time.nanoTimestamp();
    const elapsed_us = @as(f64, @floatFromInt(end - start)) / 1000.0;

    // Get logits and apply softmax
    const logits = outputs[0].data;
    var probs: [10]f32 = undefined;
    softmax(logits, &probs);

    // Find predicted digit
    var max_prob: f32 = 0;
    var predicted: usize = 0;
    for (probs, 0..) |p, i| {
        if (p > max_prob) {
            max_prob = p;
            predicted = i;
        }
    }

    // Print results
    std.debug.print("\n=== Results ({d:.2} us) ===\n", .{elapsed_us});
    std.debug.print("\nPredicted Digit: {d}\n", .{predicted});
    std.debug.print("Confidence:      {d:.1}%\n", .{max_prob * 100});
    std.debug.print("\nAll probabilities:\n", .{});

    for (probs, 0..) |p, i| {
        const bar_len = @as(usize, @intFromFloat(p * 40));
        std.debug.print("  {d}: {d:5.1}% ", .{ i, p * 100 });
        for (0..bar_len) |_| std.debug.print("█", .{});
        for (0..(40 - bar_len)) |_| std.debug.print("░", .{});
        if (i == predicted) {
            std.debug.print(" ◄ PREDICTED", .{});
        }
        std.debug.print("\n", .{});
    }
}

fn printUsage(program: []const u8) void {
    std.debug.print(
        \\MNIST Digit Classifier
        \\
        \\USAGE:
        \\  {s} <model.onnx> [image.pgm]
        \\
        \\ARGUMENTS:
        \\  model.onnx   Path to MNIST ONNX model
        \\  image.pgm    Optional 28x28 PGM grayscale image
        \\
        \\If no image is provided, a synthetic digit will be generated.
        \\
        \\EXAMPLE:
        \\  {s} models/mnist.onnx
        \\  {s} models/mnist.onnx my_digit.pgm
        \\
    , .{ program, program, program });
}

fn softmax(logits: []const f32, probs: *[10]f32) void {
    // Find max for numerical stability
    var max_val: f32 = logits[0];
    for (logits[1..]) |v| {
        if (v > max_val) max_val = v;
    }

    // Compute exp and sum
    var sum: f32 = 0;
    for (logits, 0..) |v, i| {
        probs[i] = @exp(v - max_val);
        sum += probs[i];
    }

    // Normalize
    for (probs) |*p| {
        p.* /= sum;
    }
}

fn generateSyntheticDigit(data: *[28 * 28]f32, digit: u8) void {
    // Clear to black
    @memset(data, 0);

    // Draw simple patterns for digits 0-9
    switch (digit) {
        0 => drawOval(data, 8, 4, 12, 20),
        1 => drawLine(data, 14, 4, 14, 24),
        2 => {
            drawLine(data, 6, 4, 22, 4);
            drawLine(data, 22, 4, 22, 14);
            drawLine(data, 6, 14, 22, 14);
            drawLine(data, 6, 14, 6, 24);
            drawLine(data, 6, 24, 22, 24);
        },
        3 => {
            drawLine(data, 6, 4, 22, 4);
            drawLine(data, 22, 4, 22, 24);
            drawLine(data, 6, 14, 22, 14);
            drawLine(data, 6, 24, 22, 24);
        },
        4 => {
            drawLine(data, 6, 4, 6, 14);
            drawLine(data, 6, 14, 22, 14);
            drawLine(data, 20, 4, 20, 24);
        },
        5 => {
            drawLine(data, 6, 4, 22, 4);
            drawLine(data, 6, 4, 6, 14);
            drawLine(data, 6, 14, 22, 14);
            drawLine(data, 22, 14, 22, 24);
            drawLine(data, 6, 24, 22, 24);
        },
        6 => {
            drawLine(data, 6, 4, 6, 24);
            drawLine(data, 6, 14, 22, 14);
            drawLine(data, 22, 14, 22, 24);
            drawLine(data, 6, 24, 22, 24);
        },
        7 => {
            drawLine(data, 6, 4, 22, 4);
            drawLine(data, 22, 4, 14, 24);
        },
        8 => {
            drawOval(data, 8, 4, 12, 10);
            drawOval(data, 8, 14, 12, 10);
        },
        9 => {
            drawOval(data, 8, 4, 12, 10);
            drawLine(data, 20, 14, 20, 24);
        },
        else => {},
    }
}

fn drawLine(data: *[28 * 28]f32, x1: usize, y1: usize, x2: usize, y2: usize) void {
    const dx: i32 = @as(i32, @intCast(x2)) - @as(i32, @intCast(x1));
    const dy: i32 = @as(i32, @intCast(y2)) - @as(i32, @intCast(y1));
    const steps: usize = @intCast(@max(@abs(dx), @abs(dy)) + 1);

    for (0..steps) |i| {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(steps));
        const x: usize = @intFromFloat(@as(f32, @floatFromInt(x1)) + @as(f32, @floatFromInt(dx)) * t);
        const y: usize = @intFromFloat(@as(f32, @floatFromInt(y1)) + @as(f32, @floatFromInt(dy)) * t);

        if (x < 28 and y < 28) {
            setPixel(data, x, y, 1.0);
            // Add thickness
            if (x > 0) setPixel(data, x - 1, y, 0.8);
            if (x < 27) setPixel(data, x + 1, y, 0.8);
            if (y > 0) setPixel(data, x, y - 1, 0.8);
            if (y < 27) setPixel(data, x, y + 1, 0.8);
        }
    }
}

fn drawOval(data: *[28 * 28]f32, cx: usize, cy: usize, w: usize, h: usize) void {
    const steps: usize = 60;
    for (0..steps) |i| {
        const angle = @as(f32, @floatFromInt(i)) * 2.0 * std.math.pi / @as(f32, @floatFromInt(steps));
        const x: usize = @intFromFloat(@as(f32, @floatFromInt(cx)) + @cos(angle) * @as(f32, @floatFromInt(w)) / 2.0);
        const y: usize = @intFromFloat(@as(f32, @floatFromInt(cy)) + @sin(angle) * @as(f32, @floatFromInt(h)) / 2.0);

        if (x < 28 and y < 28) {
            setPixel(data, x, y, 1.0);
            if (x > 0) setPixel(data, x - 1, y, 0.7);
            if (x < 27) setPixel(data, x + 1, y, 0.7);
        }
    }
}

fn setPixel(data: *[28 * 28]f32, x: usize, y: usize, value: f32) void {
    if (x < 28 and y < 28) {
        const idx = y * 28 + x;
        data[idx] = @max(data[idx], value);
    }
}

fn loadPgmImage(path: []const u8, data: *[28 * 28]f32) !void {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var reader = file.reader();

    // Read PGM header
    var header_buf: [256]u8 = undefined;

    // Magic number (P5 for binary PGM)
    const magic = try reader.readUntilDelimiter(&header_buf, '\n');
    if (!std.mem.eql(u8, std.mem.trim(u8, magic, " \r\n"), "P5")) {
        return error.InvalidFormat;
    }

    // Skip comments
    while (true) {
        const line = try reader.readUntilDelimiter(&header_buf, '\n');
        if (line.len > 0 and line[0] != '#') {
            // Parse dimensions
            var iter = std.mem.splitScalar(u8, std.mem.trim(u8, line, " \r\n"), ' ');
            const width_str = iter.next() orelse return error.InvalidFormat;
            const height_str = iter.next() orelse return error.InvalidFormat;

            const width = try std.fmt.parseInt(usize, width_str, 10);
            const height = try std.fmt.parseInt(usize, height_str, 10);

            if (width != 28 or height != 28) {
                return error.WrongDimensions;
            }
            break;
        }
    }

    // Max value
    _ = try reader.readUntilDelimiter(&header_buf, '\n');

    // Read pixel data
    var pixel_data: [28 * 28]u8 = undefined;
    const bytes_read = try reader.readAll(&pixel_data);
    if (bytes_read != 28 * 28) {
        return error.InsufficientData;
    }

    // Convert to float [0, 1]
    for (pixel_data, 0..) |pixel, i| {
        data[i] = @as(f32, @floatFromInt(pixel)) / 255.0;
    }
}
