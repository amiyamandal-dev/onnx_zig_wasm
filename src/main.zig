const std = @import("std");
const onnx_zig = @import("onnx_zig");

pub fn main() !void {
    // Demonstrate tensor functionality
    const allocator = std.heap.page_allocator;

    var tensor = try onnx_zig.TensorF32.init(allocator, &[_]usize{ 2, 3 });
    defer tensor.deinit();

    // Fill with values
    tensor.set(&[_]usize{ 0, 0 }, 1.0);
    tensor.set(&[_]usize{ 0, 1 }, 2.0);
    tensor.set(&[_]usize{ 0, 2 }, 3.0);
    tensor.set(&[_]usize{ 1, 0 }, 4.0);
    tensor.set(&[_]usize{ 1, 1 }, 5.0);
    tensor.set(&[_]usize{ 1, 2 }, 6.0);

    std.debug.print("ONNX Zig - ONNX Inference Engine for Edge Computing\n", .{});
    std.debug.print("===================================================\n\n", .{});
    std.debug.print("Created tensor with shape: [{d}, {d}]\n", .{ tensor.shape.dims[0], tensor.shape.dims[1] });
    std.debug.print("Tensor data: ", .{});
    for (tensor.data) |v| {
        std.debug.print("{d:.1} ", .{v});
    }
    std.debug.print("\n\n", .{});

    std.debug.print("To run ONNX inference, ensure ONNX Runtime is installed:\n", .{});
    std.debug.print("  macOS:  brew install onnxruntime\n", .{});
    std.debug.print("  Linux:  Download from https://github.com/microsoft/onnxruntime/releases\n", .{});
    std.debug.print("\nBuild with: zig build -Donnxruntime_path=/path/to/onnxruntime\n", .{});
}

test "simple test" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(gpa); // Try commenting this out and see if zig detects the memory leak!
    try list.append(gpa, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}
