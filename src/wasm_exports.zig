//! WebAssembly exports for browser-based ONNX inference.
//!
//! This module provides exported functions that JavaScript can call to:
//! - Allocate and free memory
//! - Create and manipulate tensors
//! - Perform SIMD-optimized tensor operations
//!
//! The inference itself is handled by onnxruntime-web on the JavaScript side,
//! while this module provides efficient tensor preprocessing and postprocessing.

const std = @import("std");
const tensor_mod = @import("tensor.zig");

const TensorF32 = tensor_mod.TensorF32;
const Shape = tensor_mod.Shape;
const SimdOps = tensor_mod.SimdOps;
const MAX_DIMS = tensor_mod.MAX_DIMS;

// =============================================================================
// WASM Allocator
// =============================================================================

/// WebAssembly page allocator for memory management
const wasm_allocator = std.heap.wasm_allocator;

// =============================================================================
// Memory Management Exports
// =============================================================================

/// Allocate a block of memory. Returns pointer (0 on failure).
export fn wasm_alloc(size: usize) usize {
    const slice = wasm_allocator.alloc(u8, size) catch return 0;
    return @intFromPtr(slice.ptr);
}

/// Free a previously allocated block of memory.
export fn wasm_free(ptr: usize, size: usize) void {
    if (ptr == 0) return;
    const p: [*]u8 = @ptrFromInt(ptr);
    wasm_allocator.free(p[0..size]);
}

/// Allocate aligned memory for f32 data. Returns pointer (0 on failure).
export fn wasm_alloc_f32(count: usize) usize {
    const slice = wasm_allocator.alloc(f32, count) catch return 0;
    return @intFromPtr(slice.ptr);
}

/// Free f32 array.
export fn wasm_free_f32(ptr: usize, count: usize) void {
    if (ptr == 0) return;
    const p: [*]f32 = @ptrFromInt(ptr);
    wasm_allocator.free(p[0..count]);
}

// =============================================================================
// Tensor Handle System
// =============================================================================

/// Maximum number of active tensor handles
const MAX_TENSORS = 256;

/// Tensor handle storage
var tensor_storage: [MAX_TENSORS]?TensorF32 = [_]?TensorF32{null} ** MAX_TENSORS;

/// Find a free tensor slot
fn findFreeSlot() ?usize {
    for (tensor_storage, 0..) |t, i| {
        if (t == null) return i;
    }
    return null;
}

/// Validate a tensor handle and return the index, or null if invalid
inline fn validateHandle(handle: i32) ?usize {
    if (handle < 0 or handle >= MAX_TENSORS) return null;
    return @intCast(handle);
}

/// Get tensor from handle, returns null if invalid handle or no tensor
inline fn getTensor(handle: i32) ?TensorF32 {
    const idx = validateHandle(handle) orelse return null;
    return tensor_storage[idx];
}

/// Get mutable tensor pointer from handle
inline fn getTensorPtr(handle: i32) ?*TensorF32 {
    const idx = validateHandle(handle) orelse return null;
    return if (tensor_storage[idx] != null) &tensor_storage[idx].? else null;
}

// =============================================================================
// Tensor Creation Exports
// =============================================================================

/// Create a new f32 tensor with given dimensions.
/// Returns tensor handle (index) or -1 on failure.
/// shape_ptr: pointer to dimension array
/// ndim: number of dimensions (max 8)
export fn tensor_create(shape_ptr: [*]const usize, ndim: usize) i32 {
    if (ndim == 0 or ndim > MAX_DIMS) return -1;

    const slot = findFreeSlot() orelse return -1;
    const shape_slice = shape_ptr[0..ndim];

    const tensor = TensorF32.init(wasm_allocator, shape_slice) catch return -1;
    tensor_storage[slot] = tensor;

    return @intCast(slot);
}

/// Create a tensor filled with zeros.
export fn tensor_zeros(shape_ptr: [*]const usize, ndim: usize) i32 {
    return tensor_create(shape_ptr, ndim);
}

/// Create a tensor filled with ones.
export fn tensor_ones(shape_ptr: [*]const usize, ndim: usize) i32 {
    if (ndim == 0 or ndim > MAX_DIMS) return -1;

    const slot = findFreeSlot() orelse return -1;
    const shape_slice = shape_ptr[0..ndim];

    const tensor = TensorF32.ones(wasm_allocator, shape_slice) catch return -1;
    tensor_storage[slot] = tensor;

    return @intCast(slot);
}

/// Create a tensor filled with a specific value.
export fn tensor_full(shape_ptr: [*]const usize, ndim: usize, value: f32) i32 {
    if (ndim == 0 or ndim > MAX_DIMS) return -1;

    const slot = findFreeSlot() orelse return -1;
    const shape_slice = shape_ptr[0..ndim];

    const tensor = TensorF32.full(wasm_allocator, shape_slice, value) catch return -1;
    tensor_storage[slot] = tensor;

    return @intCast(slot);
}

/// Create a tensor from existing data (copies the data).
export fn tensor_from_data(data_ptr: [*]const f32, data_len: usize, shape_ptr: [*]const usize, ndim: usize) i32 {
    if (ndim == 0 or ndim > MAX_DIMS) return -1;

    const slot = findFreeSlot() orelse return -1;
    const shape_slice = shape_ptr[0..ndim];

    var tensor = TensorF32.init(wasm_allocator, shape_slice) catch return -1;
    const copy_len = @min(data_len, tensor.numel());
    @memcpy(tensor.data[0..copy_len], data_ptr[0..copy_len]);

    tensor_storage[slot] = tensor;
    return @intCast(slot);
}

/// Release a tensor and free its memory.
export fn tensor_free(handle: i32) void {
    const idx = validateHandle(handle) orelse return;
    if (tensor_storage[idx]) |*t| {
        t.deinit();
        tensor_storage[idx] = null;
    }
}

// =============================================================================
// Tensor Accessors
// =============================================================================

/// Get pointer to tensor's raw data buffer (returns address as usize, 0 on error).
export fn tensor_data_ptr(handle: i32) usize {
    const t = getTensor(handle) orelse return 0;
    return @intFromPtr(t.data.ptr);
}

/// Get total number of elements in tensor.
export fn tensor_numel(handle: i32) usize {
    const t = getTensor(handle) orelse return 0;
    return t.numel();
}

/// Get number of dimensions.
export fn tensor_ndim(handle: i32) usize {
    const t = getTensor(handle) orelse return 0;
    return t.ndim();
}

/// Get dimension at index.
export fn tensor_dim(handle: i32, dim_idx: usize) usize {
    const t = getTensor(handle) orelse return 0;
    return if (dim_idx < t.ndim()) t.shape.dims[dim_idx] else 0;
}

/// Get element at flat index.
export fn tensor_get(handle: i32, index: usize) f32 {
    const t = getTensor(handle) orelse return 0.0;
    return if (index < t.numel()) t.data[index] else 0.0;
}

/// Set element at flat index.
export fn tensor_set(handle: i32, index: usize, value: f32) void {
    const t = getTensorPtr(handle) orelse return;
    if (index < t.numel()) {
        t.data[index] = value;
    }
}

// =============================================================================
// SIMD-Optimized Operations (operate on raw arrays)
// =============================================================================

/// Element-wise add: dst = a + b
export fn simd_add(dst: [*]f32, a: [*]const f32, b: [*]const f32, len: usize) void {
    SimdOps.add(dst[0..len], a[0..len], b[0..len]);
}

/// Element-wise subtract: dst = a - b
export fn simd_sub(dst: [*]f32, a: [*]const f32, b: [*]const f32, len: usize) void {
    SimdOps.sub(dst[0..len], a[0..len], b[0..len]);
}

/// Element-wise multiply: dst = a * b
export fn simd_mul(dst: [*]f32, a: [*]const f32, b: [*]const f32, len: usize) void {
    SimdOps.mul(dst[0..len], a[0..len], b[0..len]);
}

/// Element-wise divide: dst = a / b
export fn simd_div(dst: [*]f32, a: [*]const f32, b: [*]const f32, len: usize) void {
    SimdOps.div(dst[0..len], a[0..len], b[0..len]);
}

/// Scale by constant: dst = a * scalar
export fn simd_scale(dst: [*]f32, a: [*]const f32, scalar: f32, len: usize) void {
    SimdOps.scale(dst[0..len], a[0..len], scalar);
}

/// Add constant: dst = a + scalar
export fn simd_add_scalar(dst: [*]f32, a: [*]const f32, scalar: f32, len: usize) void {
    SimdOps.addScalar(dst[0..len], a[0..len], scalar);
}

/// ReLU activation: dst = max(0, a)
export fn simd_relu(dst: [*]f32, a: [*]const f32, len: usize) void {
    SimdOps.relu(dst[0..len], a[0..len]);
}

/// Sum all elements
export fn simd_sum(a: [*]const f32, len: usize) f32 {
    return SimdOps.sum(a[0..len]);
}

/// Dot product of two vectors
export fn simd_dot(a: [*]const f32, b: [*]const f32, len: usize) f32 {
    return SimdOps.dot(a[0..len], b[0..len]);
}

/// Find maximum value
export fn simd_max(a: [*]const f32, len: usize) f32 {
    return SimdOps.max(a[0..len]);
}

/// Find minimum value
export fn simd_min(a: [*]const f32, len: usize) f32 {
    return SimdOps.min(a[0..len]);
}

/// Fused multiply-add: dst = a * b + c
export fn simd_fma(dst: [*]f32, a: [*]const f32, b: [*]const f32, c: [*]const f32, len: usize) void {
    SimdOps.fma(dst[0..len], a[0..len], b[0..len], c[0..len]);
}

/// Copy data
export fn simd_copy(dst: [*]f32, src: [*]const f32, len: usize) void {
    SimdOps.copy(dst[0..len], src[0..len]);
}

/// Fill with value
export fn simd_fill(dst: [*]f32, value: f32, len: usize) void {
    SimdOps.fill(dst[0..len], value);
}

// =============================================================================
// Tensor Operations (operate on tensor handles)
// =============================================================================

/// Add two tensors element-wise: result = a + b
/// Creates a new tensor for the result.
export fn tensor_add(a_handle: i32, b_handle: i32) i32 {
    const a = getTensor(a_handle) orelse return -1;
    const b = getTensor(b_handle) orelse return -1;
    if (a.numel() != b.numel()) return -1;

    const slot = findFreeSlot() orelse return -1;
    const result = TensorF32.initUninit(wasm_allocator, a.shape.slice()) catch return -1;
    SimdOps.add(result.data, a.data, b.data);
    tensor_storage[slot] = result;
    return @intCast(slot);
}

/// Multiply two tensors element-wise: result = a * b
export fn tensor_mul(a_handle: i32, b_handle: i32) i32 {
    const a = getTensor(a_handle) orelse return -1;
    const b = getTensor(b_handle) orelse return -1;
    if (a.numel() != b.numel()) return -1;

    const slot = findFreeSlot() orelse return -1;
    const result = TensorF32.initUninit(wasm_allocator, a.shape.slice()) catch return -1;
    SimdOps.mul(result.data, a.data, b.data);
    tensor_storage[slot] = result;
    return @intCast(slot);
}

/// Apply ReLU to tensor: result = max(0, input)
export fn tensor_relu(input_handle: i32) i32 {
    const input = getTensor(input_handle) orelse return -1;
    const slot = findFreeSlot() orelse return -1;
    const result = TensorF32.initUninit(wasm_allocator, input.shape.slice()) catch return -1;
    SimdOps.relu(result.data, input.data);
    tensor_storage[slot] = result;
    return @intCast(slot);
}

/// Scale tensor by constant: result = input * scalar
export fn tensor_scale(input_handle: i32, scalar: f32) i32 {
    const input = getTensor(input_handle) orelse return -1;
    const slot = findFreeSlot() orelse return -1;
    const result = TensorF32.initUninit(wasm_allocator, input.shape.slice()) catch return -1;
    SimdOps.scale(result.data, input.data, scalar);
    tensor_storage[slot] = result;
    return @intCast(slot);
}

/// Clone a tensor (deep copy)
export fn tensor_clone(input_handle: i32) i32 {
    const input = getTensor(input_handle) orelse return -1;
    const slot = findFreeSlot() orelse return -1;
    const cloned = input.clone(wasm_allocator) catch return -1;
    tensor_storage[slot] = cloned;
    return @intCast(slot);
}

// =============================================================================
// Softmax Implementation
// =============================================================================

/// Apply softmax to tensor along the last dimension.
/// Creates a new tensor with softmax applied.
export fn tensor_softmax(input_handle: i32) i32 {
    const input = getTensor(input_handle) orelse return -1;
    const slot = findFreeSlot() orelse return -1;
    const result = TensorF32.initUninit(wasm_allocator, input.shape.slice()) catch return -1;

    const n = input.numel();
    if (n == 0) {
        tensor_storage[slot] = result;
        return @intCast(slot);
    }

    // Find max for numerical stability
    const max_val = SimdOps.max(input.data);

    // Compute exp(x - max) and sum
    var sum: f32 = 0.0;
    for (input.data, 0..) |val, i| {
        const exp_val = @exp(val - max_val);
        result.data[i] = exp_val;
        sum += exp_val;
    }

    // Normalize
    if (sum > 0) {
        SimdOps.scale(result.data, result.data, 1.0 / sum);
    }

    tensor_storage[slot] = result;
    return @intCast(slot);
}

// =============================================================================
// Argmax Implementation
// =============================================================================

/// Find index of maximum value in tensor.
export fn tensor_argmax(input_handle: i32) i32 {
    const input = getTensor(input_handle) orelse return -1;
    if (input.numel() == 0) return -1;

    var max_idx: usize = 0;
    var max_val = input.data[0];
    for (input.data[1..], 1..) |val, i| {
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    return @intCast(max_idx);
}

// =============================================================================
// Image Preprocessing Utilities
// =============================================================================

/// Normalize image data: output[i] = (input[i] - mean) / std
/// Useful for preprocessing images before inference.
export fn normalize_image(
    dst: [*]f32,
    src: [*]const f32,
    len: usize,
    mean: f32,
    std_val: f32,
) void {
    const inv_std = 1.0 / std_val;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = (src[i] - mean) * inv_std;
    }
}

/// Convert uint8 image data to f32 (0-255 -> 0.0-1.0)
export fn uint8_to_f32(dst: [*]f32, src: [*]const u8, len: usize) void {
    const scale: f32 = 1.0 / 255.0;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = @as(f32, @floatFromInt(src[i])) * scale;
    }
}

/// Convert f32 to uint8 (0.0-1.0 -> 0-255, clamped)
export fn f32_to_uint8(dst: [*]u8, src: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const clamped = @max(0.0, @min(1.0, src[i]));
        dst[i] = @intFromFloat(clamped * 255.0);
    }
}

// =============================================================================
// Memory Info Export
// =============================================================================

/// Get current WASM memory size in pages (64KB each)
export fn wasm_memory_pages() usize {
    return @wasmMemorySize(0);
}

/// Grow WASM memory by N pages. Returns previous size or -1 on failure.
export fn wasm_memory_grow(pages: usize) isize {
    return @wasmMemoryGrow(0, pages);
}
